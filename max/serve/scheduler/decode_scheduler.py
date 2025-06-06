# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import logging
import queue
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import cast

import zmq
from max.nn.kv_cache import PagedKVCacheManager
from max.pipelines.core import (
    InputContext,
    TextGenerationResponse,
    TextResponse,
    TokenGenerator,
)
from max.serve.process_control import ProcessControl

from .base import Scheduler
from .queues import STOP_STREAM
from .zmq_queue import ZmqPullSocket, ZmqPushSocket

logger = logging.getLogger("max.serve")


@dataclass
class DecodeSchedulerConfig:
    """Decode Specific Scheduler Config."""

    max_batch_size_tg: int
    """The maximum number of requests that can be in the token generation batch."""

    max_forward_steps_tg: int
    """The number of tokens to generate for each request in the token generation iteration."""


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        process_control: ProcessControl,
        pipeline: TokenGenerator,
        scheduler_config: DecodeSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        prefill_zmq_endpoint: str,
        decode_zmq_endpoint: str,
        zmq_ctx: zmq.Context,
    ):
        # Initialize Pipeline and Config
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.paged_manager = paged_manager
        self.zmq_ctx = zmq_ctx

        # Multiprocessing resources.
        self.pc = process_control

        # Initialize Queues
        self.request_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx, zmq_endpoint=request_zmq_endpoint
        )
        self.response_push_socket = ZmqPushSocket[tuple[str, TextResponse]](
            zmq_ctx=zmq_ctx, zmq_endpoint=response_zmq_endpoint
        )
        self.cancel_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=cancel_zmq_endpoint
        )

        self.decode_pull_socket = ZmqPullSocket[tuple[str, InputContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=decode_zmq_endpoint
        )
        self.prefill_push_socket = ZmqPushSocket[tuple[str, InputContext]](
            zmq_ctx=zmq_ctx, zmq_endpoint=prefill_zmq_endpoint
        )

        self.preempted_decode: queue.Queue[tuple[str, InputContext]] = (
            queue.Queue()
        )

        # Initialize Scheduler state.
        self.active_batch: OrderedDict[str, InputContext] = OrderedDict()
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_tg)
        )

    def pull_from_request_socket(self) -> tuple[str, InputContext]:
        """Pulls a request from the request socket.

        Returns:
            tuple[str, InputContext]: A tuple containing the request ID and input context.

        Raises:
            queue.Empty: If no requests are available.
            zmq.ZMQError: If there is an error receiving from the socket.
        """
        return self.request_pull_socket.get_nowait()

    def pull_from_decode_socket(self) -> tuple[str, InputContext]:
        """Pulls a request from the decode socket, checking preempted requests first.

        Returns:
            tuple[str, InputContext]: A tuple containing the request ID and input context.

        Raises:
            queue.Empty: If no requests are available.
            zmq.ZMQError: If there is an error receiving from the socket.
        """
        # First try and return from pre-empted requests queue.
        if not self.preempted_decode.empty():
            return self.preempted_decode.get()

        return self.decode_pull_socket.get_nowait()

    def push_to_response_socket(
        self, responses: list[dict[str, TextResponse]] = [{}]
    ) -> None:
        """Pushes response messages to the response socket.

        Args:
            responses: List of response dictionaries to send, defaults to empty dict.

        Raises:
            zmq.ZMQError: If there is an error sending on the socket.
        """
        self.response_push_socket.put_nowait(responses)

    def push_to_prefill_socket(
        self,
        request_id: str,
        data: InputContext,
    ) -> None:
        """Pushes a request to the prefill socket.

        Args:
            request_id: The ID of the request to send
            data: The InputContext containing the request data

        Raises:
            zmq.ZMQError: If there is an error sending on the socket
        """
        self.prefill_push_socket.put_nowait((request_id, data))

    def reserve_memory_and_send_to_prefill(self) -> None:
        """Continuously pulls requests from the request queue and forwards them to the prefill node.

        Breaks when the request queue is empty. Memory reservation is pending implementation.
        """
        # TODO: E2EOPT-219 - Eagerly reserve memory prior to sending to prefill.
        while True:
            try:
                # Pop off request queue
                new_request_id, new_request_data = (
                    self.pull_from_request_socket()
                )

                # Send to the Prefill Node
                self.push_to_prefill_socket(new_request_id, new_request_data)

            except:
                # Break loop when no items in queue
                break

    def return_to_decode_queue(self, request_id: str, data: InputContext):
        """Resets a request and returns it to the preempted decode queue rather than directly
        to the decode socket. This allows preempted requests to be retried later when resources
        become available.

        Args:
            request_id: The ID of the request to return
            data: The InputContext containing the request data
        """
        self.available_cache_indices.add(data.cache_seq_id)
        self.pipeline.release(data)
        data.reset()
        self.preempted_decode.put((request_id, data))

    def update_batch(self) -> None:
        """Updates the active batch by adding new requests from the decode queue and managing memory prefetching.

        Adds new requests to the batch while cache indices are available. For each request, attempts to prefetch
        required memory. If prefetch fails, handles preemption by returning newer requests to the decode queue.
        """
        # Add new items to batch.
        while self.available_cache_indices:
            try:
                # Retrieve new item from the decode queue.
                request_id, context = self.pull_from_decode_socket()

                # Assign to cache.
                if not context.is_assigned_to_cache:
                    context.assign_to_cache(self.available_cache_indices.pop())
                    self.paged_manager.external_claim([context.cache_seq_id])

                # Add to active batch.
                self.active_batch[request_id] = context

            except queue.Empty:
                # Break this loop when the decode queue is empty.
                break
            except Exception as e:
                logger.error(e)
                raise e

        # We can assume that no item in the active batch is complete.
        candidate_requests = deque(self.active_batch.keys())
        while len(candidate_requests):
            # Grab the first request.
            request_id = candidate_requests.popleft()
            context = self.active_batch[request_id]

            # Calculate number of maximum available forward steps.
            num_available_steps = context.compute_num_available_steps(
                self.paged_manager.max_seq_len
            )
            num_steps = (
                self.scheduler_config.max_forward_steps_tg
                if self.scheduler_config.max_forward_steps_tg
                < num_available_steps
                else num_available_steps
            )

            if not self.paged_manager.prefetch(context, num_steps):
                # If there are no outstanding candidate requests.
                # Add this candidate back to the request queue,
                # remove from the active batch and continue.
                if len(candidate_requests) == 0:
                    self.return_to_decode_queue(request_id, context)
                    del self.active_batch[request_id]
                    break

                # Remove the newest request from the active batch.
                newest_request_id = candidate_requests.pop()
                newest_context = self.active_batch.pop(newest_request_id)

                # Preempt the newest candidate back to the request queue
                # and try to prefetch again.
                self.return_to_decode_queue(newest_request_id, newest_context)

                candidate_requests.appendleft(request_id)

    def calculate_batch_num_steps(self) -> int:
        """Calculate the number of steps to process in the current batch.

        Returns:
            int: Number of steps to process, either max_forward_steps_tg or a smaller value
                based on request max_lengths.

        Raises:
            RuntimeError: If active_batch is empty.
        """
        if not self.active_batch:
            raise RuntimeError(
                "active_batch must contain at least one context to calculate num_steps"
            )

        # Calculate the maximum number of steps for an individual context.
        batch_available_steps = -1
        for data in self.active_batch.values():
            # If any request has no max_length, we should not change num_steps.
            if data.max_length is None:
                return self.scheduler_config.max_forward_steps_tg

            request_available_steps = data.compute_num_available_steps(
                data.max_length
            )
            if request_available_steps > batch_available_steps:
                batch_available_steps = request_available_steps

        if (
            batch_available_steps > 0
            and batch_available_steps
            < self.scheduler_config.max_forward_steps_tg
        ):
            return batch_available_steps

        return self.scheduler_config.max_forward_steps_tg

    def stream_responses_to_frontend(
        self, responses: dict[str, TextGenerationResponse]
    ) -> None:
        """Streams text generation responses to the frontend by converting them into a format suitable for streaming.

        Args:
            responses: Dictionary mapping request IDs to their text generation responses.
        """
        if not responses:
            return

        # Convert this to list[dict[str, Any]]
        stream_responses: list[dict[str, TextResponse]] = [{}]
        for request_id, response in responses.items():
            # This will just ensure that there is always a response for each token
            # We add one here, as we need to send a stop sentinel
            while (len(response.tokens) + (1 if response.is_done else 0)) > len(
                stream_responses
            ):
                stream_responses.append({})

            for token_idx, text_response in enumerate(response.tokens):
                stream_responses[token_idx][request_id] = text_response

            if response.is_done:
                stream_responses[len(response.tokens)][request_id] = cast(
                    TextResponse, STOP_STREAM
                )

        self.push_to_response_socket(stream_responses)

    def _handle_terminated_responses(
        self, responses: dict[str, TextGenerationResponse]
    ) -> None:
        """Handles cleanup for completed text generation responses by releasing cache and removing from active batch.

        Args:
            responses: Dictionary mapping request IDs to their text generation responses.
        """
        if responses is None:
            return

        for request_id, response in responses.items():
            if response.is_done:
                # Release from cache, and active batch.
                cache_id = self.active_batch[request_id].cache_seq_id
                self.pipeline.release(self.active_batch[request_id])
                self.available_cache_indices.add(cache_id)
                del self.active_batch[request_id]

    def schedule_batch(self, num_steps: int):
        """Schedules a batch of requests for token generation and handles the responses.

        Args:
            num_steps: Number of tokens to generate for this batch.
        """
        responses = self.pipeline.next_token(
            self.active_batch, num_steps=num_steps
        )

        self._handle_terminated_responses(responses)
        self.stream_responses_to_frontend(responses)

    def run(self) -> None:
        """Main scheduling loop that processes decode requests.

        Continuously receives requests, updates batches, and schedules them for processing
        while handling memory management. The loop continues until the process is cancelled.
        """
        while not self.pc.is_canceled():
            # Indicate that the process is still alive.
            self.pc.beat()

            # Eagerly reserve memory and send to prefill worker
            self.reserve_memory_and_send_to_prefill()

            # Update the active decode batch
            self.update_batch()

            # If empty, skip
            if not self.active_batch:
                continue

            # Calculate num_steps
            num_steps = self.calculate_batch_num_steps()

            # Schedule Batch
            self.schedule_batch(num_steps)
