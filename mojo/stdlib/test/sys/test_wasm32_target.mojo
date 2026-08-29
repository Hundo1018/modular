# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

# REQUIRES: wasm-backend

# WebAssembly cross-compilation: architecture predicates, the ILP32 data
# model, and SIMD lowering. The `wasm-backend` feature gates these on a
# `mojo` whose build registered the WebAssembly backend.

# Predicates must track the triple, including the WASI OS variant.
# RUN: %bare-mojo build --target-triple=wasm32-unknown-unknown --target-cpu=generic -D EXPECT_WASM32 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=wasm32-wasi --target-cpu=generic -D EXPECT_WASM32 --emit=llvm %s -o /dev/null
# RUN: %bare-mojo build --target-triple=wasm64-unknown-unknown --target-cpu=generic -D EXPECT_WASM64 --emit=llvm %s -o /dev/null

# 32-bit data layout, and SIMD reaching LLVM IR as real vector types.
# RUN: %bare-mojo build --target-triple=wasm32-unknown-unknown --target-cpu=generic -D EXPECT_WASM32 --emit=llvm %s -o - | FileCheck %s

# With `+simd128`, vector ops must survive to actual WebAssembly SIMD
# instructions rather than being scalarized by the backend.
# RUN: %bare-mojo build --target-triple=wasm32-unknown-unknown --target-cpu=generic --target-features=+simd128 -D EXPECT_WASM32 --emit=asm %s -o - | FileCheck %s --check-prefix=CHECK_ASM

from std.ffi import c_long, c_long_long
from std.sys import is_defined, simd_bit_width
from std.sys.info import CompilationTarget, is_32bit, is_64bit, size_of

# CHECK: target datalayout = "e-m:e-p:32:32
# CHECK: target triple = "wasm32-unknown-unknown"


# The exported symbol pins down SIMD codegen: the IR check proves vector ops
# reach LLVM as vector types, and the asm check proves the WebAssembly
# backend selects simd128 instructions for them.
@export
def mul_f32x4(
    a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]
) abi("C") -> SIMD[DType.float32, 4]:
    return a * b


# CHECK-LABEL: @mul_f32x4
# CHECK: fmul {{.*}}<4 x float>

# CHECK_ASM-LABEL: mul_f32x4
# CHECK_ASM: f32x4.mul


def main():
    comptime expect_wasm32 = is_defined["EXPECT_WASM32"]()
    comptime expect_wasm64 = is_defined["EXPECT_WASM64"]()
    comptime expect_wasm = expect_wasm32 or expect_wasm64

    comptime assert (
        CompilationTarget.is_wasm() == expect_wasm
    ), "is_wasm() disagrees with the target triple"
    comptime assert (
        CompilationTarget.is_wasm32() == expect_wasm32
    ), "is_wasm32() disagrees with the target triple"
    comptime assert (
        CompilationTarget.is_wasm64() == expect_wasm64
    ), "is_wasm64() disagrees with the target triple"

    # WebAssembly is its own architecture, not any CPU ISA.
    comptime assert (
        not CompilationTarget.is_x86()
    ), "is_x86() must be False on wasm"
    comptime assert (
        not CompilationTarget.is_arm()
    ), "is_arm() must be False on wasm"
    comptime assert (
        not CompilationTarget.is_riscv()
    ), "is_riscv() must be False on wasm"

    # ILP32 on wasm32; 64-bit index width on wasm64. Both fall out of the
    # LLVM data layout, so this pins the whole width-derivation chain.
    comptime assert (
        is_32bit() == expect_wasm32
    ), "is_32bit() disagrees with the wasm pointer width"
    comptime assert (
        is_64bit() == expect_wasm64
    ), "is_64bit() disagrees with the wasm pointer width"
    comptime if expect_wasm32:
        comptime assert (
            size_of[c_long]() == 4
        ), "c_long should be 4 bytes on wasm32 (ILP32)"
        comptime assert (
            size_of[c_long_long]() == 8
        ), "c_long_long should be 8 bytes on wasm32"

    # Wasm SIMD128 gives the stdlib its default 128-bit SIMD width.
    comptime assert (
        simd_bit_width() == 128
    ), "simd_bit_width() should be 128 on wasm"
