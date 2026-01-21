# Array<Line<T>> Runtime Scalar Bug

## Problem Description

When using `Array<Line<T>>` with runtime scalar parameters (`line_size=1`), CUDA kernels may return all zeros due to incorrect pointer stride in generated CUDA code.

## Root Cause

The issue occurs in the CUDA code generation for array indexing operations. When `Line<f32>` with `line_size=1` is used:

1. The kernel parameter type is generated as `float*` (4-byte stride)
2. But should be `float4*` (16-byte stride) for correct alignment
3. Pointer arithmetic: `input + idx` steps by 4 bytes instead of 16
4. Results in reading from wrong memory location → zeros

### Affected Code Pattern

```rust
// This FAILS when line_size=1 (runtime parameter)
let val = input[idx][0];  // Chained indexing loses Line type
```

The intermediate `input[idx]` loses the Line type information during code generation, causing incorrect pointer types.

## Workaround 1: Helper Function (Recommended)

Use a helper function to preserve Line type through function boundaries:

```rust
#[cube]
fn load_line_element<F: Float>(input: &Array<Line<F>>, index: usize) -> F {
    let line = input[index];  // Line<F> type preserved
    line[0]
}

// In kernel:
let val = load_line_element(&input, idx);  // Works correctly!
```

**Why it works**: The explicit `let line = input[idx]` inside the function creates a named variable that preserves Line type information. The function boundary forces the compiler to materialize this intermediate value correctly.

## Workaround 2: Explicit Temporary Variable

You can also use an explicit temporary variable in your kernel:

```rust
let line = input[idx];  // Explicit temporary - Line type preserved
let val = line[0];      // Then index
```

This works similarly to the helper function by creating an intermediate named variable.

## Technical Details

### Why This Can't Be Fixed Automatically

Multiple fix attempts were made but all failed due to fundamental architectural limitations:

1. **`is_line_type` field approach**: The IR Type system returns `Type::Scalar` instead of `Type::Line` for `line_size=1`, making it impossible to distinguish `Array<Line<f32>>` from `Array<f32>` at compile time.

2. **Changing `Type::line()` method**: Always returning `Type::Line` (even for `line_size=1`) breaks local variables, which get declared as vector types (`uint4`) that don't support arithmetic operations.

3. **Context mismatch**: 
   - **Kernel arguments** need vector types: `float4* input` (for 16-byte stride)
   - **Local variables** need array types: `float tmp[4]` (for arithmetic)
   - Both use the same `Item` type, but require different code generation

### Current Status

This is a **known limitation** of the current IR Type system design. A true fix would require:
- Separate IR Type variants for Line types in different contexts
- Context-aware code generation throughout the pipeline
- Major architectural changes affecting the entire codebase

## Test Coverage

The issue is documented in `crates/cubecl-std/src/tests/array_inline_indexing.rs` with the following tests:
- `test_exact_bug_pattern_f32` - Demonstrates the bug (fails)
- `test_exact_bug_pattern_bf16` - Demonstrates the bug (fails)
- `test_with_helper_f32` - Shows helper function workaround (passes)
- `test_with_helper_bf16` - Shows helper function workaround (passes)

## Investigation Log

See `INVESTIGATION_LOG.md` at the repository root for complete investigation history and all attempted fixes.

## Impact

- **Severity**: Medium - affects specific code patterns with runtime scalar line sizes
- **Workaround**: Available and easy to apply
- **Regression**: Existing code using `line_size > 1` is unaffected
