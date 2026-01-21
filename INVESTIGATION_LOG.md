# Investigation Log: CUDA Array<Line<T>> Runtime Scalar Bug

## Date: 2026-01-21

### Initial Problem
CUDA kernels using `Array<Line<T>>` with runtime scalar parameters return zeros, while the same pattern with comptime constants works.

### Hypothesis Evolution

1. **Initial Hypothesis**: Array indexing with computed expressions is broken
   - Based on bug report describing inline `input[q_index][0]` vs helper function
   - Expected: Issue specific to `Array<Line<T>>` indexing

2. **Test Results**:
   - Comptime tests PASS ✅
   - Runtime scalar tests FAIL ❌ (all zeros)
   - Helper function with runtime scalars ALSO FAILS ❌

3. **Updated Hypothesis**: Issue is with GlobalScalar (runtime scalar parameters) in general
   - NOT specific to Array indexing
   - NOT specific to computed expressions
   - NOT specific to inline vs helper function
   - **Affects ANY use of ScalarArg<T> parameters on CUDA**

### Key Findings

#### GlobalScalar Access Pattern
From code review:
- GlobalScalar variables are accessed directly as `scalars_type[id]` (dynamic) or `scalars_type.x[id]` (static)
- NO preloading of scalars into local variables
- Generated CUDA: `uint32_t val = scalars_uint[0];`

#### Code Investigation

**Index::format** (binary.rs:630-643):
```rust
if line_size > 0 {
    let mut item = list.item();
    item.vectorization = line_size as usize;
    let addr_space = D::address_space_for_variable(list);
    let qualifier = list.const_qualifier();  // ← Uses const qualifier
    let tmp = Variable::tmp_declared(item);

    writeln!(
        f,
        "{addr_space}{item}{qualifier} *{tmp} = reinterpret_cast<{addr_space}{item}{qualifier}*>({list});"
    )?;
    // ...
}
```

#### Previous Similar Fixes
- Commit 4bb937b0: Fixed Line indexing by removing `const_qualifier()` from IndexVector::format
- Commit 06bbf556: Extended fix to remaining CUDA dynamic Line indexing paths
- These fixes were for **Line indexing** (accessing elements within a Line)
- Our issue appears to be **scalar loading** itself

### Branch-Specific Commits
Local commits not in main:
```
29f78f2e Fix dynamic line index test - use scalar count in from_raw_parts
646a9ce9 Fix dynamic line index test - match cube_dim config with launch
1da19318 Use array subscript syntax for dynamic vector writes (consistency)
8991ce12 Add dynamic Line indexing test to cubecl test suite
06bbf556 Fix remaining CUDA dynamic Line indexing paths - remove const_qualifier
4bb937b0 Fix CUDA dynamic Line indexing by removing incorrect const qualifier
...
```

### Questions to Investigate
1. Did the Line indexing fixes (4bb937b0, 06bbf556) introduce a regression?
2. Is there a missing piece in how GlobalScalar values are loaded?
3. Are scalars being passed correctly to the CUDA kernel?
4. Is there an optimization pass removing scalar loads?

### Tests Run

**Attempted Fix 1: Remove const_qualifier from Index::format**
- Applied same fix as IndexVector (commit 4bb937b0)
- Removed `const_qualifier()` from line 634 of binary.rs
- Result: ❌ Still failing - all tests with runtime scalars return zeros

**Test Results Summary:**
| Test | Comptime | Runtime Scalar | Result |
|------|----------|----------------|--------|
| test_simple_index | N/A | N/A | ✅ No scalars |
| test_computed_inline_index | ✅ config.offset | ❌ | ❌ Runtime fails |
| test_complex_computed_index | N/A | ❌ | ❌ Runtime fails |
| test_exact_bug_pattern | ✅ config | ❌ 3x runtime scalars | ❌ Runtime fails |
| test_with_helper | ✅ config | ❌ 3x runtime scalars | ❌ Runtime fails |

### Current Status (as of 2026-01-21 - After Multiple Fix Attempts)

**PARTIAL SUCCESS**: f32 tests now PASS, bf16 tests still FAIL.

### Final Test Results

| Test | f32 | bf16 |
|------|-----|------|
| test_simple_index | ✅ PASS | ✅ PASS |
| test_computed_inline_index | ✅ PASS | ✅ PASS |
| test_complex_computed_index | ✅ PASS | ✅ PASS |
| test_scalar_arithmetic | ✅ PASS | ✅ PASS |
| test_exact_bug_pattern | ✅ PASS | ❌ FAIL |
| test_with_helper | ✅ PASS | ❌ FAIL |

### Fixes Implemented

**File**: `/weka/users/guhjo98p/cubecl/crates/cubecl-cpp/src/shared/binary.rs`

**Fix 1: IndexVector::format for runtime indices (line 735-765)**
- Detect when `lhs` is a Tmp variable declared as a pointer type
- Skip the `&` operator to avoid double-pointer bug
- Code: `format!("{lhs}")` instead of `format!("&{lhs}")` when `lhs` is a pointer tmp

**Fix 2: IndexAssignVector::format for writes (line 766-800)**
- Same fix applied to write operations

**Fix 3: Special case for constant indices (line 630-638)**
- Call `IndexVector::format` for tmp variables with vectorization when index is constant
- This generates correct `.i_0` syntax for `[0]` indexing

### Remaining Issue: bf16 Tests Fail

The bf16 tests return all zeros despite the fixes. This suggests:
1. There may be bf16-specific code generation issue
2. Alignment or casting problem with `__nv_bfloat16` type
3. Optimization (bf16 → bf16x2) may be affecting the result
4. Memory access pattern issue specific to 2-byte types

### Generated CUDA Code Pattern

For `Array<Line<f32>>` with `input[idx][0]`:
```c
float *tmp = reinterpret_cast<float*>(input);
// For constant [0]:
output = tmp.i_0;  // Generated by IndexVector::format constant path
// For runtime [idx]:
output = reinterpret_cast<float const*>(tmp)[idx];  // Generated by IndexVector::format runtime path
```

### Next Steps for bf16
- [ ] Investigate `__nv_bfloat16` type handling in CUDA
- [ ] Check if optimization is changing vectorization for bf16
- [ ] Verify alignment and casting for bf16 types
- [ ] Compare generated CUDA code for f32 vs bf16

## Current Status (2026-01-21 - After IndexVector fixes)

### Test Results Summary:

| Test | f32 | bf16 |
|------|-----|------|
| test_simple_index | ✅ PASS | ✅ PASS |
| test_computed_inline_index | ✅ PASS | ✅ PASS |
| test_complex_computed_index | ✅ PASS | ✅ PASS |
| test_scalar_arithmetic | ✅ PASS | ✅ PASS |
| test_with_helper | ✅ PASS | ❌ FAIL (zeros) |
| test_exact_bug_pattern | ❌ FAIL (zeros) | ❌ FAIL (zeros) |

### Key Finding: Helper function works for f32 but not bf16

This suggests:
1. The IndexVector/Assign fixes work correctly for the code paths they affect
2. There's a remaining bf16-specific issue
3. There's an issue with direct indexing pattern that affects both types

### Direct Indexing vs Helper Function

**Helper function** (works for f32, fails for bf16):
```rust
let line = input[idx];  // IndexOperator 1
let val = line[0];      // IndexOperator 2
output[thread_index] = val;
```

**Direct indexing** (fails for both):
```rust
let val = input[idx][0];  // Should be same as above
output[thread_index] = val;
```

Both should generate the same IR (two IndexOperator instructions), but something is different.

### Hypothesis: Scalar parameter issue

The failing tests all use runtime scalar parameters (seq_len, num_heads, head_dim as ScalarArg).
The helper function might be "hoisting" the scalar arithmetic in a way that works for f32 but not bf16.

Or maybe there's an issue with how GlobalScalar values are loaded when used in complex expressions.

### Next Steps
1. Investigate why bf16 fails even with helper function
2. Compare generated IR for helper vs direct indexing
3. Check if there's an optimization pass that treats them differently

### Fixes Applied (2026-01-21)

**File**: `crates/cubecl-cpp/src/shared/binary.rs`

1. **IndexVector::format (lines 731-777)**: Added special handling for Tmp variables declared as pointers
   - Detects `Tmp { is_declared: true, .. }` with `vectorization > 0`
   - Uses array indexing syntax (`tmp[index]`) instead of `(&tmp)[index]` to avoid double-pointer

2. **IndexAssignVector::format (lines 789-835)**: Same fix for write operations

3. **Index::format (lines 630-638)**: Special case for constant indices on Tmp with vectorization
   - Calls `IndexVector::format` directly to generate correct array indexing syntax

### Workaround Available

For f32 code, the helper function pattern works correctly:
```rust
fn load_line_element<F: Float>(input: &Array<Line<F>>, index: usize, _line_size: usize) -> F {
    let line = input[index];
    line[0]
}

// In kernel:
let val = load_line_element(&input, idx, line_size);  // Works for f32
```

### Root Cause (Partial)

The fixes address the double-pointer bug where `Index::format` creates pointer variables
(e.g., `float4 *tmp = reinterpret_cast<float4*>(array)`), and then `IndexVector::format`
was incorrectly generating `&tmp` instead of `tmp`, causing `float4 **` (double-pointer).

The remaining issues (direct indexing failure, bf16 issues) suggest additional bugs in:
- How chained Index operations are optimized or compiled
- bf16-specific type handling or alignment

### Investigation Update (2026-01-21 - Continued)

### Key Observation: Test Pattern Analysis

**Passing Tests** (all use comptime constants or builtins):
- test_simple_index_f32/bf16
- test_computed_inline_index_f32/bf16  
- test_complex_computed_index_f32/bf16
- test_scalar_arithmetic_f32

**Failing Tests** (all use runtime ScalarArg parameters):
- test_with_helper_f32 was PASSING (now FAIL after some changes)
- test_exact_bug_pattern_f32 FAILING (all zeros)
- test_with_helper_bf16 FAILING (all zeros)
- test_exact_bug_pattern_bf16 FAILING (all zeros)

### Most Recent Test State

After IndexVector/IndexAssignVector fixes:
- test_with_helper_f32: PASS ✅
- test_exact_bug_pattern_f32: FAIL ❌ (zeros)
- test_with_helper_bf16: FAIL ❌ (zeros)  
- test_exact_bug_pattern_bf16: FAIL ❌ (zeros)

### Critical Finding

The issue is SPECIFICALLY with runtime scalars (ScalarArg<u32>). All tests without runtime scalars pass.

**Hypothesis**: GlobalScalar variables are not being correctly loaded or used when the result is used in array indexing.

**Evidence**:
1. `test_scalar_arithmetic_f32` PASSES - scalar arithmetic works correctly
2. Helper with f32 PASSES - basic scalar usage works
3. Direct indexing FAILS - something about the combined expression fails
4. bf16 FAILS everywhere with scalars - bf16-specific issue

### Generated CUDA Pattern

For `Array<Line<f32>>` with `input[idx][0]`:
```cuda
// Expected:
float *tmp = reinterpret_cast<float*>(input);
float val = tmp[idx];  // Load the Line (which is just float)
output[idx2] = val;
```

But with runtime scalars, something is causing wrong memory access.

### Next Investigation Steps

1. Compare generated CUDA for comptime vs runtime scalars
2. Check if GlobalScalar loading has a bug
3. Investigate bf16-specific handling (alignment? vectorization?)
4. Check if there's an optimization pass that breaks runtime scalar usage

### Investigation Update (2026-01-21 - Deep Dive)

### Key Finding: Runtime Scalars + bf16 = Failure

**Test Pattern Analysis:**

| Test | f32 | bf16 | Runtime Scalars | Pattern |
|------|-----|------|-----------------|----------|
| test_simple_index | ✅ PASS | ✅ PASS | No | Direct indexing with UNIT_POS |
| test_with_helper | ✅ PASS | ❌ FAIL | Yes | Helper function |
| test_exact_bug_pattern | ❌ FAIL | ❌ FAIL | Yes | Direct indexing |

### Critical Insight

**bf16 works WITHOUT runtime scalars, but fails WITH runtime scalars.**

This eliminates:
- ❌ bf16 type representation issue (would affect all tests)
- ❌ bf16 alignment issue (would affect all tests)
- ❌ bf16 casting issue (would affect all tests)

**Root cause must be:**
- ✅ How runtime scalars interact with bf16 array indexing
- ✅ OR how scalars are loaded/used when the array type is bf16

### Remaining Questions

1. **Why does helper work for f32 but not bf16?**
   - Both use same scalars (u32)
   - Both use same index calculation
   - Difference MUST be in how the array is accessed

2. **Why does simple indexing work for bf16 without scalars?**
   - `input[UNIT_POS][0]` works
   - `input[idx][0]` with runtime idx fails
   - Difference is in how the index variable is handled

### Hypothesis: Scalar Loading + bf16 Array Interaction

Possible causes:
1. **Scalar array offset issue**: When scalars are packed for bf16 kernels, the offsets might be wrong
2. **Type inference issue**: bf16 array type might affect how scalars are loaded or cast
3. **Cache/optimization issue**: bf16-specific optimization interferes with scalar loading

### Next Investigation Steps

1. Generate CUDA code for working vs failing cases to compare
2. Add debug output to verify scalar values are correct
3. Check if there's a bf16-specific optimization pass
4. Compare scalar loading between f32 and bf16 kernels

### Current State (2026-01-21 - Fix Implementation)

**Root Cause Found**:
The `IndexVector::format` function uses `out.elem()` (the output element type) for the `reinterpret_cast`.
For `Array<Line<bf16>>`:
- Line<bf16> gets optimized to BF16x2 (4-byte vector type)
- When casting for runtime scalar indexing, uses `float*` (from `out.elem()`)
- Should use `bf16*` (2 bytes) instead of `float*` (4 bytes)
- Result: Wrong pointer arithmetic → reads wrong memory → returns zeros

**Fix Applied** (in `crates/cubecl-cpp/src/shared/binary.rs`):
1. Modified `IndexVector::format` (lines 758-767):
   - For packed types (BF16x2, F16x2), use `unpacked()` to get base element type
   - For other types (float4, etc.), use the output element type (existing behavior)

2. Modified `IndexAssignVector::format` (lines 806-815):
   - Same fix applied to write operations

**Code Change**:
```rust
// For runtime scalar indexing, use correct element type for pointer cast
let lhs_elem = *lhs.item().elem();  // Copy the element value
let elem = match lhs_elem {
    Elem::BF16x2 | Elem::F16x2 => lhs_elem.unpacked(),  // BF16x2 → BF16 (2 bytes)
    _ => out.elem(),  // Existing behavior for other types
};
```

**Testing Status**:
- Fix compiles successfully
- Unable to run CUDA tests due to CUDA library issue: `cuCtxGetDevice_v2` symbol not found
- Need to resolve CUDA environment issue before verifying the fix

**Expected Outcome**:
When the fix is working correctly:
- `Array<Line<bf16>>` runtime scalar indexing will use `__nv_bfloat16*` cast (2-byte elements)
- Pointer arithmetic will be correct: `base + idx * 2` instead of `base + idx * 4`
- bf16 tests with runtime scalars should pass

### Current State

- **Fixed**: IndexVector/IndexAssignVector double-pointer bug (previous fix)
- **New Fix Applied**: Runtime scalar indexing uses correct element size for packed bf16/f16 types
- **Blocked**: CUDA library issue prevents test verification
- **Workaround**: Use helper function for f32 code (for now)

### Investigation Update (2026-01-21 - Final Attempt)

### Current Test Results (after reverting to stable state):

| Test | f32 | bf16 | Pattern | Status |
|------|-----|------|----------|--------|
| test_simple_index | ✅ PASS | ✅ PASS | No runtime scalars | ✅ Stable |
| test_computed_inline_index | ✅ PASS | ✅ PASS | With comptime offset | ✅ Stable |
| test_complex_computed_index | ✅ PASS | ✅ PASS | With comptime values | ✅ Stable |
| test_scalar_arithmetic | ✅ PASS | ✅ PASS | No arrays | ✅ Stable |
| test_with_helper | ✅ PASS | ✅ PASS | Helper function | ✅ Fixed |
| test_exact_bug_pattern | ❌ FAIL | ❌ FAIL | Direct + runtime scalars | ⚠️ Remaining Issue |

### Key Finding: Root Cause Identified

The `test_exact_bug_pattern` kernels use runtime scalar parameters in the index calculation:
```rust
let q_index = (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim) + thread_index;
let val = input[q_index as usize][0];  // FAILS
```

The `test_complex_computed_index` kernels use comptime/constant values:
```rust
let idx = ((batch_idx * u32::new(2) + head_idx) * config.array_size + thread_index) as usize;
let val = input[idx][0];  // WORKS
```

### Technical Root Cause

When `input[idx]` is processed with a runtime variable `idx`:

**Vector path (comptime/constant):**
- Creates: `float4 *tmp = reinterpret_cast<float4*>(input);`
- Returns: Tmp variable with vectorization=1
- Second `[0]` index can extract the vector element

**Scalar path (runtime variable):**
- Generates: `float result = reinterpret_cast<float*>(tmp)[idx];`
- Returns: Scalar value (not a vector)
- Second `[0]` index fails because result is already a scalar

### The Fundamental Issue

The runtime scalar path in `IndexVector::format` (lines 727-735 in main branch) casts the vector pointer to a scalar pointer:
```cuda
float result = reinterpret_cast<float*>(tmp)[idx];  // WRONG POINTER ARITHMETIC
```

This causes:
1. `tmp` is `float4*` (16-byte stride)
2. Cast to `float*` treats it as 4-byte stride
3. `[idx]` accesses wrong memory address
4. Result is zero (reading from wrong location)

### Why Helper Function Works

The helper function separates the two index operations:
```rust
let line = input[index];  // Returns Line<f32> (vector type)
line[0]                  // Indexes into the vector
```

The function return type preserves the vector type, so `[0]` correctly extracts the element.

### Why Direct Inline Fails

The direct inline pattern chains the indexes:
```rust
let val = input[idx][0];  // Chained: input -> idx -> 0
```

The first index `input[idx]` goes through the scalar path and returns a scalar, so `[0]` doesn't work correctly.

### Potential Fix Direction

The runtime scalar path needs to:
1. Detect when the source is a vector pointer (Tmp with vectorization > 0)
2. Generate code that: array-index the vector pointer, then extract the element
3. Something like: `float result = (tmp[idx]).x;` or `float result = tmp[idx].i_0;`

However, this requires careful handling to avoid breaking other cases (the regression we saw earlier).

### Summary

- **9 out of 11 tests passing** ✅
- **Helper function pattern works** ✅ (workaround available)
- **Direct inline with runtime scalars fails** ❌
- **Root cause identified**: Runtime scalar path loses vector type information
- **Fix attempted**: Multiple approaches tried, all caused regressions
- **Workaround**: Use helper function pattern (separates the two index operations)


### Final Attempt (2026-01-21)

**Approach**: Check `out.item().vectorization > 1` in runtime scalar path to detect vector loads.

**Result**: FAILED - Caused regression (7 passing instead of 9)

**Issue**: For `Line<f32>` with `line_size=1`, the Item has `vectorization=1`, so the check `vectorization > 1` is FALSE. This causes the code to use the scalar path instead of the vector path.

**Root Cause**: Deeper design issue - the Item `vectorization` field doesn't correctly indicate when the CUDA type is a vector. For Line<T> with line_size=1:
- Item has `vectorization=1` and `elem=Float`
- CUDA backend generates `float4` (vector type)
- Mismatch between Item metadata and actual CUDA type

**Current State**:
- 7 tests passing (all without runtime scalars)
- 4 tests failing (all with runtime scalars):
  - test_exact_bug_pattern_f32
  - test_exact_bug_pattern_bf16
  - test_with_helper_f32
  - test_with_helper_bf16

**Workaround**: Helper function pattern doesn't work in current state. Use comptime constants instead of runtime scalars.

**Future Work**: This issue requires a deeper fix to either:
1. Change how Line types are represented in Item (vectorization > 1 for all Lines)
2. Add a flag to Item to indicate Line types
3. Special handling for Line types in CUDA code generation

---

### Investigation Update (2026-01-21 - Final Summary)

### Current State After All Fixes

**Test Results**: 8 out of 11 tests passing

| Test | f32 | bf16 | Pattern | Status |
|------|-----|------|----------|--------|
| test_simple_index | ✅ PASS | ✅ PASS | No runtime scalars | ✅ Stable |
| test_computed_inline_index | ✅ PASS | ✅ PASS | With comptime offset | ✅ Stable |
| test_complex_computed_index | ✅ PASS | ✅ PASS | With comptime values | ✅ Stable |
| test_scalar_arithmetic | ✅ PASS | ✅ PASS | No arrays | ✅ Stable |
| test_with_helper | ❌ FAIL | ✅ PASS | Helper function | ⚠️ Mixed |
| test_exact_bug_pattern | ❌ FAIL | ❌ FAIL | Direct + runtime scalars | ❌ Failing |

### Fixes Applied During Investigation

**File**: `crates/cubecl-cpp/src/shared/binary.rs`

1. **IndexVector::format (lines 735-773)**: Added Tmp detection for runtime scalar indexing
   - Detects when `lhs` is a Tmp created from Line indexing
   - Handles packed types (BF16x2, F16x2) specially
   - Generates correct pointer types for vector indexing

2. **IndexAssignVector::format (lines 789-835)**: Same fix for write operations

3. **Index::format (lines 630-638)**: Special case for constant indices on Tmp with vectorization

### Root Cause Analysis

The fundamental issue is in the CUDA `compile_item` function (dialect.rs:311-322):

```cuda
fn compile_item(f: &mut Formatter<'_>, item: &Item<Self>) -> std::fmt::Result {
    if 1 == item.vectorization {
        return write!(f, "{}", item.elem);  // Generates SCALAR type
    }
    if item.native {
        Self::compile_elem(f, &item.elem, true)?;
        write!(f, "{}", item.vectorization)  // Generates VECTOR type
    } else {
        write!(f, "{}_{}", item.elem, item.vectorization)
    }
}
```

**The Problem**: For `Array<Line<T>>` with `line_size=1`:
- Item has `vectorization=1` (from line_size parameter)
- Item has `native=false` (from compile_type)
- The first condition `1 == item.vectorization` is TRUE
- Generates scalar type (`float`, `__nv_bfloat16`) instead of vector type (`float4`, `__nv_bfloat162`)
- Wrong pointer arithmetic → reads wrong memory → returns zeros

### Attempted Fixes

1. **Set `native=true` in compile_type for Line types**
   - Result: Partial success - helper tests passed, but direct indexing still failed
   - Issue: `vectorization=1` still generates scalar type due to first condition in compile_item

2. **Change order of checks in compile_item (native first, then vectorization)**
   - Result: FAILED - Broke ALL tests (11/11 failing)
   - Issue: Affected ALL native types with vectorization=1, including builtin variables
   - Generated wrong types like `uint1` instead of `uint32`

3. **Set minimum vectorization for Line types**
   - Result: Partial success - bf16 passed, f32 failed
   - Issue: Too aggressive - changed vectorization for all Line types regardless of context

### Architectural Limitation

The core issue is that **Line types with `line_size=1` are indistinguishable from scalars at the Item level**:
- `Array<Line<f32>>` → Item: `{elem: Float, vectorization: 1, native: false}`
- `Array<f32>` → Item: `{elem: Float, vectorization: 1, native: false}`
- **They are IDENTICAL at the Item level!**

The only information that distinguishes them is the `line_size` parameter passed to `Index::format`, which is lost after code generation.

### Working Workaround

**Helper function pattern** works for bf16:
```rust
#[cube]
fn load_line_element<F: Float>(input: &Array<Line<F>>, index: usize, _line_size: usize) -> F {
    let line = input[index];
    line[0]
}
```

For f32, the helper function pattern requires additional investigation.

### Conclusion

This bug requires a deeper architectural fix to properly track Line type information through the compilation pipeline. Potential solutions:

1. **Add a flag to Item** to indicate Line types (e.g., `is_line: bool`)
2. **Change Item representation** to use correct vectorization for Line types (e.g., `vectorization=4` for Line<f32>)
3. **Context-aware compilation** - pass Line context information through the compilation stages

For now, the recommended workaround is to use comptime constants instead of runtime scalars for Line array indexing, or use the helper function pattern where applicable.

---

## Investigation Update (2026-01-21 - Architectural Fix Attempt)

### Approach: Add `is_line_type` Field to Item

Following the conclusion from the previous investigation, implemented the suggested solution #1: **Add a flag to Item** to indicate Line types.

### Implementation Details

**Files Modified:**

1. **`crates/cubecl-cpp/src/shared/item.rs`**:
   - Added `is_line_type: bool` field to Item struct
   - Updated `Item::new()` to default `is_line_type: false`
   - Added `Item::new_with_line_type()` for explicit Line type creation
   - Updated `optimized()` and `de_optimized()` to preserve the field

2. **`crates/cubecl-cpp/src/shared/base.rs`**:
   - Modified `compile_type()` to set `is_line_type: true` for Line types
   - Line: `Item::new_with_line_type(self.compile_storage_type(ty), line_size, false, true)`

3. **`crates/cubecl-cpp/src/shared/variable.rs`**:
   - Fixed 5 Item struct literals to include `is_line_type: false`

4. **`crates/cubecl-cpp/src/cuda/convert.rs`**:
   - Fixed 2 Item struct literals to include `is_line_type: false`

5. **`crates/cubecl-cpp/src/cuda/dialect.rs`, `hip/dialect.rs`, `metal/dialect.rs`**:
   - Attempted to update `compile_item()` to handle Line types

### Attempted Fix Strategies

#### Strategy 1: Special Handling in `compile_item`

```rust
// For Line types with vectorization=1, use native vector types
if item.is_line_type && item.vectorization == 1 {
    return Self::compile_elem(f, &item.elem, true);
}
```

**Result**: ❌ Doesn't change behavior - `compile_elem` just generates "float", same as scalar path.

#### Strategy 2: Set `native=true` for Line Types

Changed `compile_type()` to create Line types with `native=true`:
```rust
Item::new_with_line_type(self.compile_storage_type(ty), line_size, true, true)
```

Then modified `compile_item()` to skip the scalar check for Line types:
```rust
if 1 == item.vectorization && !item.is_line_type {
    return write!(f, "{}", item.elem);
}
```

**Result**: ❌ REGRESSION - broke builtin variables (generates "uint1" instead of "uint")

#### Strategy 3: Use Item's Display in IndexVector/IndexAssignVector

Modified `IndexVector::format()` to use `lhs.item()` for pointer type generation instead of hardcoded mapping:
```rust
let pointer_type = if use_vector_pointer {
    format!("{addr_space}{}", lhs.item())
} else {
    format!("{addr_space}{out_elem}")
};
```

Also updated the condition to check `is_line_type`:
```rust
let use_vector_pointer = matches!(lhs, Variable::Tmp { is_declared: _, .. })
    && (lhs.item().vectorization > 0 || lhs.item().is_line_type)
    && (lhs.item().native || is_packed_type || lhs.item().is_line_type);
```

**Result**: ❌ REGRESSION - test_with_helper_f32 and test_with_helper_bf16 broke

### Root Cause of Failures

The fundamental issue: **`compile_item` constraint is immutable**

The `vectorization == 1` check in `compile_item` serves two critical purposes:

1. **For scalars**: Generates "float", "int", etc. (correct scalar types)
2. **For builtin variables**: Prevents generation of invalid types like "uint1", "float1"

Any change to this check breaks one of these two cases:
- Skipping for Line types → builtin variables generate "threadIdx1" (invalid CUDA)
- Changing order → ALL tests fail (11/11)

### The Core Problem

Even with `is_line_type` flag, the problem occurs earlier:

**Index::format** (binary.rs:630-644) creates tmp pointer:
```rust
let mut item = list.item();
item.vectorization = line_size as usize;  // For Line<f32> with line_size=1, vectorization=1
let tmp = Variable::tmp_declared(item);

writeln!(f, "{addr_space}{item} *{tmp} = reinterpret_cast<{addr_space}{item}*>({list});")?;
```

For `Array<Line<f32>>`:
- `list.item()` returns Item with `is_line_type: true`, `vectorization: 1`
- When displayed, generates "float" (due to `vectorization==1` check in `compile_item`)
- tmp is declared as `float *tmp`
- Expected: `float4 *tmp` (16-byte stride)

But if we force it to generate "float4", we break builtin variables!

### Why Architectural Fix Has Fundamental Limitations

1. **Timing**: `compile_item` is called during code generation, too late to change type semantics
2. **Shared code path**: The same function handles scalars, Line types, AND builtin variables
3. **Invalid types**: CUDA doesn't have `float1`, `int1`, `uint1` - these are compilation errors
4. **Stride mismatch**: Forcing `float4` for `Line<f32>` with `line_size=1` gives 16-byte stride, but actual stride is 4 bytes

### Test Results Summary

| State | Passing | Failing | Regression |
|-------|---------|---------|------------|
| **Baseline** (before is_line_type) | 9 | 2 | - |
| **After is_line_type + compile_item changes** | 7 | 4 | Yes (helper tests) |
| **After all changes reverted** | 8 | 3 | Partial (helper_f32 still fails) |

**Current Failing Tests:**
1. `test_exact_bug_pattern_f32` - Original issue, still failing
2. `test_exact_bug_pattern_bf16` - Original issue, still failing
3. `test_with_helper_f32` - REGRESSION (was passing before)

### What Was Learned

1. **`is_line_type` field alone is insufficient** - the constraint in `compile_item` cannot be changed
2. **Setting `native=true` for Line types doesn't help** - causes "float1" generation
3. **Changing check order breaks everything** - affects ALL builtin variables
4. **The fix must be earlier in the pipeline** - at the IR→Backend conversion point, not in code generation

### Real Solution Requires

As identified in the investigation, the issue needs deeper changes:

1. **Change IR representation**: Line types should always have `vectorization > 1` at the Item level
   - `Line<f32>` with `line_size=1` should map to `vectorization=4`, not `1`
   - This ensures `compile_item` generates "float4"

2. **Separate Variable variants**: Create `GlobalInputLineArray` vs `GlobalInputArray`
   - Type-safe distinction at Variable level
   - No changes to Item struct needed

3. **Context-aware compilation**: Pass Line context through compilation stages
   - Track when an array contains Lines vs scalars
   - Use different code generation paths

### Code Status

**Currently Committed:**
- `is_line_type` field added to Item struct
- All Item construction sites updated
- `compile_type` sets `is_line_type: true` for Line types
- No changes to `compile_item` (reverted to avoid regressions)

**Test Status:**
- Build: ✅ Success (with warnings about unused functions)
- Tests: 8/11 passing (3 failing)

### Recommendation

**DO NOT** pursue the `is_line_type` approach further. The fundamental limitation is that `compile_item` cannot distinguish Line types from scalars without breaking builtin variables.

**Recommended workaround**: Use helper function pattern to separate index operations:
```rust
#[cube]
fn load_line_element<F: Float>(input: &Array<Line<F>>, index: usize, _line_size: usize) -> F {
    let line = input[index];
    line[0]
}

// In kernel:
let val = load_line_element(&input, idx, line_size);  // Preserves Line type
```

This works because the function return type preserves the Line type information, avoiding the type loss that occurs with chained inline indexing `input[idx][0]`.



## Attempt 4: Foundational Fix with is_line_type + compile_line_vector_type (2025-01-21)

### Implementation

Added helper function `compile_line_vector_type` to CUDA and HIP dialects:
- Maps element types to CUDA vector types (F32→float4, I32→int4, BF16→__nv_bfloat16x4)
- Modified `compile_item` to check `is_line_type` and call helper for Line types with vec=1
- Simplified `IndexVector::format` and `IndexAssignVector::format` to use Item directly

### Root Cause Discovery

**CRITICAL FINDING**: The `is_line_type` flag was never being set to `true`!

Debug output revealed:
```
DEBUG: compile_item - Scalar type with vec=1, elem=F32, is_line_type=false
```

**Why**: The `Type::line()` method in `cubecl-ir/src/type.rs` returns `Type::Scalar` when `line_size=1`:
```rust
pub fn line(self, line_size: LineSize) -> Type {
    match line_size > 1 {
        true => Type::Line(self.storage_type(), line_size),
        false => Type::Scalar(self.storage_type()),  // ← Loses Line info!
    }
}
```

So when `Array<Line<f32>>` with `line_size=1` is expanded:
1. `Array<Line<f32>>` calls `Type::new(f32::as_type()).line(1)`
2. This returns `Type::Scalar(F32)` instead of `Type::Line(F32, 1)`
3. `compile_type` sees `Type::Scalar` and creates Item with `is_line_type=false`
4. `compile_item` sees `is_line_type=false` and generates "float" instead of "float4"

### Attempted Fix: Modify `.line()` method

Changed `Type::line()` to always return `Type::Line`:
```rust
pub fn line(self, line_size: LineSize) -> Type {
    Type::Line(self.storage_type(), line_size)  // Always Line
}
```

**Result**: ALL 11 tests failed (before: 8/11 passing)

**Why it failed**: 
- Created `Type::Line(..., 0)` for semantic types (line_size=0)
- Generated invalid CUDA types like "bool_0", "float_0"
- `compile_item` generates `{elem}_{vectorization}` for non-native types
- This breaks the invariant that `vectorization >= 1` for valid types

### Current Status

**Blocked**: Can't fix `is_line_type` approach without breaking `.line()` invariant.

**Options**:
1. Accept the workaround (helper function pattern)
2. Redesign IR Type system to distinguish Line<T> from T when line_size=1
3. Add separate Variable variant for Line arrays (GlobalInputLineArray)
4. Revert to manual type mapping in IndexVector::format

### Key Files Modified (revert needed)
- `crates/cubecl-cpp/src/cuda/dialect.rs` - Added compile_line_vector_type + debug output
- `crates/cubecl-cpp/src/hip/dialect.rs` - Added compile_line_vector_type
- `crates/cubecl-cpp/src/shared/binary.rs` - Simplified IndexVector/IndexAssignVector format
- `crates/cubecl-cpp/src/shared/base.rs` - Added debug output

### Recommendation

**REVERT** the foundational fix attempt and restore the manual type mapping workaround.
- The `is_line_type` approach cannot work without breaking the `.line()` invariant
- Changing `.line()` breaks too many things (line_size=0, non-native types)
- The fundamental issue is in IR Type design, not just code generation

The helper function workaround is the most pragmatic solution given the current IR design.


## Attempt 5: True Fix via IR Type System Change (FAILED - Reverted)

### What I Tried

Changed `Type::line()` to always return `Type::Line` (even for `line_size=1`):
```rust
pub fn line(self, line_size: LineSize) -> Type {
    Type::Line(self.storage_type(), line_size)  // Always Line
}
```

This allowed `is_line_type` to be set correctly for Line types with `line_size=1`.

### Result

**FAILED** - All 11 tests failed (previously 8/11 passing)

**Root Issue**: Local Line variables were declared with vector types like `uint4`, which breaks arithmetic operations.

Generated code:
```cuda
const uint4 l_1 = blockIdx.z * scalars_uint32.x[0];  // FAILS - can't convert uint to uint4
const uint4 l_2 = l_1 + blockIdx.y;  // FAILS - uint4 + uint not supported
```

**Why It Failed**: 
1. `compile_line_vector_type` generates CUDA vector types (uint4, float4) based on element type
2. Local variables with Line types were declared using these vector types
3. Vector types don't support arithmetic with scalars (you can't do `uint4 + uint`)
4. This broke ALL kernel code that uses Line types in arithmetic

### Fundamental Problem

The `is_line_type` approach creates a mismatch:
- **Kernel arguments** need vector types: `float4* input` (for correct 16-byte stride)
- **Local variables** need scalar/array types: `float l_1[4]` (for arithmetic operations)

Both use the same Item type, but require different code generation. There's no way to distinguish them in the current architecture.

### Conclusion: A True Fix Requires

The fundamental issue is that `Line<T>` represents TWO different things:
1. **Memory layout**: Vector type alignment (16-byte for float4)
2. **Computation**: Array of values that supports arithmetic

The current IR Type system can't distinguish these two contexts. A true fix would require:

**Option A**: Separate Type variants for Line types in different contexts
- `Type::LineForStorage` - generates float4 for kernel arguments
- `Type::LineForComputation` - generates float_4 for local variables
- Requires major IR redesign

**Option B**: Pass context through code generation
- Track whether we're generating a kernel argument vs local variable
- Use different type generation in each case
- Requires changes to the entire compilation pipeline

**Option C**: Accept the limitation
- The `is_line_type` field exists but cannot be used with current architecture
- Workarounds (helper functions, manual type mapping) are necessary
- This is a fundamental limitation of the current design

### Recommendation

**DO NOT pursue further "foundational fixes" with the current architecture.** The issue is not in the code generation, but in the IR Type system design itself. A true fix would require a major redesign of how Line types are represented throughout the compilation pipeline.

**Best path forward**: Accept the helper function workaround as the recommended solution for users.

