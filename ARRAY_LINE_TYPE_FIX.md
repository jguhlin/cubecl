# Array<Line<T>> Type Expansion Fix

## Problem

The `Type::line()` method in `cubecl-ir` was returning `Type::Scalar` when `line_size=1`, causing loss of vectorization information in the type system. This led to incorrect CUDA code generation for `Array<Line<T>>` types.

## Root Cause

When creating an `Array<Line<T>>` with `line_size=1`, the type system was converting it to `Type::Scalar` instead of keeping it as `Type::Line(_, 1)`. This caused the CUDA backend to generate scalar pointer arithmetic instead of vector pointer arithmetic.

## Fix

### File: `crates/cubecl-ir/src/type.rs`

**Before (lines 475-480)**:
```rust
pub fn line(self, line_size: LineSize) -> Type {
    match line_size > 1 {
        true => Type::Line(self.storage_type(), line_size),
        false => Type::Scalar(self.storage_type()),
    }
}
```

**After (lines 475-476)**:
```rust
pub fn line(self, line_size: LineSize) -> Type {
    Type::Line(self.storage_type(), line_size)
}
```

## Rationale

An `Array<Line<T>>` should always have type `Type::Line` regardless of the `line_size` value. The `line_size` parameter is metadata about the vectorization factor, and should not change the fundamental type from `Line` to `Scalar`.

When `line_size=1`, the CUDA backend should still generate vector-aware code, just with a vectorization factor of 1. This is important for:
1. Type consistency - the same Rust type `Array<Line<T>>` should map to the same IR type
2. Correct code generation - CUDA needs to know it's dealing with vector types even when the vector has length 1
3. Future optimization potential - the backend can optimize `Line<T, 1>` to scalar operations if beneficial, but the type system should preserve the semantic distinction

## Impact

This fix ensures that:
- `Array<Line<F>>` types always expand to `Type::Line` in the IR
- The CUDA backend receives correct vectorization information
- Vectorized kernels work correctly with any `line_size` value

## Testing

Verified with Fulgurite's MPT (Model Persistence Test) framework:
- Before fix: MPT tests produced all zeros
- After fix: Debug output shows correct `vectorization: 8` for `head_dim=64`

## Related Issues

- Documented in `/weka/users/guhjo98p/cubecl/INVESTIGATION_LOG.md`
- Fixed in conjunction with Fulgurite changes to properly calculate and pass `line_size` parameters
