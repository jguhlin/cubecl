# Upstream PR Plan

Source branch: `cubecl-do-now` (16 commits, rebased on `main` @ `b0fde5cf`)
Target: `tracel-ai/cubecl` main

All changes compile against current upstream (`cargo check` passes for
cubecl-runtime, cubecl-cpp, cubecl-cuda).

---

## PR 1: Memory Pool Tombstone Deallocation

**Priority:** Critical
**Branch name:** `fix/memory-pool-tombstone-dealloc`
**Strategy:** Cherry-pick from commits `9eff2131` (tombstone + handle changes) and `2db688d8` (cleanup)

**Problem:** Pool cleanup uses drain-and-rebuild, which invalidates page/slice
indices for any outstanding bindings still in the task queue. This causes
use-after-free and stale binding panics under concurrent workloads.

**Solution:** Replace `Vec<MemoryPage>` with `Vec<Option<MemoryPage>>` in all
three pool types. Deallocated entries become `None` tombstones; indices stay
stable. Trailing tombstones are trimmed to prevent unbounded growth.

**Files:**
- `cubecl-runtime/src/memory_management/memory_pool/exclusive_pool.rs`
- `cubecl-runtime/src/memory_management/memory_pool/persistent_pool.rs`
- `cubecl-runtime/src/memory_management/memory_pool/sliced_pool.rs`
- `cubecl-runtime/src/memory_management/memory_pool/handle.rs` (debug_summary, is_initialized)
- `cubecl-runtime/src/memory_management/memory_pool/memory_page.rs` (remove dead update_page)
- `cubecl-runtime/src/memory_management/memory_manage.rs` (ensure_initialized_reservation, uninitialized-binding guard, descriptor-mismatch soft check)
- `cubecl-runtime/src/server/base.rs` (IoError::Unknown format fix)

**Also includes:**
- `ensure_initialized_reservation()` helper validates every reservation returns an initialized handle
- Uninitialized-binding guard in `find()` returns NotFound instead of resolving to slot 0
- Descriptor-mismatch soft check catches stale bindings from dropped handles

**Notes:**
- Upstream's `Cell`-based MemoryLocation (from #1239) is preserved as-is
- Removes `pages_tmp` swap pattern from ExclusivePool and SlicedPool
- Removes now-dead `update_page()` from handle.rs and memory_page.rs

---

## PR 2: Warp Shuffle bf16/f16 Fix

**Priority:** High
**Branch name:** `fix/warp-shuffle-bf16-f16`
**Strategy:** Cherry-pick relevant hunks from `9eff2131`

**Problem:** `__shfl_*_sync` operates on 32-bit registers. For bf16/f16 (16-bit
types), the upper 16 bits are undefined after shuffle, corrupting subsequent
comparisons and arithmetic.

**Solution:** Add `elem: &Elem<D>` parameter to all four shuffle trait methods.
CUDA implementation casts bf16/f16 to short, promotes to int for the shuffle,
truncates back to short, and reinterprets as the original type. HIP and Metal
just add the unused parameter.

**Files:**
- `cubecl-cpp/src/shared/dialect.rs` (trait signature)
- `cubecl-cpp/src/shared/warp.rs` (pass elem to all shuffle call sites)
- `cubecl-cpp/src/cuda/dialect.rs` (bf16/f16 casting implementation)
- `cubecl-cpp/src/hip/dialect.rs` (stub `_elem` param)
- `cubecl-cpp/src/metal/dialect.rs` (stub `_elem` param)

---

## PR 3: DCE Over-Aggression Fix

**Priority:** High
**Branch name:** `fix/dce-global-input-preservation`
**Strategy:** Cherry-pick from `d7dbe9e0`

**Problem:** Dead code elimination pass removes operations whose results feed
into `GlobalInputArray` arguments, even when those operations are needed at
runtime (e.g. index computations for kernel arguments).

**Solution:** Add check in DCE pass to preserve operations that contribute to
`GlobalInputArray` dependencies.

**Files:**
- `cubecl-opt/src/passes/dead_code.rs` (+39 lines)

---

## PR 4: CUDA Graph Capture + Multi-Stream Memory Cleanup

**Priority:** Medium-High
**Branch name:** `feat/cuda-graph-capture`
**Strategy:** Cherry-pick from `ad4ea058`, `05913c1e`, and relevant parts of `9eff2131`

**Problem:** (a) Multi-threaded workloads allocate on different streams; cleaning
only one stream's pool on unload causes OOM. (b) No API for CUDA graph capture,
which is needed for inference optimization.

**Solution:**
- `memory_cleanup_all()` iterates all initialized stream pools
- `for_each_initialized()` / `for_each_stream_initialized()` stream iteration APIs
- `flush_dealloc_queue()` on MemoryManagement
- CUDA graph capture: `graph_begin_capture`, `graph_end_and_instantiate`, `graph_launch`, `graph_destroy`
- Pinned host memory: `alloc_host_pinned`, `free_host_pinned`
- H2D async copy: `htod_async_raw`
- `with_server()` accessor on ComputeClient

**Files:**
- `cubecl-cuda/src/compute/server.rs` (~280 lines added)
- `cubecl-runtime/src/client.rs` (with_server)
- `cubecl-runtime/src/stream/base.rs` (for_each_initialized)
- `cubecl-runtime/src/stream/event.rs` (for_each_stream_initialized)
- `cubecl-runtime/src/memory_management/memory_manage.rs` (flush_dealloc_queue)

**Pre-merge TODO:** Wrap 6 unsafe calls in explicit `unsafe { }` blocks
(rust-2024 `unsafe_op_in_unsafe_fn` warnings).

---

## PR 5: Indexing & Scalar Fixes

**Priority:** Medium
**Branch name:** `fix/indexing-vectorization`
**Strategy:** Cherry-pick from `0884c476`, `8d82f871`, `58b59dbe`

**Changes:**
1. **Vectorization auto-detection:** Use `find_vectorization()` instead of
   `unwrap_or(0)` in `index_expand` and `expand_index_assign_native`, so
   dynamic indexing on `Array<Vector<F,N>>` gets the correct vector size.
2. **Bool scalar support:** Store as u8 instead of panicking in `InputScalar::new()`.
3. **Const qualifier fix:** Remove incorrect `const` on `reinterpret_cast` in
   vector index-assign codegen (prevents CUDA compilation errors).

**Files:**
- `cubecl-core/src/frontend/indexation.rs`
- `cubecl-core/src/frontend/operation/base.rs`
- `cubecl-core/src/frontend/scalar.rs`
- `cubecl-cpp/src/shared/binary.rs`

---

## PR 6: CUDA Robustness & Compatibility

**Priority:** Medium-Low
**Branch name:** `fix/cuda-robustness`
**Strategy:** Cherry-pick from `9eff2131` (command.rs, stream.rs), `df8ff2ff` (TMA, WMMA), `b9daf042` (storage)

**Changes:**
- Better error messages with `debug_summary()` in `command.rs` bind/reserve_pinned
- `unwrap_or(0)` instead of `unwrap()` for `get_cursor` in stream.rs
- TMA polyfill header (`tma.cuh`) for CUDA installs without CCCL headers
- WMMA cleanup: remove unsupported fp8/fp6/fp4 MMA combinations
- CUDA version feature flags in Cargo.toml
- Storage handle underflow prevention (saturating_sub in offset calculations)

**Files:**
- `cubecl-cuda/src/compute/command.rs`
- `cubecl-cuda/src/compute/stream.rs`
- `cubecl-cuda/src/compute/server.rs` (partial - error handling only)
- `cubecl-cuda/Cargo.toml`
- `cubecl-cpp/src/cuda/ptx/tma.cuh` (new)
- `cubecl-cpp/src/cuda/ptx/mod.rs`
- `cubecl-cpp/src/cuda/mma/ptx_wmma_compiler.rs`
- `cubecl-runtime/src/storage/base.rs`
- `cubecl-runtime/src/memory_management/base.rs` (should_optimize)

---

## PR 7: Test Coverage

**Priority:** Low (submit alongside or after relevant fix PRs)
**Branch name:** `test/comprehensive-coverage`
**Strategy:** Cherry-pick from `d3a0bce9`, `92c6c358`, `b9daf042` (test parts), `d7dbe9e0` (test parts), `adf64106`

**New test modules:**
- `cubecl-std/src/tests/arg_binding_optimizer.rs` (369 lines)
- `cubecl-std/src/tests/array_inline_indexing.rs` (585 lines)
- `cubecl-std/src/tests/dynamic_line_index.rs` (174 lines)
- `cubecl-std/src/tests/tensor/contiguous.rs` (36 lines)
- `cubecl-std/src/tests/tensor/test_macros/contiguous.rs` (33 lines)
- `cubecl-std/src/tests/test_scalar_simple.rs` (15 lines)

**Supporting changes:**
- `cubecl-std/src/tests/mod.rs` (register modules)
- `cubecl-cpu/src/lib.rs` (bf16 import, test macros, API updates)
- `cubecl-cuda/src/lib.rs` (testgen_cuda macro)

**Notes:** Tests use current upstream Vector API (updated in `adf64106`).
Some tests exercise bugs fixed in PRs 3 and 5 -- consider bundling test
subsets with those PRs instead of a standalone test PR.

---

## PR 8: Documentation

**Priority:** Low (optional)
**Branch name:** `docs/known-issues`
**Strategy:** Cherry-pick from `58be16eb`

**File:** `docs/known_issues/line_type_runtime_scalar_bug.md` (91 lines)

Documents the `Array<Line<T>>` runtime scalar bug: root cause, workarounds,
why automatic fixes failed (architectural limitations), and test coverage.
Useful as a reference even if the underlying issue gets fixed.

---

## Execution Checklist

For each PR above:

1. `git checkout -b <branch-name> main`
2. Cherry-pick relevant commits (may need `git cherry-pick -n` + manual staging for partial commits)
3. `cargo check` the affected crates
4. `git push -u origin <branch-name>`
5. `gh pr create` targeting `upstream/main`

**Ordering:** PRs 1-3 are independent and can go up in parallel. PR 4 has a
soft dependency on PR 1 (uses `flush_dealloc_queue`). PRs 5-8 are fully
independent.

**After merging:** The `cubecl-do-now` branch can be rebased to drop merged
commits, or simply archived once all PRs land.
