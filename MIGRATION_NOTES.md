# cubecl-do-now upstream rebase notes

Date: 2026-06-04

## Rebase target

- Branch rebased: `cubecl-do-now`
- Previous branch tip: `b702e2bd41f47a59a2a05e097ccda07b2af57a03`
- Safety branch: `backup/cubecl-do-now-pre-upstream-rebase-20260604`
- New base: `upstream/main` at `3d948fc4766977a6ac24c84cdd38cbfa744c0bf6`
- Preserved replay before final squash: 26 local commits plus post-rebase validation fixups

## Integration notes

- Kept upstream runtime aggregation for `ComputeClient::memory_usage()` and `memory_cleanup()` over initialized stream IDs.
- Kept CUDA graph helpers, metadata staging, graph-capture read/sync guards, upload staging ownership/flush behavior, pinned host helpers, raw async H2D copy, tensor device pointer access, and CUDA binding hardening.
- Kept upstream's zero-cube launch guard and safer CUDA kernel binding error handling.
- Adapted local line/vector indexing fixes to upstream's current `Index` / `ExtractComponent` / `InsertComponent` and `Memory::Index` APIs.
- Preserved CUDA cleanup behavior for GPU dealloc queues and pinned CPU staging pools.
- Adapted resurrected std regression tests from obsolete `ArrayArg` / `Array<T>` launch APIs to `BufferArg` and slice launch parameters.
- Kept CUDA 12.4 driver support as the compatibility floor: CUDA 12.8-only tensor-map paths remain behind `cuda_12080`, and explicit `cubecl-cuda` version feature aliases are present.

## Conflicts resolved

- `crates/cubecl-cpp/src/shared/binary.rs`
- `crates/cubecl-cuda/src/compute/context.rs`
- `crates/cubecl-cuda/src/compute/server.rs`
- `crates/cubecl-core/src/frontend/indexation.rs`
- `crates/cubecl-core/src/frontend/operation/base.rs`
- `crates/cubecl-opt/src/passes/dead_code.rs`

No commits were intentionally skipped as upstream-solved during the rebase.

## Validation

- `git range-diff backup/cubecl-do-now-pre-upstream-rebase-20260604~26..backup/cubecl-do-now-pre-upstream-rebase-20260604 upstream/main..HEAD`
- `cargo fmt --check`
- `cargo check -p cubecl-runtime -p cubecl-cpp -p cubecl-cuda -p cubecl-cpu -p cubecl-std`
- `cargo test -p cubecl-runtime persistent_pool_try_reserve_reuses_slice_with_padding`
- `cargo test -p cubecl-runtime storage_handle_offsets_clamp_without_underflow`
- `cargo test -p cubecl-cuda read_and_sync_refuse_cuda_stream_capture`
- `cargo test -p cubecl-cuda writes_flush_upload_drop_queue_when_threshold_trips`
- `cargo test -p cubecl-cuda memory_cleanup_all_cleans_pinned_cpu_staging_pools`
- `cargo test -p cubecl-cuda dynamic_line_index -- --nocapture`
- `cargo test -p cubecl-cuda array_inline_indexing -- --nocapture`
- `cargo test -p cubecl-cuda arg_binding -- --nocapture`

CUDA runtime validation ran on NVIDIA H100 with driver `550.54.15`. `nvcc` is not installed in this environment.
