// Minimal polyfills for CUDA TMA / cp.async.bulk.* wrappers.
//
// Some CUDA installations ship <cuda/barrier> without the CCCL experimental
// wrappers used by CubeCL codegen. Provide a small subset of these wrappers so
// kernels can compile with NVRTC even when the headers are missing them.

#if !defined(__cccl_lib_experimental_ctk12_cp_async_exposure)

namespace cuda {
namespace device {
namespace experimental {

inline __device__ void fence_proxy_async_shared_cta() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

inline __device__ void cp_async_bulk_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.commit_group;" :::);
#endif
}

template <int N32>
inline __device__ void cp_async_bulk_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N32) : "memory");
#endif
}

template <int N32>
inline __device__ void cp_async_bulk_wait_group_read() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N32) : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_1d_global_to_shared(
    void *smem_ptr, const CUtensorMap *__tensor_map, int32 const &coord0,
    ::cuda::barrier<::cuda::thread_scope_block> &__bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_mbar = static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2}], [%3];"
      :
      : "r"(smem_int_ptr), "l"(gmem_tensor_map), "r"(coord0), "r"(smem_int_mbar)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_2d_global_to_shared(
    void *smem_ptr, const CUtensorMap *__tensor_map, int32 const &coord0,
    int32 const &coord1, ::cuda::barrier<::cuda::thread_scope_block> &__bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_mbar = static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(smem_int_ptr), "l"(gmem_tensor_map), "r"(coord0), "r"(coord1),
        "r"(smem_int_mbar)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_3d_global_to_shared(
    void *smem_ptr, const CUtensorMap *__tensor_map, int32 const &coord0,
    int32 const &coord1, int32 const &coord2,
    ::cuda::barrier<::cuda::thread_scope_block> &__bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_mbar = static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(smem_int_ptr), "l"(gmem_tensor_map), "r"(coord0), "r"(coord1),
        "r"(coord2), "r"(smem_int_mbar)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_4d_global_to_shared(
    void *smem_ptr, const CUtensorMap *__tensor_map, int32 const &coord0,
    int32 const &coord1, int32 const &coord2, int32 const &coord3,
    ::cuda::barrier<::cuda::thread_scope_block> &__bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_mbar = static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];"
      :
      : "r"(smem_int_ptr), "l"(gmem_tensor_map), "r"(coord0), "r"(coord1),
        "r"(coord2), "r"(coord3), "r"(smem_int_mbar)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_5d_global_to_shared(
    void *smem_ptr, const CUtensorMap *__tensor_map, int32 const &coord0,
    int32 const &coord1, int32 const &coord2, int32 const &coord3,
    int32 const &coord4, ::cuda::barrier<::cuda::thread_scope_block> &__bar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_mbar = static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  uint32 smem_int_ptr = static_cast<uint32>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
      :
      : "r"(smem_int_ptr), "l"(gmem_tensor_map), "r"(coord0), "r"(coord1),
        "r"(coord2), "r"(coord3), "r"(coord4), "r"(smem_int_mbar)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_1d_shared_to_global(
    const CUtensorMap *__tensor_map, int32 const &coord0, const void *__src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_src = static_cast<uint32>(__cvta_generic_to_shared(__src));
  asm volatile(
      "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group "
      "[%0, {%1}], [%2];"
      :
      : "l"(gmem_tensor_map), "r"(coord0), "r"(smem_int_src)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_2d_shared_to_global(
    const CUtensorMap *__tensor_map, int32 const &coord0, int32 const &coord1,
    const void *__src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_src = static_cast<uint32>(__cvta_generic_to_shared(__src));
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
      "[%0, {%1, %2}], [%3];"
      :
      : "l"(gmem_tensor_map), "r"(coord0), "r"(coord1), "r"(smem_int_src)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_3d_shared_to_global(
    const CUtensorMap *__tensor_map, int32 const &coord0, int32 const &coord1,
    int32 const &coord2, const void *__src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_src = static_cast<uint32>(__cvta_generic_to_shared(__src));
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
      "[%0, {%1, %2, %3}], [%4];"
      :
      : "l"(gmem_tensor_map), "r"(coord0), "r"(coord1), "r"(coord2),
        "r"(smem_int_src)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_4d_shared_to_global(
    const CUtensorMap *__tensor_map, int32 const &coord0, int32 const &coord1,
    int32 const &coord2, int32 const &coord3, const void *__src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_src = static_cast<uint32>(__cvta_generic_to_shared(__src));
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group "
      "[%0, {%1, %2, %3, %4}], [%5];"
      :
      : "l"(gmem_tensor_map), "r"(coord0), "r"(coord1), "r"(coord2),
        "r"(coord3), "r"(smem_int_src)
      : "memory");
#endif
}

inline __device__ void cp_async_bulk_tensor_5d_shared_to_global(
    const CUtensorMap *__tensor_map, int32 const &coord0, int32 const &coord1,
    int32 const &coord2, int32 const &coord3, int32 const &coord4,
    const void *__src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 gmem_tensor_map = reinterpret_cast<uint64>(__tensor_map);
  uint32 smem_int_src = static_cast<uint32>(__cvta_generic_to_shared(__src));
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group "
      "[%0, {%1, %2, %3, %4, %5}], [%6];"
      :
      : "l"(gmem_tensor_map), "r"(coord0), "r"(coord1), "r"(coord2),
        "r"(coord3), "r"(coord4), "r"(smem_int_src)
      : "memory");
#endif
}

} // namespace experimental
} // namespace device
} // namespace cuda

#endif // !defined(__cccl_lib_experimental_ctk12_cp_async_exposure)

#if !defined(__cccl_lib_local_barrier_arrive_tx)

namespace cuda {
namespace device {

inline __device__ void barrier_expect_tx(
    ::cuda::barrier<::cuda::thread_scope_block> &__bar,
    uint32 const &__transaction_count_update) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64 *mbar_ptr = ::cuda::device::barrier_native_handle(__bar);
  uint32 smem_int_mbar = static_cast<uint32>(__cvta_generic_to_shared(mbar_ptr));
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_int_mbar), "r"(__transaction_count_update)
               : "memory");
#endif
}

inline __device__ ::cuda::barrier<::cuda::thread_scope_block>::arrival_token
barrier_arrive_tx(::cuda::barrier<::cuda::thread_scope_block> &__bar,
                  uint32 const &__arrive_count_update,
                  uint32 const &__transaction_count_update) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // For __arrive_count_update == 1, CCCL uses a single instruction
  // (mbarrier.arrive.expect_tx). Expect + arrive is functionally equivalent and
  // avoids depending on arrival_token representation.
  barrier_expect_tx(__bar, __transaction_count_update);
#endif
  return __bar.arrive(__arrive_count_update);
}

} // namespace device
} // namespace cuda

#endif // !defined(__cccl_lib_local_barrier_arrive_tx)

