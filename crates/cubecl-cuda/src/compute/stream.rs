use crate::compute::{
    storage::{
        cpu::{PINNED_MEMORY_ALIGNMENT, PinnedMemoryStorage},
        gpu::GpuStorage,
    },
    sync::Fence,
};
use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{Binding, IoError, ServerError},
};
use cubecl_runtime::{
    config::streaming::StreamPriority,
    logging::ServerLogger,
    memory_management::{
        MemoryAllocationMode, MemoryManagement, MemoryManagementOptions, drop_queue,
    },
    stream::EventStreamBackend,
};
use std::{mem::MaybeUninit, sync::Arc};

const METADATA_STAGING_CAPACITY: usize = 16 * 1024;

/// Pre-allocated staging buffers for kernel metadata uploads.
///
/// Avoids GPU memory allocation during kernel launch, which is required
/// for CUDA graph capture compatibility. The GPU buffer is allocated once
/// at stream creation via `malloc_sync` (not stream-ordered), and reused
/// for every kernel launch on this stream.
pub struct MetadataStaging {
    gpu_ptr: cudarc::driver::sys::CUdeviceptr,
    host_buf: Vec<u8>,
}

impl MetadataStaging {
    fn new(capacity: usize) -> Self {
        // SAFETY: Synchronous allocation, safe outside of stream capture.
        let gpu_ptr = unsafe { cudarc::driver::result::malloc_sync(capacity) }
            .expect("Failed to allocate metadata staging GPU buffer");
        Self {
            gpu_ptr,
            host_buf: vec![0u8; capacity],
        }
    }

    /// Copy `data` into the host staging buffer, then async-copy to the GPU buffer.
    pub fn upload(
        &mut self,
        data: &[u8],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), IoError> {
        if data.len() > self.host_buf.len() {
            self.grow(data.len());
        }
        self.host_buf[..data.len()].copy_from_slice(data);
        // SAFETY: gpu_ptr was allocated with at least `data.len()` bytes (after
        // potential grow), host_buf slice is valid, and stream is an active CUDA stream.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                self.gpu_ptr,
                &self.host_buf[..data.len()],
                stream,
            )
        }
        .map_err(|e| IoError::Unknown {
            description: format!("metadata staging memcpy failed: {e}"),
            backtrace: BackTrace::capture(),
        })
    }

    /// Returns a raw pointer to the stored `gpu_ptr` field, for use as a kernel
    /// argument binding. Valid as long as `self` is not moved or dropped.
    pub fn gpu_ptr_binding(&self) -> *mut std::ffi::c_void {
        &self.gpu_ptr as *const _ as *mut std::ffi::c_void
    }

    pub fn gpu_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.gpu_ptr
    }

    fn grow(&mut self, min_capacity: usize) {
        let new_capacity = min_capacity.next_power_of_two();
        // SAFETY: Freeing the old buffer (sync, not stream-ordered) and allocating a new one.
        unsafe {
            if let Err(e) = cudarc::driver::result::free_sync(self.gpu_ptr) {
                eprintln!("metadata staging free error during grow: {e}");
            }
            self.gpu_ptr = cudarc::driver::result::malloc_sync(new_capacity)
                .expect("Failed to reallocate metadata staging GPU buffer");
        }
        self.host_buf.resize(new_capacity, 0);
    }
}

impl Drop for MetadataStaging {
    fn drop(&mut self) {
        // SAFETY: gpu_ptr was allocated via malloc_sync and has not been freed.
        unsafe {
            if let Err(e) = cudarc::driver::result::free_sync(self.gpu_ptr) {
                eprintln!("metadata staging free error: {e}");
            }
        }
    }
}

impl core::fmt::Debug for MetadataStaging {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetadataStaging")
            .field("gpu_ptr", &self.gpu_ptr)
            .field("capacity", &self.host_buf.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct Stream {
    pub sys: cudarc::driver::sys::CUstream,
    pub memory_management_gpu: MemoryManagement<GpuStorage>,
    pub memory_management_cpu: MemoryManagement<PinnedMemoryStorage>,
    pub errors: Vec<ServerError>,
    pub drop_queue: drop_queue::PendingDropQueue<Fence>,
    pub metadata_staging: MetadataStaging,
}

impl drop_queue::Fence for Fence {
    fn sync(self) {
        let _ = self.wait_sync().ok();
    }
}

#[derive(new, Debug)]
pub struct CudaStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
    logger: Arc<ServerLogger>,
    priority: StreamPriority,
}

/// Create a non-blocking CUDA stream, applying the requested priority hint.
///
/// `StreamPriority::Default` preserves the historical `cuStreamCreate` path so
/// existing users see no change. `Low`/`High` go through
/// `cuStreamCreateWithPriority` using the device's range as queried via
/// `cuCtxGetStreamPriorityRange`. CUDA convention: lower number = higher
/// priority, so the queried `greatest` is numerically smallest (most
/// aggressive) and `least` is numerically largest (least aggressive). On
/// devices without priority support both values are 0 and CUDA silently
/// ignores the priority argument — equivalent to the default path.
///
/// Both calls require a current CUDA context; callers in this crate always
/// set the context before invoking stream creation.
pub(crate) fn create_cuda_stream(priority: StreamPriority) -> cudarc::driver::sys::CUstream {
    use cudarc::driver::sys::{self, CUstream_flags};

    let use_greatest = match priority {
        StreamPriority::Default => {
            return cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .expect("Can create a new stream.");
        }
        StreamPriority::High => true,
        StreamPriority::Low => false,
    };

    // SAFETY: `cuCtxGetStreamPriorityRange` writes through both pointers on
    // success; we only read the locals after the `.expect()` confirms success.
    let value = unsafe {
        let mut least: i32 = 0;
        let mut greatest: i32 = 0;
        sys::cuCtxGetStreamPriorityRange(&mut least, &mut greatest)
            .result()
            .expect("Can query CUDA stream priority range.");
        if use_greatest { greatest } else { least }
    };

    // SAFETY: `cuStreamCreateWithPriority` writes the new stream handle through
    // the out pointer on success; `.expect()` ensures we only `assume_init` on
    // success.
    unsafe {
        let mut stream = MaybeUninit::uninit();
        sys::cuStreamCreateWithPriority(
            stream.as_mut_ptr(),
            CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            value,
        )
        .result()
        .expect("Can create a new CUDA stream with priority.");
        stream.assume_init()
    }
}

impl EventStreamBackend for CudaStreamBackend {
    type Stream = Stream;
    type Event = Fence;

    fn create_stream(&self) -> Self::Stream {
        let stream = create_cuda_stream(self.priority);

        let storage = GpuStorage::new(self.mem_alignment, stream);
        let memory_management_gpu = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );
        // We use the same page size and memory pools configuration for CPU pinned memory, since we
        // expect the CPU to have at least the same amount of RAM as GPU memory.
        let memory_management_cpu = MemoryManagement::from_configuration(
            PinnedMemoryStorage::new(),
            &MemoryDeviceProperties {
                max_page_size: self.mem_props.max_page_size,
                alignment: PINNED_MEMORY_ALIGNMENT as u64,
            },
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Pinned CPU Memory").mode(MemoryAllocationMode::Auto),
        );

        Stream {
            sys: stream,
            memory_management_gpu,
            memory_management_cpu,
            errors: Vec::new(),
            drop_queue: Default::default(),
            metadata_staging: MetadataStaging::new(METADATA_STAGING_CAPACITY),
        }
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        Fence::new(stream.sys)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        event.wait_async(stream.sys);
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }

    fn handle_cursor(stream: &Self::Stream, binding: &Binding) -> u64 {
        stream
            .memory_management_gpu
            .get_cursor(binding.memory.clone())
            .unwrap_or(0)
    }

    fn is_healthy(stream: &Self::Stream) -> bool {
        stream.errors.is_empty()
    }
}
