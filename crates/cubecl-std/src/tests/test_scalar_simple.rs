use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_runtime_scalar_add<F: Float>(
    output: &mut Array<F>,
    offset: u32,
) {
    let thread_index = UNIT_POS;
    if thread_index >= 8u32 {
        terminate!();
    }

    // Simple test: add runtime scalar to thread index
    output[thread_index as usize] = F::new(thread_index as f32 + offset as f32);
}
