use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::CubeElement;

/// Test for CUDA Array<Line<T>> inline indexing with computed expressions bug.
///
/// Issue: CUDA kernels using inline Array indexing of `Array<Line<T>>` with computed
/// expressions return zeros, while the same pattern through a helper function works.

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ArrayIndexConfig {
    pub line_size: u32,
    pub array_size: u32,
    pub offset: u32, // For computed index test
}

/// Test basic scalar arithmetic - NO Array<Line<T>>
/// This tests if GlobalScalar loading itself is broken
#[cube(launch_unchecked)]
pub fn kernel_scalar_arithmetic<F: Float>(
    output: &mut [F],
    offset: u32, // Runtime scalar
) {
    let thread_index = UNIT_POS;
    if thread_index >= 8u32 {
        terminate!();
    }

    // Simple scalar arithmetic: output[i] = (thread_index + offset) as float
    let sum = thread_index + offset;
    output[thread_index as usize] = F::cast_from(sum as f32);
}

/// Test simple indexing with variable index (should work)
#[cube(launch_unchecked)]
pub fn kernel_simple_index<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [F],
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    if thread_index >= config.array_size {
        terminate!();
    }

    // Simple index - should work
    let idx = thread_index as usize;
    let val = input[idx].extract(0);
    output[idx] = val;
}

/// Test computed inline indexing (likely fails on CUDA)
#[cube(launch_unchecked)]
pub fn kernel_computed_inline_index<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [F],
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    if thread_index >= config.array_size {
        terminate!();
    }

    // Computed index - may fail on CUDA
    let idx = (thread_index + config.offset) as usize;
    let val = input[idx].extract(0);
    output[thread_index as usize] = val;
}

/// Test complex computed expression (definitely fails on CUDA)
/// Matches the pattern from the bug report:
/// query[((batch * seq + seq) * heads + head) * head_dim + thread][0]
#[cube(launch_unchecked)]
pub fn kernel_complex_computed_index<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [F],
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    let head_idx = CUBE_POS_X;
    let batch_idx = CUBE_POS_Y;

    if thread_index >= config.array_size {
        terminate!();
    }

    // Complex computed index - this is the failing pattern
    // Matches: query[((batch * seq + seq) * heads + head) * head_dim + thread]
    let idx = ((batch_idx * u32::new(2) + head_idx) * config.array_size + thread_index) as usize;
    let val = input[idx].extract(0);
    // Write using the computed (global) index so every element is produced exactly once.
    output[idx] = val;
}

/// Helper function that should work (workaround)
#[cube]
fn load_line_element<F: Float, N: Size>(
    input: &[Vector<F, N>],
    index: usize,
    _line_size: usize,
) -> F {
    let line = input[index];
    line.extract(0)
}

/// Test with helper function workaround (should work)
#[cube(launch_unchecked)]
pub fn kernel_with_helper<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [F],
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    let head_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let batch_idx = CUBE_POS_Z;

    if thread_index >= head_dim {
        terminate!();
    }

    // Use helper function - should work even with computed index
    let q_index =
        (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim) + thread_index;
    let total = seq_len * num_heads * head_dim;
    if q_index >= total {
        terminate!();
    }

    let idx = q_index as usize;
    let val = load_line_element(&input, idx, config.line_size as usize);
    output[idx] = val;
}

/// Test exact pattern from bug report - WITHOUT helper (should fail on buggy CUDA)
/// This matches: query[((batch * seq_len + seq_idx) * num_heads + head_idx) * head_dim + thread][0]
#[cube(launch_unchecked)]
pub fn kernel_exact_bug_pattern<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [F],
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    #[comptime] _config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    let head_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let batch_idx = CUBE_POS_Z;

    if thread_index >= head_dim {
        terminate!();
    }

    // EXACT BUG PATTERN from seq kernel
    // This is the failing pattern: query[((batch * seq_len + seq_idx) * num_heads + head_idx) * head_dim + thread][0]
    let q_index =
        (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim) + thread_index;
    let total = seq_len * num_heads * head_dim;
    if q_index >= total {
        terminate!();
    }

    let val = input[q_index as usize].extract(0);
    output[q_index as usize] = val;
}

/// Line-size > 1 writeback path using the same computed index pattern.
///
/// This mirrors the pattern seen in attention kernels where the head dimension is stored as
/// `Array<Line<F>>` with `line_size > 1`, and the address computation is done via arithmetic on
/// runtime scalars.
#[cube(launch_unchecked)]
pub fn kernel_line_write_index<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    seq_len: u32,
    num_heads: u32,
    head_dim_lines: u32,
    #[comptime] _config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    let head_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let batch_idx = CUBE_POS_Z;

    if thread_index >= head_dim_lines {
        terminate!();
    }

    let q_index =
        (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim_lines) + thread_index;
    if q_index as usize >= input.len() || q_index as usize >= output.len() {
        terminate!();
    }

    let val = input[q_index as usize];
    output[q_index as usize] = val;
}

// ============================================================================
// Test functions
// ============================================================================

pub fn test_simple_index<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let array_size = 8u32;
    let input_vals: Vec<F> = (0..array_size).map(|i| F::new(i as f32 + 1.0)).collect();

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty(array_size as usize * core::mem::size_of::<F>());

    let config = ArrayIndexConfig {
        line_size: 1,
        array_size,
        offset: 0,
    };

    unsafe {
        kernel_simple_index::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(array_size),
            1usize,
            BufferArg::from_raw_parts(input.clone(), array_size as usize),
            BufferArg::from_raw_parts(output.clone(), array_size as usize),
            config,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    // Should be identical to input
    assert_eq!(&actual[..array_size as usize], &input_vals[..]);
}

pub fn test_computed_inline_index<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let array_size = 8u32;
    let offset = 2u32;
    let input_size = array_size + offset; // Create larger input to avoid out-of-bounds
    let input_vals: Vec<F> = (0..input_size).map(|i| F::new(i as f32 + 1.0)).collect();

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty(array_size as usize * core::mem::size_of::<F>());

    let config = ArrayIndexConfig {
        line_size: 1,
        array_size,
        offset,
    };

    unsafe {
        kernel_computed_inline_index::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(array_size),
            1usize,
            BufferArg::from_raw_parts(input.clone(), input_size as usize),
            BufferArg::from_raw_parts(output.clone(), array_size as usize),
            config,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    // Expected: input[2], input[3], ..., input[9] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    let expected: Vec<F> = (0..array_size)
        .map(|i| input_vals[(i + offset) as usize])
        .collect();

    // CRITICAL ASSERTION: Output should NOT be all zeros
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "Output is all zeros - bug detected! Got {:?}",
        actual
    );

    assert_eq!(&actual[..array_size as usize], &expected[..]);
}

pub fn test_complex_computed_index<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let array_size = 4u32; // 2x2 grid

    // Create input: 16 elements (2 * 2 * 4 = 16)
    let input_vals: Vec<F> = (0..16).map(|i| F::new(i as f32 + 1.0)).collect();

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty(16 * core::mem::size_of::<F>());

    let config = ArrayIndexConfig {
        line_size: 1,
        array_size,
        offset: 0,
    };

    unsafe {
        kernel_complex_computed_index::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(2, 2, 1),  // X=2 (heads), Y=2 (batch)
            CubeDim::new_1d(array_size), // threads per cube = head_dim
            1usize,
            BufferArg::from_raw_parts(input.clone(), 16),
            BufferArg::from_raw_parts(output.clone(), 16),
            config,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    // Each thread writes output[idx] = input[idx][0], so the full output should match input.
    assert_eq!(&actual[..], &input_vals[..]);
}

pub fn test_with_helper<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let seq_len = 2u32; // Reduced to avoid too many threads
    let num_heads = 2u32;
    let head_dim = 4u32; // Reduced
    let total_threads = seq_len * num_heads * head_dim; // 2 * 2 * 4 = 16 threads

    // Input size: batch=1 * seq_len * num_heads * head_dim
    let input_size = 1 * seq_len * num_heads * head_dim;
    let input_vals: Vec<F> = (0..input_size).map(|i| F::new(i as f32 + 1.0)).collect();

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty(total_threads as usize * core::mem::size_of::<F>());

    let config = ArrayIndexConfig {
        line_size: 1,
        array_size: head_dim,
        offset: 0,
    };

    unsafe {
        kernel_with_helper::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(num_heads, seq_len, 1), // X=num_heads, Y=seq_len, Z=batch(=1)
            CubeDim::new_1d(head_dim),                // threads per cube = head_dim
            1usize,
            BufferArg::from_raw_parts(input.clone(), input_size as usize),
            BufferArg::from_raw_parts(output.clone(), total_threads as usize),
            seq_len,
            num_heads,
            head_dim,
            config,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    // Output should be an exact copy of input (output[q_index] = input[q_index][0]).
    assert_eq!(&actual[..], &input_vals[..]);
}

pub fn test_scalar_arithmetic<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let array_size = 8u32;
    let offset = 10u32;

    let output = client.empty(array_size as usize * core::mem::size_of::<F>());

    unsafe {
        kernel_scalar_arithmetic::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(array_size),
            BufferArg::from_raw_parts(output.clone(), array_size as usize),
            offset,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    // Check that values are non-zero - if scalars work, output should be [10.0, 11.0, ..., 17.0]
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "Scalar arithmetic failed - output is all zeros or very small. Got {:?}",
        actual
    );
}

pub fn test_exact_bug_pattern<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let array_size = 8u32;
    let seq_len = 4u32;
    let num_heads = 2u32;
    let head_dim = array_size; // Each head processes array_size elements
    let total_threads = seq_len * num_heads * head_dim; // 4 * 2 * 8 = 64 threads

    // Input size: batch=1 * seq_len=4 * num_heads=2 * head_dim=8 = 64 elements
    let input_size = 1 * seq_len * num_heads * head_dim;
    let input_vals: Vec<F> = (0..input_size).map(|i| F::new(i as f32 + 1.0)).collect();

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty(total_threads as usize * core::mem::size_of::<F>());

    let config = ArrayIndexConfig {
        line_size: 1,
        array_size,
        offset: 0,
    };

    unsafe {
        kernel_exact_bug_pattern::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(num_heads, seq_len, 1), // X=num_heads, Y=seq_len, Z=batch(=1)
            CubeDim::new_1d(head_dim),                // threads per cube = head_dim
            1usize,
            BufferArg::from_raw_parts(input.clone(), input_size as usize),
            BufferArg::from_raw_parts(output.clone(), total_threads as usize),
            seq_len,
            num_heads,
            head_dim,
            config,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    // Output should be an exact copy of input (output[q_index] = input[q_index][0]).
    assert_eq!(&actual[..], &input_vals[..]);
}

pub fn test_line_write_index<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let line_size = 4u32;
    let seq_len = 2u32;
    let num_heads = 2u32;
    let head_dim_lines = 8u32;

    let total_lines = seq_len * num_heads * head_dim_lines;
    let scalar_count = total_lines * line_size;

    let input_vals: Vec<F> = (0..scalar_count).map(|i| F::new(i as f32 + 1.0)).collect();

    let input = client.create_from_slice(F::as_bytes(&input_vals));

    // Initialize output to zeros so failures are deterministic (no uninitialized memory reads).
    let zeros: Vec<F> = vec![F::new(0.0); scalar_count as usize];
    let output = client.create_from_slice(F::as_bytes(&zeros));

    let config = ArrayIndexConfig {
        line_size,
        array_size: head_dim_lines,
        offset: 0,
    };

    unsafe {
        kernel_line_write_index::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(num_heads, seq_len, 1),
            CubeDim::new_1d(head_dim_lines),
            line_size as usize,
            BufferArg::from_raw_parts(input.clone(), scalar_count as usize),
            BufferArg::from_raw_parts(output.clone(), scalar_count as usize),
            seq_len,
            num_heads,
            head_dim_lines,
            config,
        )
    }

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual.len(), input_vals.len());
    assert_eq!(&actual[..], &input_vals[..]);
}

#[macro_export]
macro_rules! testgen_array_inline_indexing {
    () => {
        mod array_inline_indexing {
            use super::*;
            use $crate::tests::array_inline_indexing::*;

            #[$crate::tests::test_log::test]
            fn test_simple_index_f32() {
                let client = TestRuntime::client(&Default::default());
                test_simple_index::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_computed_inline_index_f32() {
                let client = TestRuntime::client(&Default::default());
                test_computed_inline_index::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_complex_computed_index_f32() {
                let client = TestRuntime::client(&Default::default());
                test_complex_computed_index::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_exact_bug_pattern_f32() {
                let client = TestRuntime::client(&Default::default());
                test_exact_bug_pattern::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_with_helper_f32() {
                let client = TestRuntime::client(&Default::default());
                test_with_helper::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_scalar_arithmetic_f32() {
                let client = TestRuntime::client(&Default::default());
                test_scalar_arithmetic::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_line_write_index_f32() {
                let client = TestRuntime::client(&Default::default());
                test_line_write_index::<TestRuntime, f32>(client);
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_array_inline_indexing_bf16 {
    () => {
        mod array_inline_indexing_bf16 {
            use super::*;
            use $crate::tests::array_inline_indexing::*;

            #[$crate::tests::test_log::test]
            fn test_simple_index_bf16() {
                let client = TestRuntime::client(&Default::default());
                test_simple_index::<TestRuntime, bf16>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_computed_inline_index_bf16() {
                let client = TestRuntime::client(&Default::default());
                test_computed_inline_index::<TestRuntime, bf16>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_complex_computed_index_bf16() {
                let client = TestRuntime::client(&Default::default());
                test_complex_computed_index::<TestRuntime, bf16>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_exact_bug_pattern_bf16() {
                let client = TestRuntime::client(&Default::default());
                test_exact_bug_pattern::<TestRuntime, bf16>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_with_helper_bf16() {
                let client = TestRuntime::client(&Default::default());
                test_with_helper::<TestRuntime, bf16>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_line_write_index_bf16() {
                let client = TestRuntime::client(&Default::default());
                test_line_write_index::<TestRuntime, bf16>(client);
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_array_inline_indexing_f16 {
    () => {
        mod array_inline_indexing_f16 {
            use super::*;
            use $crate::tests::array_inline_indexing::*;

            #[$crate::tests::test_log::test]
            fn test_line_write_index_f16() {
                let client = TestRuntime::client(&Default::default());
                test_line_write_index::<TestRuntime, f16>(client);
            }
        }
    };
}
