use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::{CubeElement, prelude::ScalarArg};

/// Test for CUDA Array<Line<T>> inline indexing with computed expressions bug.
///
/// Issue: CUDA kernels using inline Array indexing of `Array<Line<T>>` with computed
/// expressions return zeros, while the same pattern through a helper function works.

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ArrayIndexConfig {
    pub line_size: u32,
    pub array_size: u32,
    pub offset: u32,  // For computed index test
}

/// Test basic scalar arithmetic - NO Array<Line<T>>
/// This tests if GlobalScalar loading itself is broken
#[cube(launch_unchecked)]
pub fn kernel_scalar_arithmetic<F: Float>(
    output: &mut Array<F>,
    offset: u32,  // Runtime scalar
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
pub fn kernel_simple_index<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    if thread_index >= config.array_size {
        terminate!();
    }

    // Simple index - should work
    let idx = thread_index as usize;
    let val = input[idx][0];
    output[idx] = val;
}

/// Test computed inline indexing (likely fails on CUDA)
#[cube(launch_unchecked)]
pub fn kernel_computed_inline_index<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    if thread_index >= config.array_size {
        terminate!();
    }

    // Computed index - may fail on CUDA
    let idx = (thread_index + config.offset) as usize;
    let val = input[idx][0];
    output[thread_index as usize] = val;
}

/// Test complex computed expression (definitely fails on CUDA)
/// Matches the pattern from the bug report:
/// query[((batch * seq + seq) * heads + head) * head_dim + thread][0]
#[cube(launch_unchecked)]
pub fn kernel_complex_computed_index<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
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
    let val = input[idx][0];
    output[thread_index as usize] = val;
}

/// Helper function that should work (workaround)
#[cube]
fn load_line_element<F: Float>(input: &Array<Line<F>>, index: usize, _line_size: usize) -> F {
    let line = input[index];
    line[0]
}

/// Test with helper function workaround (should work)
#[cube(launch_unchecked)]
pub fn kernel_with_helper<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    let head_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let batch_idx = CUBE_POS_Z;

    if thread_index >= config.array_size {
        terminate!();
    }

    // Use helper function - should work even with computed index
    let idx = (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim + thread_index) as usize;
    let val = load_line_element(&input, idx, config.line_size as usize);
    output[thread_index as usize] = val;
}

/// Test exact pattern from bug report - WITHOUT helper (should fail on buggy CUDA)
/// This matches: query[((batch * seq_len + seq_idx) * num_heads + head_idx) * head_dim + thread][0]
#[cube(launch_unchecked)]
pub fn kernel_exact_bug_pattern<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    #[comptime] config: ArrayIndexConfig,
) {
    let thread_index = UNIT_POS;
    let head_idx = CUBE_POS_X;
    let seq_idx = CUBE_POS_Y;
    let batch_idx = CUBE_POS_Z;

    if thread_index >= config.array_size {
        terminate!();
    }

    // EXACT BUG PATTERN from seq kernel
    // This is the failing pattern: query[((batch * seq_len + seq_idx) * num_heads + head_idx) * head_dim + thread][0]
    let q_index = (((batch_idx * seq_len + seq_idx) * num_heads + head_idx) * head_dim) + thread_index;
    let val = input[q_index as usize][0];
    output[thread_index as usize] = val;
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
            ArrayArg::from_raw_parts::<Line<F>>(&input, array_size as usize, 1),
            ArrayArg::from_raw_parts::<F>(&output, array_size as usize, 1),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // Should be identical to input
    assert_eq!(&actual[..array_size as usize], &input_vals[..]);
}

pub fn test_computed_inline_index<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let array_size = 8u32;
    let offset = 2u32;
    let input_size = array_size + offset;  // Create larger input to avoid out-of-bounds
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
            ArrayArg::from_raw_parts::<Line<F>>(&input, input_size as usize, 1),
            ArrayArg::from_raw_parts::<F>(&output, array_size as usize, 1),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
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
    let array_size = 4u32;  // 2x2 grid

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
            CubeCount::Static(1, 2, 2),  // Y=2 (batch), X=2 (heads)
            CubeDim::new_2d(2, array_size),  // X=2 (heads), Y=array_size (threads)
            ArrayArg::from_raw_parts::<Line<F>>(&input, 16, 1),
            ArrayArg::from_raw_parts::<F>(&output, 16, 1),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // CRITICAL ASSERTION: Output should NOT be all zeros
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "Output is all zeros - bug detected! Got {:?}",
        actual
    );

    // Verify at least first element is correct
    // idx = ((0 * 2 + 0) * 4 + 0) = 0, so should read input[0]
    assert!(actual[0] > F::new(0.0), "First element is zero");
}

pub fn test_with_helper<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let seq_len = 2u32;  // Reduced to avoid too many threads
    let num_heads = 2u32;
    let head_dim = 4u32;  // Reduced
    let total_threads = seq_len * num_heads * head_dim;  // 2 * 2 * 4 = 16 threads

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
            CubeCount::Static(1, num_heads, seq_len),  // Z=1 (batch), Y=seq_len, X=num_heads
            CubeDim::new_3d(num_heads, seq_len, head_dim),  // X=num_heads, Y=seq_len, Z=head_dim
            ArrayArg::from_raw_parts::<Line<F>>(&input, input_size as usize, 1),
            ArrayArg::from_raw_parts::<F>(&output, total_threads as usize, 1),
            ScalarArg::new(seq_len),
            ScalarArg::new(num_heads),
            ScalarArg::new(head_dim),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // CRITICAL ASSERTION: Output should NOT be all zeros
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "Output is all zeros - bug detected! Got {:?}",
        actual
    );
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
            ArrayArg::from_raw_parts::<F>(&output, array_size as usize, 1),
            ScalarArg::new(offset),
        )
        .unwrap();
    }

    let actual = client.read_one(output);
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
    let head_dim = array_size;  // Each head processes array_size elements
    let total_threads = seq_len * num_heads * head_dim;  // 4 * 2 * 8 = 64 threads

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
            CubeCount::Static(1, num_heads, seq_len),  // Z=1 (batch), Y=seq_len, X=num_heads
            CubeDim::new_3d(num_heads, seq_len, array_size),  // X=num_heads, Y=seq_len, Z=array_size
            ArrayArg::from_raw_parts::<Line<F>>(&input, input_size as usize, 1),
            ArrayArg::from_raw_parts::<F>(&output, total_threads as usize, 1),
            ScalarArg::new(seq_len),
            ScalarArg::new(num_heads),
            ScalarArg::new(head_dim),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // CRITICAL ASSERTION: Output should NOT be all zeros
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "BUG DETECTED! Output is all zeros. Got {:?}",
        actual
    );
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
        }
    };
}
