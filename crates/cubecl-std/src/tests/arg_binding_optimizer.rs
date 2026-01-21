use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::CubeElement;

/// Repro test for CubeCL argument binding optimizer bug.
///
/// Tests that function arguments only used in comptime conditionals
/// that evaluate to false don't cause the optimizer to drop critical code.
///
/// Bug: When comptime!(condition) is false, the branch is eliminated during
/// frontend expansion, making arguments appear "unused" to dead code elimination.
/// This causes DCE to remove operations involving these arguments, which can
/// cascade into removing critical computation code, resulting in all-zero output.

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct AccumConfig {
    pub line_size: u32,
    pub cube_dim: u32,
}

/// Test case 1: Simple conditional argument usage
///
/// This kernel tests the simplest case where an argument is only used
/// in a comptime conditional branch. When use_unused=false, the argument
/// appears unused and should trigger the bug (all-zero output).
#[cube(launch_unchecked)]
pub fn repro_arg_binding_simple<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    unused_arg: &Array<F>,  // Only used when use_unused=true
    #[comptime] use_unused: bool,
) {
    let unit_id = UNIT_POS;

    if unit_id < output.len() as u32 {
        let val = if unit_id < input.len() as u32 {
            input[unit_id as usize]
        } else {
            F::new(0.0)
        };

        // Conditionally use unused_arg based on comptime condition
        let result = if comptime!(use_unused) && unit_id < unused_arg.len() as u32 {
            val + unused_arg[unit_id as usize]
        } else {
            val
        };

        output[unit_id as usize] = result;
    }
}

/// Test case 2: Simple conditional with workaround
///
/// Same as test case 1 but with the argument touching workaround
/// that prevents the optimizer bug.
#[cube(launch_unchecked)]
pub fn repro_arg_binding_simple_workaround<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    unused_arg: &Array<F>,
    #[comptime] use_unused: bool,
) {
    let unit_id = UNIT_POS;

    // WORKAROUND: Touch the unused argument to prevent optimizer from dropping it
    // This is a no-op that makes the argument appear "used" to DCE
    let _force_use = unused_arg.len() + input.len();

    if unit_id < output.len() as u32 {
        let val = if unit_id < input.len() as u32 {
            input[unit_id as usize]
        } else {
            F::new(0.0)
        };

        let result = if comptime!(use_unused) && unit_id < unused_arg.len() as u32 {
            val + unused_arg[unit_id as usize]
        } else {
            val
        };

        output[unit_id as usize] = result;
    }
}

/// Test case 3: Line accumulation pattern (real-world usage)
///
/// This kernel tests the pattern used in Fulgurite's decode_attention:
/// - Line-based input
/// - Accumulation across multiple lines
/// - Unused cache argument (only used in other configurations)
#[cube(launch_unchecked)]
pub fn repro_arg_binding_line<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
    _unused_cache: &Array<F>,  // Not used in this config
    #[comptime] config: AccumConfig,
) {
    let cube_dim = config.cube_dim;
    let thread_index = UNIT_POS;

    if cube_dim == 0 || thread_index >= cube_dim {
        terminate!();
    }

    let mut acc = F::new(0.0);
    let mut idx = thread_index;

    // Accumulate values across lines
    while idx < input.len() as u32 {
        let line = input[idx as usize];
        let mut l = 0u32;
        while l < config.line_size {
            acc = acc + line[l as usize];
            l += 1;
        }
        idx += cube_dim;
    }

    output[thread_index as usize] = acc;
}

/// Test case 4: Line accumulation with workaround
///
/// Same as test case 3 but with the argument touching workaround.
#[cube(launch_unchecked)]
pub fn repro_arg_binding_line_workaround<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<F>,
    unused_cache: &Array<F>,
    #[comptime] config: AccumConfig,
) {
    let cube_dim = config.cube_dim;
    let thread_index = UNIT_POS;

    // WORKAROUND: Touch the unused cache argument
    let _force_use = unused_cache.len();

    if cube_dim == 0 || thread_index >= cube_dim {
        terminate!();
    }

    let mut acc = F::new(0.0);
    let mut idx = thread_index;

    while idx < input.len() as u32 {
        let line = input[idx as usize];
        let mut l = 0u32;
        while l < config.line_size {
            acc = acc + line[l as usize];
            l += 1;
        }
        idx += cube_dim;
    }

    output[thread_index as usize] = acc;
}

// Test functions

pub fn test_arg_binding_simple<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let input_vals: Vec<F> = vec![F::new(1.0), F::new(2.0), F::new(3.0), F::new(4.0)];
    let unused_vals: Vec<F> = vec![F::new(10.0), F::new(20.0), F::new(30.0), F::new(40.0)];

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let unused = client.create_from_slice(F::as_bytes(&unused_vals));
    let output = client.empty(input_vals.len() * core::mem::size_of::<F>());

    // Launch with use_unused=false to trigger the bug
    unsafe {
        repro_arg_binding_simple::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            ArrayArg::from_raw_parts::<F>(&input, input_vals.len(), 1),
            ArrayArg::from_raw_parts::<F>(&output, input_vals.len(), 1),
            ArrayArg::from_raw_parts::<F>(&unused, unused_vals.len(), 1),
            false,  // use_unused = false (triggers bug)
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // CRITICAL ASSERTION: Output should NOT be all zeros
    // This is the primary bug detection - if DCE removes critical code,
    // output will be [0.0, 0.0, 0.0, 0.0] instead of [1.0, 2.0, 3.0, 4.0]
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "Output is all zeros - optimizer bug detected! Got {:?}",
        actual
    );

    // Verify correctness - should be a copy of input
    assert_eq!(&actual[..], &input_vals[..], "Output doesn't match input");
}

pub fn test_arg_binding_simple_workaround<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
) {
    let input_vals: Vec<F> = vec![F::new(1.0), F::new(2.0), F::new(3.0), F::new(4.0)];
    let unused_vals: Vec<F> = vec![F::new(10.0), F::new(20.0), F::new(30.0), F::new(40.0)];

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let unused = client.create_from_slice(F::as_bytes(&unused_vals));
    let output = client.empty(input_vals.len() * core::mem::size_of::<F>());

    // Launch with workaround - should always work
    unsafe {
        repro_arg_binding_simple_workaround::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            ArrayArg::from_raw_parts::<F>(&input, input_vals.len(), 1),
            ArrayArg::from_raw_parts::<F>(&output, input_vals.len(), 1),
            ArrayArg::from_raw_parts::<F>(&unused, unused_vals.len(), 1),
            false,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // Verify correctness
    assert_eq!(&actual[..], &input_vals[..], "Output doesn't match input");
}

pub fn test_arg_binding_line<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let line_size = 4u32;
    let cube_dim = 2u32;
    let num_lines = 2u32;  // 2 lines for 2 threads - each thread processes 1 line

    // Create input: 2 lines, each containing [1.0, 2.0, 3.0, 4.0]
    let input_vals: Vec<F> = vec![
        // Line 0
        F::new(1.0),
        F::new(2.0),
        F::new(3.0),
        F::new(4.0),
        // Line 1
        F::new(1.0),
        F::new(2.0),
        F::new(3.0),
        F::new(4.0),
    ];

    let unused_vals: Vec<F> = vec![F::new(100.0), F::new(200.0)];

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let unused = client.create_from_slice(F::as_bytes(&unused_vals));
    let output = client.empty(cube_dim as usize * core::mem::size_of::<F>());

    let config = AccumConfig { line_size, cube_dim };

    unsafe {
        repro_arg_binding_line::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts::<Line<F>>(&input, num_lines as usize, line_size as usize),
            ArrayArg::from_raw_parts::<F>(&output, cube_dim as usize, 1),
            ArrayArg::from_raw_parts::<F>(&unused, unused_vals.len(), 1),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // Thread 0 accumulates line 0: 1+2+3+4 = 10.0
    // Thread 1 accumulates line 1: 1+2+3+4 = 10.0
    let expected_sum = F::new(10.0);

    // CRITICAL ASSERTION: Output should NOT be all zeros
    let sum: F = actual.iter().cloned().fold(F::new(0.0), |a, b| a + b);
    assert!(
        sum > F::new(0.001),
        "Output is all zeros - optimizer bug detected! Got {:?}",
        actual
    );

    // Verify each thread's output
    for i in 0..cube_dim as usize {
        assert_eq!(
            actual[i], expected_sum,
            "Thread {} output incorrect: expected {:?}, got {:?}",
            i, expected_sum, actual[i]
        );
    }
}

pub fn test_arg_binding_line_workaround<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
) {
    let line_size = 4u32;
    let cube_dim = 2u32;
    let num_lines = 2u32;  // 2 lines for 2 threads - each thread processes 1 line

    let input_vals: Vec<F> = vec![
        F::new(1.0),
        F::new(2.0),
        F::new(3.0),
        F::new(4.0),
        F::new(1.0),
        F::new(2.0),
        F::new(3.0),
        F::new(4.0),
    ];

    let unused_vals: Vec<F> = vec![F::new(100.0), F::new(200.0)];

    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let unused = client.create_from_slice(F::as_bytes(&unused_vals));
    let output = client.empty(cube_dim as usize * core::mem::size_of::<F>());

    let config = AccumConfig { line_size, cube_dim };

    unsafe {
        repro_arg_binding_line_workaround::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts::<Line<F>>(&input, num_lines as usize, line_size as usize),
            ArrayArg::from_raw_parts::<F>(&output, cube_dim as usize, 1),
            ArrayArg::from_raw_parts::<F>(&unused, unused_vals.len(), 1),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    let expected_sum = F::new(10.0);

    // Verify each thread's output
    for i in 0..cube_dim as usize {
        assert_eq!(
            actual[i], expected_sum,
            "Thread {} output incorrect: expected {:?}, got {:?}",
            i, expected_sum, actual[i]
        );
    }
}

#[macro_export]
macro_rules! testgen_arg_binding_optimizer {
    () => {
        mod arg_binding_optimizer {
            use super::*;
            use $crate::tests::arg_binding_optimizer::*;

            #[$crate::tests::test_log::test]
            fn test_arg_binding_simple_f32() {
                let client = TestRuntime::client(&Default::default());
                test_arg_binding_simple::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_arg_binding_simple_workaround_f32() {
                let client = TestRuntime::client(&Default::default());
                test_arg_binding_simple_workaround::<TestRuntime, f32>(client);
            }
        }
    };
}
