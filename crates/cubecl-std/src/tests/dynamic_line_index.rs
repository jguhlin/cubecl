use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::CubeElement;

/// Repro test for CUDA dynamic Line indexing bug.
///
/// Tests that dynamic indexing on Line<T> works correctly.
/// Static indexing (using loop variable directly) works, but
/// dynamic indexing (with modulo/arithmetic) was broken.

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ReproConfig {
    pub line_size: u32,
    pub cube_dim: u32,
}

/// Test case A: Static index pattern (works)
/// This uses the loop variable directly as the index.
#[cube(launch_unchecked)]
pub fn repro_line_index_static<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] config: ReproConfig,
) {
    let cube_dim = config.cube_dim;
    let thread_index = UNIT_POS;
    if cube_dim == u32::new(0) || thread_index >= cube_dim {
        terminate!();
    }
    if input.len() == 0 || output.len() == 0 {
        terminate!();
    }
    let line = input[0];

    if thread_index == u32::new(0) {
        if config.line_size == u32::new(4) {
            let mut out_line = Line::empty(4usize).fill(F::new(0.0));
            let mut l = u32::new(0);
            while l < config.line_size {
                let idx = l as usize;
                out_line[idx] = line[idx] + F::new(1.0);
                l += u32::new(1);
            }
            output[0] = out_line;
        }
    }
}

/// Test case B: Dynamic index with modulo (was broken, now fixed)
/// This computes the index dynamically using modulo arithmetic.
#[cube(launch_unchecked)]
pub fn repro_line_index_dynamic<F: Float>(
    input: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] config: ReproConfig,
) {
    let cube_dim = config.cube_dim;
    let thread_index = UNIT_POS;
    if cube_dim == u32::new(0) || thread_index >= cube_dim {
        terminate!();
    }
    if input.len() == 0 || output.len() == 0 {
        terminate!();
    }
    let line = input[0];

    if cube_dim == u32::new(1) {
        if thread_index == u32::new(0) {
            if config.line_size == u32::new(4) {
                let mut out_line = Line::empty(4usize).fill(F::new(0.0));
                let mut l = u32::new(0);
                while l < config.line_size {
                    let idx = (l + u32::new(1)) % config.line_size;
                    let idx_usize = idx as usize;
                    let out_idx = l as usize;
                    out_line[out_idx] = line[idx_usize] + F::new(1.0);
                    l += u32::new(1);
                }
                output[0] = out_line;
            }
        }
        terminate!();
    }
}

pub fn test_line_index_static<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let line_size = 4u32;
    let input_vals: Vec<F> = vec![F::new(1.0), F::new(2.0), F::new(3.0), F::new(4.0)];
    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty((line_size * core::mem::size_of::<F>() as u32) as usize);

    let config = ReproConfig {
        line_size,
        cube_dim: 2,
    };

    unsafe {
        repro_line_index_static::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<F>(&input, 1, line_size as usize),
            ArrayArg::from_raw_parts::<F>(&output, 1, line_size as usize),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // Expected: [2.0, 3.0, 4.0, 5.0] - each element + 1.0
    let expected: Vec<F> = vec![F::new(2.0), F::new(3.0), F::new(4.0), F::new(5.0)];

    assert_eq!(&actual[..line_size as usize], &expected[..]);
}

pub fn test_line_index_dynamic<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let line_size = 4u32;
    let input_vals: Vec<F> = vec![F::new(1.0), F::new(2.0), F::new(3.0), F::new(4.0)];
    let input = client.create_from_slice(F::as_bytes(&input_vals));
    let output = client.empty((line_size * core::mem::size_of::<F>() as u32) as usize);

    let config = ReproConfig {
        line_size,
        cube_dim: 1,
    };

    unsafe {
        repro_line_index_dynamic::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(2),
            ArrayArg::from_raw_parts::<F>(&input, 1, line_size as usize),
            ArrayArg::from_raw_parts::<F>(&output, 1, line_size as usize),
            config,
        )
        .unwrap();
    }

    let actual = client.read_one(output);
    let actual = F::from_bytes(&actual);

    // Expected: [3.0, 4.0, 5.0, 2.0] - rotated indices + 1.0
    // line[(0+1)%4] + 1 = line[1] + 1 = 2.0 + 1.0 = 3.0
    // line[(1+1)%4] + 1 = line[2] + 1 = 3.0 + 1.0 = 4.0
    // line[(2+1)%4] + 1 = line[3] + 1 = 4.0 + 1.0 = 5.0
    // line[(3+1)%4] + 1 = line[0] + 1 = 1.0 + 1.0 = 2.0
    let expected: Vec<F> = vec![F::new(3.0), F::new(4.0), F::new(5.0), F::new(2.0)];

    assert_eq!(&actual[..line_size as usize], &expected[..]);
}

#[macro_export]
macro_rules! testgen_dynamic_line_index {
    () => {
        mod dynamic_line_index {
            use super::*;
            use $crate::tests::dynamic_line_index::*;

            #[$crate::tests::test_log::test]
            fn test_line_index_static_f32() {
                let client = TestRuntime::client(&Default::default());
                test_line_index_static::<TestRuntime, f32>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_line_index_dynamic_f32() {
                let client = TestRuntime::client(&Default::default());
                test_line_index_dynamic::<TestRuntime, f32>(client);
            }
        }
    };
}
