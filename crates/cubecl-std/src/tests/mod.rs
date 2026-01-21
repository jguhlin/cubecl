/// Re-export for testgen macros.
pub use test_log;

pub mod arg_binding_optimizer;
pub mod array_inline_indexing;
pub mod dynamic_line_index;
pub mod event;
pub mod reinterpret_slice;
pub mod tensor;
pub mod trigonometry;
pub mod view;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_cubecl_std {
            use super::*;
            use half::{bf16, f16};

            cubecl_std::testgen_reinterpret_slice!();
            cubecl_std::testgen_trigonometry!();
            cubecl_std::testgen_event!();
            cubecl_std::testgen_dynamic_line_index!();
            cubecl_std::testgen_arg_binding_optimizer!();
            cubecl_std::testgen_array_inline_indexing!();
        }
    };
}

#[macro_export]
macro_rules! testgen_cuda {
    () => {
        mod test_cubecl_std_cuda {
            use super::*;
            use half::{bf16, f16};

            cubecl_std::testgen_array_inline_indexing_bf16!();
        }
    };
}
