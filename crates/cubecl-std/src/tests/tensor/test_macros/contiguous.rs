#![allow(missing_docs)]

#[macro_export]
macro_rules! testgen_tensor_contiguous_rank_mismatch {
    () => {
        mod contiguous_rank_mismatch {
            $crate::testgen_tensor_contiguous_rank_mismatch!(f32);
        }
    };
    ($numeric:ident) => {
        use super::*;
        use $crate::tests::tensor::contiguous::test_contiguous_rank_mismatch as run_contiguous_rank_mismatch;

        pub type NumericT = $numeric;

        #[$crate::tests::test_log::test]
        pub fn test_contiguous_rank_mismatch() {
            run_contiguous_rank_mismatch::<TestRuntime, NumericT>(&Default::default());
        }
    };
    ([$($numeric:ident),*]) => {
        mod contiguous_rank_mismatch {
            use super::*;
            ::paste::paste! {
                $(mod [<$numeric _ty>] {
                    use super::*;

                    $crate::testgen_tensor_contiguous_rank_mismatch!($numeric);
                })*
            }
        }
    };
}
