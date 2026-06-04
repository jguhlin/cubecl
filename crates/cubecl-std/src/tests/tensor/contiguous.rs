use cubecl_core::{
    CubeElement,
    prelude::{Numeric, Runtime, TensorBinding},
    zspace::{Shape, Strides},
};

use crate::tensor::{TensorHandle, copy_gpu_ref};

pub fn test_contiguous_rank_mismatch<R: Runtime, C: Numeric + CubeElement>(device: &R::Device) {
    let client = R::client(device);

    let shape = vec![2, 2, 2];
    let strides = vec![4, 2, 1, 1];
    let expected: Vec<C> = (0..8).map(|i| C::from_int(i as i64)).collect();

    let handle = client.create_from_slice(C::as_bytes(&expected));
    let dtype = C::cube_type();
    let output = TensorHandle::<R>::empty(&client, shape.clone(), dtype);

    let input_binding = unsafe {
        TensorBinding::from_raw_parts(handle, Strides::from(strides), Shape::from(shape))
    };

    let output_binding = output.clone().binding();

    copy_gpu_ref(&client, input_binding, output_binding, dtype);

    let actual = client.read_one_unchecked_tensor(output.into_copy_descriptor());
    let actual = C::from_bytes(&actual);

    assert_eq!(&expected[..], actual, "contiguous copy mismatch");
}
