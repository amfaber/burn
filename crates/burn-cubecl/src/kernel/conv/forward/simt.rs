//! The SIMT convolution of cubek-convolution (implicit gemm on plain GPU arithmetic, for where no
//! tensor-core tile fits the element types: f32 on wgpu, whose Vulkan cooperative matrices take
//! only f16 and narrower inputs), on burn's tensors: forward, input gradient and weight gradient.
//! cubek declines what the kernels do not cover; these allocate the outputs in the layouts its
//! kernels write.

use burn_backend::{cubecl::dtype_to_storage_type, ops::ConvOptions};
use burn_std::Shape;
use cubek::convolution::{
    ConvolutionArgs, SimtBackwardDataTensors, SimtBackwardWeightTensors, SimtTensors,
    components::ConvSetupError, launch_simt, launch_simt_backward_data,
    launch_simt_backward_weight,
};

pub use cubek::convolution::SimtTiling;

use burn_backend::ops::conv::calculate_conv_output_sizes;

use crate::{kernel::into_contiguous, ops::numeric::empty_device_dtype, tensor::CubeTensor};

/// The kernels' arguments: the padding at the beginning of each dimension (the end is in the
/// output's size).
fn args<const N: usize>(options: &ConvOptions<N>) -> ConvolutionArgs<2> {
    ConvolutionArgs {
        stride: [options.stride[0], options.stride[1]],
        padding: [options.padding[0].0, options.padding[1].0],
        dilation: [options.dilation[0], options.dilation[1]],
    }
}

/// A 2D convolution of NHWC `input` by NHWC `weight` (`[c_out, kh, kw, c_in]`), giving NHWC.
pub fn conv_simt<const N: usize>(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
    tiling: SimtTiling,
) -> Result<CubeTensor, ConvSetupError> {
    // Two spatial dimensions only; read the shapes as such after this.
    if N != 2 {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes two spatial dimensions",
        ));
    }
    let [batch, in_h, in_w, _] = input.meta.shape().dims();
    let [c_out, kh, kw, _] = weight.meta.shape().dims();
    let out_size = calculate_conv_output_sizes(
        &[kh, kw],
        &options.stride[..2],
        &options.padding[..2],
        &options.dilation[..2],
        &[in_h, in_w],
    );
    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, out_size[0], out_size[1], c_out]),
        input.dtype,
    );
    let client = input.client.clone();
    let dtype = dtype_to_storage_type(input.dtype);
    launch_simt(
        &client,
        SimtTensors {
            input: input.binding(),
            weight: weight.binding(),
            bias: bias.map(|bias| into_contiguous(bias).binding()),
            out: out.clone().binding(),
        },
        args(&options),
        options.groups,
        dtype,
        tiling,
    )?;
    Ok(out)
}

/// The input gradient of a stride-1 2D convolution, from the NHWC output gradient and the NHWC
/// weight (`[c_out, kh, kw, c_in]`), giving NHWC.
pub fn conv_data_backward_simt<const N: usize>(
    out_grad: CubeTensor,
    weight: CubeTensor,
    in_shape: Shape,
    options: ConvOptions<N>,
    tiling: SimtTiling,
) -> Result<CubeTensor, ConvSetupError> {
    if N != 2 {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes two spatial dimensions",
        ));
    }
    let in_grad = empty_device_dtype(
        out_grad.client.clone(),
        out_grad.device.clone(),
        in_shape,
        out_grad.dtype,
    );
    let client = out_grad.client.clone();
    let dtype = dtype_to_storage_type(out_grad.dtype);
    launch_simt_backward_data(
        &client,
        SimtBackwardDataTensors {
            out_grad: out_grad.binding(),
            weight: weight.binding(),
            in_grad: in_grad.clone().binding(),
        },
        args(&options),
        options.groups,
        dtype,
        tiling,
    )?;
    Ok(in_grad)
}

/// The weight gradient of a 2D convolution: NHWC `input`, NHWC `out_grad` and the NHWC weight
/// shape (`[c_out, kh, kw, c_in]`), giving the NHWC weight gradient as a view of the
/// `[kh, kw, c_in, c_out]` the kernel writes.
pub fn conv_weight_backward_simt<const N: usize>(
    input: CubeTensor,
    out_grad: CubeTensor,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    if N != 2 {
        return Err(ConvSetupError::Unsupported(
            "the SIMT convolution takes two spatial dimensions",
        ));
    }
    let [c_out, kh, kw, c_in] = weight_shape.dims();
    let dw = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::new([kh, kw, c_in, c_out]),
        input.dtype,
    );
    let dw = crate::ops::permute(dw, &[3, 0, 1, 2]);
    let client = input.client.clone();
    let dtype = dtype_to_storage_type(input.dtype);
    launch_simt_backward_weight(
        &client,
        SimtBackwardWeightTensors {
            input: input.binding(),
            out_grad: out_grad.binding(),
            weight_grad: dw.clone().binding(),
        },
        args(&options),
        options.groups,
        dtype,
    )?;
    Ok(dw)
}

/// The kernels against the definition of a convolution, on the default wgpu device.
#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::super::test_support::{
        Case, case, check_forward, check_forward_where_accepted, check_input_gradient,
        check_input_gradient_where_accepted, check_weight_gradient,
    };
    use super::*;
    use crate::CubeDevice;

    /// Input channels in multiples of the stage; output channels filling one tile, two, and a
    /// tile cut short; positions filling no tile and several; stride, dilation and asymmetric
    /// padding; a reduction over several taps per stage.
    fn cases() -> Vec<Case> {
        let none = [(0, 0); 2];
        let one = [(1, 1); 2];
        vec![
            case("plain", 2, 16, 64, [7, 6], [3, 3], [1, 1], none, [1, 1]),
            case(
                "padded, two tiles",
                3,
                32,
                128,
                [9, 9],
                [3, 3],
                [1, 1],
                one,
                [1, 1],
            ),
            case(
                "channels cut short",
                1,
                16,
                36,
                [8, 7],
                [3, 2],
                [1, 1],
                [(2, 0), (0, 1)],
                [1, 1],
            ),
            case(
                "dilated",
                2,
                16,
                32,
                [10, 9],
                [3, 3],
                [1, 1],
                [(2, 2); 2],
                [2, 2],
            ),
            case(
                "many positions",
                4,
                16,
                32,
                [40, 40],
                [3, 3],
                [1, 1],
                one,
                [1, 1],
            ),
        ]
    }

    /// Input channels short of a stage: a stage spans taps, and the reduction ends partway
    /// through one.
    fn few_channels() -> Vec<Case> {
        vec![
            case(
                "rgb stem: channels padded to 4",
                2,
                3,
                64,
                [20, 19],
                [7, 7],
                [2, 2],
                [(3, 3); 2],
                [1, 1],
            ),
            case(
                "stem",
                2,
                4,
                64,
                [20, 19],
                [7, 7],
                [2, 2],
                [(3, 3); 2],
                [1, 1],
            ),
            case(
                "deep, few positions: the reduction split",
                2,
                64,
                64,
                [5, 5],
                [3, 3],
                [1, 1],
                [(1, 1); 2],
                [1, 1],
            ),
            case(
                "twelve channels",
                2,
                12,
                32,
                [9, 8],
                [3, 3],
                [1, 1],
                [(1, 1); 2],
                [1, 1],
            ),
        ]
    }

    fn strided() -> Vec<Case> {
        let one = [(1, 1); 2];
        vec![
            case("strided", 3, 16, 64, [11, 10], [3, 3], [2, 2], one, [1, 1]),
            case(
                "pointwise strided",
                2,
                32,
                64,
                [8, 8],
                [1, 1],
                [2, 2],
                [(0, 0); 2],
                [1, 1],
            ),
        ]
    }

    fn device() -> CubeDevice {
        CubeDevice::wgpu(cubecl::device::WgpuDeviceKind::DefaultDevice).expect("a wgpu device")
    }

    #[test]
    fn forward_matches_the_definition() {
        let mut cases = cases();
        cases.extend(strided());
        cases.extend(few_channels());
        check_forward(&device(), &cases, &|input, weight, bias, options| {
            conv_simt::<2>(input, weight, bias, options, SimtTiling::NARROW)
        });
        for tiling in [SimtTiling::NARROW_SPLIT, SimtTiling::WIDE] {
            check_forward_where_accepted(
                &device(),
                &cases,
                &move |input, weight, bias, options| {
                    conv_simt::<2>(input, weight, bias, options, tiling)
                },
            );
        }
    }

    /// As for the forward pass, with the roles of the channels swapped: the output gradient's
    /// channels are the reduction, the input's are cut short.
    fn gradient_cases() -> Vec<Case> {
        let none = [(0, 0); 2];
        let one = [(1, 1); 2];
        vec![
            case("plain", 2, 64, 16, [7, 6], [3, 3], [1, 1], none, [1, 1]),
            case(
                "padded, two tiles",
                3,
                128,
                32,
                [9, 9],
                [3, 3],
                [1, 1],
                one,
                [1, 1],
            ),
            case(
                "channels cut short",
                1,
                36,
                16,
                [8, 7],
                [3, 2],
                [1, 1],
                [(2, 0), (0, 1)],
                [1, 1],
            ),
            case(
                "dilated",
                2,
                32,
                16,
                [10, 9],
                [3, 3],
                [1, 1],
                [(2, 2); 2],
                [2, 2],
            ),
            case(
                "many positions",
                4,
                32,
                16,
                [40, 40],
                [3, 3],
                [1, 1],
                one,
                [1, 1],
            ),
            case(
                "twelve channels",
                2,
                32,
                12,
                [9, 8],
                [3, 3],
                [1, 1],
                one,
                [1, 1],
            ),
        ]
    }

    #[test]
    fn input_gradient_matches_the_definition() {
        let cases = gradient_cases();
        check_input_gradient(&device(), &cases, &|out_grad, weight, shape, options| {
            conv_data_backward_simt::<2>(out_grad, weight, shape, options, SimtTiling::NARROW)
        });
        for tiling in [SimtTiling::NARROW_SPLIT, SimtTiling::WIDE] {
            check_input_gradient_where_accepted(
                &device(),
                &cases,
                &move |out_grad, weight, shape, options| {
                    conv_data_backward_simt::<2>(out_grad, weight, shape, options, tiling)
                },
            );
        }
    }

    #[test]
    fn weight_gradient_matches_the_definition() {
        let none = [(0, 0); 2];
        let mut cases = cases();
        cases.extend(strided());
        // Input channels in fours only, and rows cut short of a cube.
        cases.push(case(
            "few input channels",
            2,
            8,
            32,
            [9, 9],
            [3, 3],
            [1, 1],
            none,
            [1, 1],
        ));
        check_weight_gradient(&device(), &cases, &conv_weight_backward_simt::<2>);
    }
}
