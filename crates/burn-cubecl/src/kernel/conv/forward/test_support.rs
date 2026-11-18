//! Convolution kernels against the definition of a convolution, computed on the host, on any
//! device: the cases, the reference and the comparison shared by the kernels' tests.

use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use burn_std::{Shape, TensorData};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeDevice,
    kernel::into_contiguous,
    ops::{from_data, into_data_sync, permute},
    tensor::CubeTensor,
};

fn data(len: usize, seed: u32) -> Vec<f32> {
    (0..len as u32)
        .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 97) as f32 / 97.0 - 0.5)
        .collect()
}

#[derive(Clone)]
pub(super) struct Case {
    name: &'static str,
    batch: usize,
    c_in: usize,
    c_out: usize,
    in_hw: [usize; 2],
    k: [usize; 2],
    options: ConvOptions<2>,
}

impl Case {
    fn out_hw(&self) -> [usize; 2] {
        let size = calculate_conv_output_sizes(
            &self.k,
            &self.options.stride,
            &self.options.padding,
            &self.options.dilation,
            &self.in_hw,
        );
        [size[0], size[1]]
    }

    fn x_shape(&self) -> [usize; 4] {
        [self.batch, self.in_hw[0], self.in_hw[1], self.c_in]
    }

    fn w_shape(&self) -> [usize; 4] {
        [self.c_out, self.k[0], self.k[1], self.c_in]
    }

    fn y_shape(&self) -> [usize; 4] {
        let [oh, ow] = self.out_hw();
        [self.batch, oh, ow, self.c_out]
    }

    /// Calls `f(x index, w index, y index)` for every product the convolution sums, all NHWC
    /// (the weight `[c_out, kh, kw, c_in]`).
    fn for_each_product(&self, mut f: impl FnMut(usize, usize, usize)) {
        let [ih, iw] = self.in_hw;
        let [oh, ow] = self.out_hw();
        let o = &self.options;
        for b in 0..self.batch {
            for y in 0..oh {
                for x in 0..ow {
                    for ky in 0..self.k[0] {
                        let iy = (y * o.stride[0] + ky * o.dilation[0]) as isize
                            - o.padding[0].0 as isize;
                        for kx in 0..self.k[1] {
                            let ix = (x * o.stride[1] + kx * o.dilation[1]) as isize
                                - o.padding[1].0 as isize;
                            if iy < 0 || ix < 0 || iy >= ih as isize || ix >= iw as isize {
                                continue;
                            }
                            let x_at = ((b * ih + iy as usize) * iw + ix as usize) * self.c_in;
                            let w_at = (ky * self.k[1] + kx) * self.c_in;
                            let y_at = ((b * oh + y) * ow + x) * self.c_out;
                            for co in 0..self.c_out {
                                for ci in 0..self.c_in {
                                    f(
                                        x_at + ci,
                                        co * self.k[0] * self.k[1] * self.c_in + w_at + ci,
                                        y_at + co,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn case(
    name: &'static str,
    batch: usize,
    c_in: usize,
    c_out: usize,
    in_hw: [usize; 2],
    k: [usize; 2],
    stride: [usize; 2],
    padding: [(usize, usize); 2],
    dilation: [usize; 2],
) -> Case {
    Case {
        name,
        batch,
        c_in,
        c_out,
        in_hw,
        k,
        options: ConvOptions {
            stride,
            padding,
            dilation,
            groups: 1,
        },
    }
}

fn tensor(values: &[f32], shape: &[usize], device: &CubeDevice) -> CubeTensor {
    from_data(TensorData::new(values.to_vec(), shape.to_vec()), device)
}

/// The NHWC `values` as burn hands them over: an NHWC view of NCHW memory, or with
/// `contiguous`, NHWC memory.
fn nhwc(
    values: &[f32],
    [n, h, w, c]: [usize; 4],
    contiguous: bool,
    device: &CubeDevice,
) -> CubeTensor {
    if contiguous {
        return tensor(values, &[n, h, w, c], device);
    }
    let mut nchw = vec![0f32; values.len()];
    for b in 0..n {
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    nchw[((b * c + ch) * h + y) * w + x] = values[((b * h + y) * w + x) * c + ch];
                }
            }
        }
    }
    permute(tensor(&nchw, &[n, c, h, w], device), &[0, 2, 3, 1])
}

#[track_caller]
fn assert_close(case: &str, actual: CubeTensor, expected: &[f32], shape: [usize; 4]) {
    assert_eq!(
        actual.meta.shape().dims::<4>(),
        shape,
        "{case}: shapes differ"
    );
    let actual = into_data_sync(actual);
    let actual = actual.as_slice::<f32>().unwrap();
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case}: element counts differ"
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-4 * (1.0 + e.abs()),
            "{case}: at {i}, got {a}, want {e}"
        );
    }
}

type Forward = dyn Fn(
    CubeTensor,
    CubeTensor,
    Option<CubeTensor>,
    ConvOptions<2>,
) -> Result<CubeTensor, ConvSetupError>;
type Backward =
    dyn Fn(CubeTensor, CubeTensor, Shape, ConvOptions<2>) -> Result<CubeTensor, ConvSetupError>;

/// `conv` against the definition, with and without bias, on NHWC and NCHW-strided inputs.
pub(super) fn check_forward(device: &CubeDevice, cases: &[Case], conv: &Forward) {
    forward(device, cases, conv, false);
}

/// [`check_forward`] on the cases `conv` accepts, for a variant that declines some by design.
pub(super) fn check_forward_where_accepted(device: &CubeDevice, cases: &[Case], conv: &Forward) {
    forward(device, cases, conv, true);
}

fn forward(device: &CubeDevice, cases: &[Case], conv: &Forward, may_decline: bool) {
    for case in cases {
        for (with_bias, contiguous) in [(false, true), (true, true), (true, false)] {
            let x = data(case.x_shape().iter().product(), 1);
            let w = data(case.w_shape().iter().product(), 2);
            let bias = data(case.c_out, 3);
            let mut expected = vec![0f32; case.y_shape().iter().product()];
            if with_bias {
                for (i, v) in expected.iter_mut().enumerate() {
                    *v = bias[i % case.c_out];
                }
            }
            case.for_each_product(|xi, wi, yi| expected[yi] += x[xi] * w[wi]);

            let actual = match conv(
                nhwc(&x, case.x_shape(), contiguous, device),
                nhwc(&w, case.w_shape(), contiguous, device),
                with_bias.then(|| tensor(&bias, &[case.c_out], device)),
                case.options.clone(),
            ) {
                Ok(actual) => actual,
                Err(_) if may_decline => continue,
                Err(err) => panic!("{}: the kernel declined: {err:?}", case.name),
            };
            assert_close(case.name, actual, &expected, case.y_shape());
        }
    }
}

/// `dgrad(out_grad, weight, input shape, options)` against the definition.
pub(super) fn check_input_gradient(device: &CubeDevice, cases: &[Case], dgrad: &Backward) {
    input_gradient(device, cases, dgrad, false);
}

/// [`check_input_gradient`] on the cases `dgrad` accepts.
pub(super) fn check_input_gradient_where_accepted(
    device: &CubeDevice,
    cases: &[Case],
    dgrad: &Backward,
) {
    input_gradient(device, cases, dgrad, true);
}

fn input_gradient(device: &CubeDevice, cases: &[Case], dgrad: &Backward, may_decline: bool) {
    for case in cases {
        for contiguous in [true, false] {
            let w = data(case.w_shape().iter().product(), 2);
            let dy = data(case.y_shape().iter().product(), 4);
            let mut expected = vec![0f32; case.x_shape().iter().product()];
            case.for_each_product(|xi, wi, yi| expected[xi] += dy[yi] * w[wi]);

            let actual = match dgrad(
                nhwc(&dy, case.y_shape(), contiguous, device),
                nhwc(&w, case.w_shape(), contiguous, device),
                Shape::new(case.x_shape()),
                case.options.clone(),
            ) {
                Ok(actual) => actual,
                Err(_) if may_decline => continue,
                Err(err) => panic!("{}: the kernel declined: {err:?}", case.name),
            };
            assert_close(case.name, actual, &expected, case.x_shape());
        }
    }
}

/// `wgrad(input, out_grad, weight shape, options)` against the definition.
pub(super) fn check_weight_gradient(device: &CubeDevice, cases: &[Case], wgrad: &Backward) {
    for case in cases {
        for contiguous in [true, false] {
            let x = data(case.x_shape().iter().product(), 1);
            let dy = data(case.y_shape().iter().product(), 4);
            let mut expected = vec![0f32; case.w_shape().iter().product()];
            case.for_each_product(|xi, wi, yi| expected[wi] += x[xi] * dy[yi]);

            let actual = wgrad(
                nhwc(&x, case.x_shape(), contiguous, device),
                nhwc(&dy, case.y_shape(), contiguous, device),
                Shape::new(case.w_shape()),
                case.options.clone(),
            )
            .expect("the kernel accepts these shapes");
            assert_close(
                case.name,
                into_contiguous(actual),
                &expected,
                case.w_shape(),
            );
        }
    }
}
