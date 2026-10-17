//! Batch normalization, one fused pass per channel.
//!
//! Burn's default `batch_norm_train` and its backward compose the statistics out of tensor ops:
//! a transpose to channels-first (a full copy), means, a subtraction, a square, another
//! transpose, and the normalization, each a separate pass over the activation, plus as many
//! again for the backward. Here each channel is one task that reads its slices of the input a
//! few times while they are cache-hot and writes its output once; channels run in parallel.
//!
//! Statistics are accumulated in eight independent lanes (so the loop vectorizes) per slice and
//! folded into `f64` per slice, which keeps sums over large batches accurate.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_std::{Bytes, Shape};

use crate::{FlexTensor, Layout};

/// The element types normalized natively; half precision goes through `f32`.
trait Float: num_traits::Float + burn_backend::Element + bytemuck::Pod + Send + Sync {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl Float for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl Float for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// `[batch, channels, spatial...]` seen as `batch` slices of `spatial` elements per channel.
#[derive(Clone, Copy)]
struct Geometry {
    batch: usize,
    channels: usize,
    spatial: usize,
}

impl Geometry {
    fn of(shape: &Shape) -> Self {
        let batch = shape[0];
        let channels = shape[1];
        let spatial = shape.num_elements() / (batch * channels).max(1);
        Self {
            batch,
            channels,
            spatial,
        }
    }

    /// Number of elements per channel.
    fn per_channel(&self) -> usize {
        self.batch * self.spatial
    }

    /// Offset of channel `c`'s slice in image `n`.
    fn slice(&self, n: usize, c: usize) -> usize {
        (n * self.channels + c) * self.spatial
    }
}

/// Sum of `f(i)` over `0..len`, in eight lanes so the loop vectorizes.
#[inline]
fn lane_sum<T: Float>(len: usize, f: impl Fn(usize) -> T) -> f64 {
    let mut lanes = [T::zero(); 8];
    let whole = len / 8 * 8;
    for i in (0..whole).step_by(8) {
        for (l, lane) in lanes.iter_mut().enumerate() {
            *lane = *lane + f(i + l);
        }
    }
    let mut total = lanes.iter().map(|&v| v.to_f64()).sum::<f64>();
    for i in whole..len {
        total += f(i).to_f64();
    }
    total
}

/// Runs `f(c)` for every channel, in parallel when rayon is enabled.
fn for_each_channel(channels: usize, f: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        (0..channels).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "rayon"))]
    (0..channels).for_each(f);
}

fn values<T: Float>(t: &FlexTensor) -> Vec<T> {
    t.to_contiguous().storage::<T>().to_vec()
}

fn tensor<T: Float>(data: Vec<T>, shape: Shape, dtype: DType) -> FlexTensor {
    FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), dtype)
}

/// `y = (x - mean) / sqrt(variance + epsilon) * gamma + beta`, with the given statistics.
fn batch_norm_typed<T: Float>(
    x: FlexTensor,
    gamma: &[T],
    beta: &[T],
    mean: &[T],
    variance: &[T],
    epsilon: f64,
    dtype: DType,
) -> FlexTensor {
    let x = x.to_contiguous();
    let shape = x.layout().shape().clone();
    let g = Geometry::of(&shape);
    let src: &[T] = x.storage();
    let mut out = vec![T::zero(); shape.num_elements()];
    let dst = crate::ops::SendMutPtr::new(out.as_mut_ptr());
    for_each_channel(g.channels, |c| {
        let inv_std = 1.0 / (variance[c].to_f64() + epsilon).sqrt();
        let scale = T::from_f64(gamma[c].to_f64() * inv_std);
        let shift = T::from_f64(beta[c].to_f64() - mean[c].to_f64() * gamma[c].to_f64() * inv_std);
        for n in 0..g.batch {
            let o = g.slice(n, c);
            // SAFETY: each channel writes only its own slices.
            let out = unsafe { core::slice::from_raw_parts_mut(dst.ptr_add(o), g.spatial) };
            for (y, &v) in out.iter_mut().zip(&src[o..o + g.spatial]) {
                *y = v * scale + shift;
            }
        }
    });
    tensor(out, shape, dtype)
}

/// Batch statistics (mean, biased variance) and the normalized output.
fn batch_norm_train_typed<T: Float>(
    x: FlexTensor,
    gamma: &[T],
    beta: &[T],
    epsilon: f64,
    dtype: DType,
) -> (FlexTensor, FlexTensor, FlexTensor) {
    let x = x.to_contiguous();
    let shape = x.layout().shape().clone();
    let g = Geometry::of(&shape);
    let src: &[T] = x.storage();
    let mut out = vec![T::zero(); shape.num_elements()];
    let mut means = vec![T::zero(); g.channels];
    let mut variances = vec![T::zero(); g.channels];
    let dst = crate::ops::SendMutPtr::new(out.as_mut_ptr());
    let means_ptr = crate::ops::SendMutPtr::new(means.as_mut_ptr());
    let variances_ptr = crate::ops::SendMutPtr::new(variances.as_mut_ptr());
    let count = g.per_channel() as f64;
    for_each_channel(g.channels, |c| {
        let slices = || (0..g.batch).map(|n| &src[g.slice(n, c)..][..g.spatial]);
        let mean = slices().map(|s| lane_sum(s.len(), |i| s[i])).sum::<f64>() / count;
        let m = T::from_f64(mean);
        // Two passes rather than E[x^2] - E[x]^2, which cancels badly when the mean is large.
        let variance = slices()
            .map(|s| lane_sum(s.len(), |i| (s[i] - m) * (s[i] - m)))
            .sum::<f64>()
            / count;
        let inv_std = 1.0 / (variance + epsilon).sqrt();
        let scale = T::from_f64(gamma[c].to_f64() * inv_std);
        let shift = T::from_f64(beta[c].to_f64() - mean * gamma[c].to_f64() * inv_std);
        for n in 0..g.batch {
            let o = g.slice(n, c);
            // SAFETY: each channel writes only its own slices and its own statistics.
            let out = unsafe { core::slice::from_raw_parts_mut(dst.ptr_add(o), g.spatial) };
            for (y, &v) in out.iter_mut().zip(&src[o..o + g.spatial]) {
                *y = v * scale + shift;
            }
        }
        unsafe {
            means_ptr.write(c, m);
            variances_ptr.write(c, T::from_f64(variance));
        }
    });
    let channels = Shape::new([g.channels]);
    (
        tensor(out, shape, dtype),
        tensor(means, channels.clone(), dtype),
        tensor(variances, channels, dtype),
    )
}

/// Gradients of [`batch_norm_train_typed`] for `x`, `gamma` and `beta`:
///
/// `x_hat = (x - mean) * inv_std`, `d_beta = sum(dy)`, `d_gamma = sum(dy * x_hat)`,
/// `dx = gamma * inv_std * (dy - d_beta / m - x_hat * d_gamma / m)` over each channel's `m`
/// elements.
#[allow(clippy::too_many_arguments)]
fn batch_norm_train_backward_typed<T: Float>(
    x: FlexTensor,
    gamma: &[T],
    mean: &[T],
    variance: &[T],
    epsilon: f64,
    output_grad: FlexTensor,
    dtype: DType,
) -> (FlexTensor, FlexTensor, FlexTensor) {
    let x = x.to_contiguous();
    let dy = output_grad.to_contiguous();
    let shape = x.layout().shape().clone();
    let g = Geometry::of(&shape);
    let xs: &[T] = x.storage();
    let dys: &[T] = dy.storage();
    let mut dx = vec![T::zero(); shape.num_elements()];
    let mut d_gamma = vec![T::zero(); g.channels];
    let mut d_beta = vec![T::zero(); g.channels];
    let dx_ptr = crate::ops::SendMutPtr::new(dx.as_mut_ptr());
    let d_gamma_ptr = crate::ops::SendMutPtr::new(d_gamma.as_mut_ptr());
    let d_beta_ptr = crate::ops::SendMutPtr::new(d_beta.as_mut_ptr());
    let count = g.per_channel() as f64;
    for_each_channel(g.channels, |c| {
        let inv_std = 1.0 / (variance[c].to_f64() + epsilon).sqrt();
        let (m, istd) = (mean[c], T::from_f64(inv_std));
        let mut sum_dy = 0.0;
        let mut sum_dy_xhat = 0.0;
        for n in 0..g.batch {
            let o = g.slice(n, c);
            let (xs, dys) = (&xs[o..o + g.spatial], &dys[o..o + g.spatial]);
            sum_dy += lane_sum(g.spatial, |i| dys[i]);
            sum_dy_xhat += lane_sum(g.spatial, |i| dys[i] * (xs[i] - m) * istd);
        }
        let scale = T::from_f64(gamma[c].to_f64() * inv_std);
        let mean_dy = T::from_f64(sum_dy / count);
        let mean_dy_xhat = T::from_f64(sum_dy_xhat / count);
        for n in 0..g.batch {
            let o = g.slice(n, c);
            // SAFETY: each channel writes only its own slices and its own gradients.
            let out = unsafe { core::slice::from_raw_parts_mut(dx_ptr.ptr_add(o), g.spatial) };
            for ((d, &v), &gy) in out
                .iter_mut()
                .zip(&xs[o..o + g.spatial])
                .zip(&dys[o..o + g.spatial])
            {
                let x_hat = (v - m) * istd;
                *d = scale * (gy - mean_dy - x_hat * mean_dy_xhat);
            }
        }
        unsafe {
            d_gamma_ptr.write(c, T::from_f64(sum_dy_xhat));
            d_beta_ptr.write(c, T::from_f64(sum_dy));
        }
    });
    let channels = Shape::new([g.channels]);
    (
        tensor(dx, shape, dtype),
        tensor(d_gamma, channels.clone(), dtype),
        tensor(d_beta, channels, dtype),
    )
}

/// Runs `f` on f32 or f64 tensors as they are, and on half-precision tensors converted to f32
/// and back.
fn in_float<R>(
    tensors: Vec<FlexTensor>,
    f32_op: impl FnOnce(Vec<FlexTensor>) -> R,
    f64_op: impl FnOnce(Vec<FlexTensor>) -> R,
    back: impl Fn(R, DType) -> R,
) -> R {
    use super::module::cast_to_f32;
    use burn_std::{bf16, f16};
    let dtype = tensors[0].dtype();
    match dtype {
        DType::F32 => f32_op(tensors),
        DType::F64 => f64_op(tensors),
        DType::F16 => back(
            f32_op(
                tensors
                    .into_iter()
                    .map(|t| cast_to_f32::<f16>(t, f16::to_f32))
                    .collect(),
            ),
            dtype,
        ),
        DType::BF16 => back(
            f32_op(
                tensors
                    .into_iter()
                    .map(|t| cast_to_f32::<bf16>(t, bf16::to_f32))
                    .collect(),
            ),
            dtype,
        ),
        dtype => panic!("batch_norm: unsupported dtype {dtype:?}"),
    }
}

fn cast_back(t: FlexTensor, dtype: DType) -> FlexTensor {
    use super::module::cast_from_f32;
    use burn_std::{bf16, f16};
    match dtype {
        DType::F16 => cast_from_f32::<f16>(t, f16::from_f32),
        DType::BF16 => cast_from_f32::<bf16>(t, bf16::from_f32),
        _ => t,
    }
}

fn cast_back3(
    (a, b, c): (FlexTensor, FlexTensor, FlexTensor),
    dtype: DType,
) -> (FlexTensor, FlexTensor, FlexTensor) {
    (
        cast_back(a, dtype),
        cast_back(b, dtype),
        cast_back(c, dtype),
    )
}

pub(crate) fn batch_norm(
    x: FlexTensor,
    gamma: FlexTensor,
    beta: FlexTensor,
    mean: FlexTensor,
    variance: FlexTensor,
    epsilon: f64,
) -> FlexTensor {
    in_float(
        vec![x, gamma, beta, mean, variance],
        |t| {
            let [x, g, b, m, v] = <[FlexTensor; 5]>::try_from(t).ok().unwrap();
            batch_norm_typed::<f32>(
                x,
                &values(&g),
                &values(&b),
                &values(&m),
                &values(&v),
                epsilon,
                DType::F32,
            )
        },
        |t| {
            let [x, g, b, m, v] = <[FlexTensor; 5]>::try_from(t).ok().unwrap();
            batch_norm_typed::<f64>(
                x,
                &values(&g),
                &values(&b),
                &values(&m),
                &values(&v),
                epsilon,
                DType::F64,
            )
        },
        cast_back,
    )
}

pub(crate) fn batch_norm_train(
    x: FlexTensor,
    gamma: FlexTensor,
    beta: FlexTensor,
    epsilon: f64,
) -> (FlexTensor, FlexTensor, FlexTensor) {
    in_float(
        vec![x, gamma, beta],
        |t| {
            let [x, g, b] = <[FlexTensor; 3]>::try_from(t).ok().unwrap();
            batch_norm_train_typed::<f32>(x, &values(&g), &values(&b), epsilon, DType::F32)
        },
        |t| {
            let [x, g, b] = <[FlexTensor; 3]>::try_from(t).ok().unwrap();
            batch_norm_train_typed::<f64>(x, &values(&g), &values(&b), epsilon, DType::F64)
        },
        cast_back3,
    )
}

pub(crate) fn batch_norm_train_backward(
    x: FlexTensor,
    gamma: FlexTensor,
    mean: FlexTensor,
    variance: FlexTensor,
    epsilon: f64,
    output_grad: FlexTensor,
) -> (FlexTensor, FlexTensor, FlexTensor) {
    in_float(
        vec![x, gamma, mean, variance, output_grad],
        |t| {
            let [x, g, m, v, dy] = <[FlexTensor; 5]>::try_from(t).ok().unwrap();
            batch_norm_train_backward_typed::<f32>(
                x,
                &values(&g),
                &values(&m),
                &values(&v),
                epsilon,
                dy,
                DType::F32,
            )
        },
        |t| {
            let [x, g, m, v, dy] = <[FlexTensor; 5]>::try_from(t).ok().unwrap();
            batch_norm_train_backward_typed::<f64>(
                x,
                &values(&g),
                &values(&m),
                &values(&v),
                epsilon,
                dy,
                DType::F64,
            )
        },
        cast_back3,
    )
}
