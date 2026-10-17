//! Register-blocked direct convolution for f32, stride 1, no dilation.
//!
//! im2col + gemm spends memory traffic on expanding the input by `kh * kw`, which pays off when
//! the gemm's reduction (`c_in * kh * kw`) is long. For convolutions with few input channels it
//! is short, and the expansion dominates: on a small CNN (8->32 and 32->64 channels, 3x3)
//! im2col + gemm ran at about half of libtorch's speed. libtorch (oneDNN `jit:avx2`) runs these as
//! direct convolutions instead, which is what this is:
//!
//! - A block of `OCB` output channels by `NV` vectors of output columns is accumulated in
//!   registers.
//! - For each input channel and kernel tap, `NV` vectors of the input row are loaded, `OCB`
//!   weights broadcast, and `OCB * NV` fused multiply-adds issued; the accumulators never touch
//!   memory until the block is done.
//! - Tasks are (image, block of output rows); within a task all output channel blocks run over
//!   the same input rows, which stay in cache.
//!
//! Columns whose taps reach into the padding, and the columns left over after the vector blocks,
//! take a scalar loop.

use alloc::vec;
use burn_backend::DType;
use burn_backend::ops::ConvOptions;
use burn_backend::ops::conv::calculate_conv_output_size;
use burn_std::{Bytes, Shape};
use macerator::{Scalar, Simd, VMulAdd, vload_unaligned, vstore_unaligned};

use crate::{FlexTensor, Layout};

/// Output channels per register block.
const OCB: usize = 4;
/// Vectors of output columns per register block. With `OCB` this is 12 accumulators, which with
/// the `NV` input vectors and a broadcast weight fit AVX2's 16 vector registers.
const NV: usize = 3;
/// Output rows per task.
const ROWS_PER_TASK: usize = 8;
/// The widest f32 vector the kernel can be dispatched for (AVX-512), which the broadcast
/// weights are stored at.
const MAX_LANES: usize = 16;

/// Whether the direct kernel handles this convolution (shapes as conv3d, `[n, c, d, h, w]`).
///
/// The kernel skips taps that land in the padding instead of multiplying them by zero. That is
/// the same result unless a weight is non-finite (`0 * inf` is NaN), so padded convolutions
/// with a non-finite weight are left to im2col, which multiplies the zeros.
pub(crate) fn applies(x_shape: &[usize], weight: &FlexTensor, options: &ConvOptions<3>) -> bool {
    let w_shape = weight.layout().shape();
    let in_channels_taps = w_shape[1] * w_shape[3] * w_shape[4];
    let shape_fits = weight.dtype() == DType::F32
        && options.groups == 1
        && w_shape[2] == 1
        && x_shape[2] == 1
        && options.stride == [1, 1, 1]
        && options.dilation == [1, 1, 1]
        && options.padding_begin()[0] == 0
        && in_channels_taps <= MAX_REDUCTION;
    if !shape_fits {
        return false;
    }
    let padded = options.padding_begin().iter().any(|&p| p != 0);
    !padded
        || weight
            .to_contiguous()
            .storage::<f32>()
            .iter()
            .all(|w| w.is_finite())
}

/// Longest reduction (`c_in * kh * kw`) the direct kernel takes over from im2col + gemm.
///
/// Measured on a small CNN (x86 AVX2): the direct kernel won at 9 (1->8 channels, 1.7x)
/// and 72 (8->32, 1.2-1.3x), tied at 288 and lost from 576 up, where im2col's gemm reuses its
/// packed operands better.
const MAX_REDUCTION: usize = 128;

pub(crate) fn conv2d_direct_f32(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();
    let xs = x.layout().shape();
    let ws = weight.layout().shape();
    let (batch, c_in, in_h, in_w) = (xs[0], xs[1], xs[3], xs[4]);
    let (c_out, kh, kw) = (ws[0], ws[3], ws[4]);
    let [_, pad_h, pad_w] = options.padding_begin();
    let out_h = calculate_conv_output_size(kh, 1, pad_h, 1, in_h);
    let out_w = calculate_conv_output_size(kw, 1, pad_w, 1, in_w);

    let x_data: &[f32] = x.storage();
    let w_data: &[f32] = weight.storage();
    let bias = bias.map(|b| b.to_contiguous().storage::<f32>().to_vec());

    // Weights repacked as [oc block][ic][kh][kw][OCB][MAX_LANES]: every weight already
    // broadcast to a full vector (zero past c_out), so the kernel's hot loop loads vectors
    // instead of broadcasting scalars. macerator's generic `splat` goes through memory, and in
    // this loop broadcasting cost more than the multiply-adds.
    let blocks = c_out.div_ceil(OCB);
    let taps = c_in * kh * kw;
    let mut packed = vec![0.0f32; blocks * taps * OCB * MAX_LANES];
    for oc in 0..c_out {
        let (blk, o) = (oc / OCB, oc % OCB);
        for t in 0..taps {
            let at = ((blk * taps + t) * OCB + o) * MAX_LANES;
            packed[at..at + MAX_LANES].fill(w_data[oc * taps + t]);
        }
    }

    let geometry = Geometry {
        c_in,
        c_out,
        in_h,
        in_w,
        out_h,
        out_w,
        kh,
        kw,
        pad_h,
        pad_w,
    };
    let mut out = vec![0.0f32; batch * c_out * out_h * out_w];
    let dst = crate::ops::SendMutPtr::new(out.as_mut_ptr());
    let row_blocks = out_h.div_ceil(ROWS_PER_TASK);
    crate::parallel::for_each_index(batch * row_blocks, |task| {
        let (b, rb) = (task / row_blocks, task % row_blocks);
        let rows = rb * ROWS_PER_TASK..((rb + 1) * ROWS_PER_TASK).min(out_h);
        let x_img = &x_data[b * c_in * in_h * in_w..][..c_in * in_h * in_w];
        for blk in 0..blocks {
            let w_blk = &packed[blk * taps * OCB * MAX_LANES..][..taps * OCB * MAX_LANES];
            let biases: [f32; OCB] = core::array::from_fn(|o| {
                let oc = blk * OCB + o;
                match &bias {
                    Some(bias) if oc < c_out => bias[oc],
                    _ => 0.0,
                }
            });
            for oh in rows.clone() {
                // SAFETY: tasks cover disjoint (image, row) blocks; the kernel writes only the
                // rows `oh` of this block's output channels in image `b`.
                unsafe {
                    conv_row(
                        x_img,
                        w_blk,
                        &biases,
                        &dst,
                        (b * c_out + blk * OCB) * out_h * out_w,
                        blk * OCB,
                        oh,
                        &geometry,
                    )
                };
            }
        }
    });

    FlexTensor::new(
        Bytes::from_elems(out),
        Layout::contiguous(Shape::from(vec![batch, c_out, 1, out_h, out_w])),
        DType::F32,
    )
}

struct Geometry {
    c_in: usize,
    c_out: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
}

/// One output row of one block of output channels.
///
/// # Safety
/// `dst.ptr_add(out_base)` must point at output channel `first_oc` of this image, and no other
/// thread may write row `oh` of channels `first_oc..first_oc + OCB` of it.
#[allow(clippy::too_many_arguments)]
#[macerator::with_simd]
unsafe fn conv_row<S: Simd>(
    x: &[f32],
    w: &[f32],
    biases: &[f32; OCB],
    dst: &crate::ops::SendMutPtr<f32>,
    out_base: usize,
    first_oc: usize,
    oh: usize,
    g: &Geometry,
) {
    let lanes = f32::lanes::<S>();
    // Columns whose every tap lands inside the input row: `ow + kx - pad_w` in `0..in_w`.
    let interior_lo = g.pad_w;
    let interior_hi = (g.in_w + g.pad_w + 1)
        .saturating_sub(g.kw)
        .min(g.out_w)
        .max(interior_lo);
    let interior = interior_hi - interior_lo;
    let row = RowCtx {
        x,
        w,
        biases,
        dst,
        out_base,
        channels: OCB.min(g.c_out - first_oc),
        oh,
        g,
        // Kernel rows that land inside the input for this output row.
        ky_lo: g.pad_h.saturating_sub(oh),
        ky_hi: (g.in_h + g.pad_h).saturating_sub(oh).min(g.kh),
    };

    // Whole blocks of `NV` vectors; the last block is moved back to end at `interior_hi`,
    // recomputing a few columns rather than leaving a remainder. Rows narrower than one block
    // do the same with single vectors.
    let (mut vector_lo, mut vector_hi) = (interior_lo, interior_lo);
    if interior >= NV * lanes {
        let block = NV * lanes;
        let mut ow = interior_lo;
        while ow < interior_hi {
            let at = ow.min(interior_hi - block);
            unsafe { vector_block::<S, NV>(&row, at) };
            ow += block;
        }
        vector_hi = interior_hi;
    } else if interior >= lanes {
        let mut ow = interior_lo;
        while ow < interior_hi {
            let at = ow.min(interior_hi - lanes);
            unsafe { vector_block::<S, 1>(&row, at) };
            ow += lanes;
        }
        vector_hi = interior_hi;
    } else {
        vector_lo = interior_hi;
    }

    // The padded edges, and rows too narrow for a vector.
    for ow in (0..vector_lo).chain(vector_hi..g.out_w) {
        let (x, w, biases, ky_lo, ky_hi, channels) =
            (row.x, row.w, row.biases, row.ky_lo, row.ky_hi, row.channels);
        let plane = g.out_h * g.out_w;
        let mut acc = *biases;
        for ic in 0..g.c_in {
            for ky in ky_lo..ky_hi {
                let row = &x[(ic * g.in_h + oh + ky - g.pad_h) * g.in_w..][..g.in_w];
                let wt = &w[((ic * g.kh + ky) * g.kw) * OCB * MAX_LANES..];
                for kx in 0..g.kw {
                    let ix = ow as isize + kx as isize - g.pad_w as isize;
                    if ix < 0 || ix >= g.in_w as isize {
                        continue;
                    }
                    let v = row[ix as usize];
                    for (o, a) in acc.iter_mut().enumerate() {
                        *a += wt[(kx * OCB + o) * MAX_LANES] * v;
                    }
                }
            }
        }
        for (o, a) in acc.iter().enumerate().take(channels) {
            // SAFETY: in bounds of this channel's row; the caller guarantees exclusivity.
            unsafe {
                dst.ptr_add(out_base + o * plane + oh * g.out_w + ow)
                    .write(*a)
            };
        }
    }
}

/// What one output row of one channel block needs, passed to its vector blocks.
struct RowCtx<'a> {
    x: &'a [f32],
    w: &'a [f32],
    biases: &'a [f32; OCB],
    dst: &'a crate::ops::SendMutPtr<f32>,
    out_base: usize,
    channels: usize,
    oh: usize,
    g: &'a Geometry,
    ky_lo: usize,
    ky_hi: usize,
}

/// `OCB` output channels by `V` vectors of output columns from `ow`, all of whose taps must land
/// inside the input row, accumulated in registers and stored.
///
/// # Safety
/// As [`conv_row`]; `ow..ow + V * lanes` must be interior columns.
#[inline(always)]
unsafe fn vector_block<S: Simd, const V: usize>(r: &RowCtx<'_>, ow: usize) {
    let g = r.g;
    let lanes = f32::lanes::<S>();
    let mut acc: [[_; V]; OCB] = core::array::from_fn(|o| [r.biases[o].splat::<S>(); V]);
    for ic in 0..g.c_in {
        for ky in r.ky_lo..r.ky_hi {
            let row = &r.x[(ic * g.in_h + r.oh + ky - g.pad_h) * g.in_w..][..g.in_w];
            let wt = &r.w[((ic * g.kh + ky) * g.kw) * OCB * MAX_LANES..];
            for kx in 0..g.kw {
                let base = ow + kx - g.pad_w;
                // SAFETY: interior columns keep `base..base + V * lanes` inside the row.
                let xv: [_; V] = core::array::from_fn(|v| unsafe {
                    vload_unaligned(row.as_ptr().add(base + v * lanes))
                });
                for (o, acc_o) in acc.iter_mut().enumerate() {
                    // SAFETY: each broadcast weight holds `MAX_LANES >= lanes` copies.
                    let wv =
                        unsafe { vload_unaligned(wt.as_ptr().add((kx * OCB + o) * MAX_LANES)) };
                    for v in 0..V {
                        acc_o[v] = f32::vmul_add(wv, xv[v], acc_o[v]);
                    }
                }
            }
        }
    }
    let plane = g.out_h * g.out_w;
    for (o, acc_o) in acc.iter().enumerate().take(r.channels) {
        for (v, a) in acc_o.iter().enumerate() {
            // SAFETY: in bounds of this channel's row; the caller guarantees exclusivity.
            unsafe {
                vstore_unaligned(
                    r.dst
                        .ptr_add(r.out_base + o * plane + r.oh * g.out_w + ow + v * lanes),
                    *a,
                )
            };
        }
    }
}
