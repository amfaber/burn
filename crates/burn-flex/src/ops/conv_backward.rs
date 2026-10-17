//! Convolution backward passes, as im2col + gemm like the forward pass in `conv.rs`.
//!
//! Without these, Flex takes burn's generic backward: the input gradient as a transposed
//! convolution, and the weight gradient as a forward convolution of the input with the output
//! gradient as its kernel. That kernel spans the whole output image, which is the worst case
//! for im2col; on a ResNet18 and a small CNN both gradients ran 2-11x slower than libtorch.
//!
//! Here, with `col` the im2col block of the input and `dy` the output gradient laid out like the
//! forward gemm's result (`[c_out, positions]` per group):
//!
//! - weight gradient: `dw^T[col_len, c_out] = sum over position tiles of col @ dy^T`, split
//!   into tasks by (channel chunk, position range) whose partial sums are added at the end;
//! - input gradient: `dcol = w^T @ dy` per tile, scattered back over the input (col2im), with
//!   tasks split by (image, channel chunk) so that no two tasks add into the same element.
//!
//! All dimensions are handled as 3D; conv1d and conv2d reshape into it, as the forward does.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::ConvOptions;
use burn_backend::ops::conv::calculate_conv_output_size;
use burn_std::{Bytes, ElementAdd, Shape};

use crate::{FlexTensor, Layout};

/// The element types the backward passes run natively.
pub(crate) trait ConvElement:
    bytemuck::Pod + burn_backend::Element + ElementAdd + Send + Sync
{
}
impl<T: bytemuck::Pod + burn_backend::Element + ElementAdd + Send + Sync> ConvElement for T {}

/// Most im2col rows per channel chunk: keeps accumulators and gemm operands cache-sized.
const ROWS_PER_CHUNK: usize = 512;
/// Fewest im2col rows per channel chunk worth a gemm of its own.
const MIN_ROWS_PER_CHUNK: usize = 64;
/// Size of an im2col tile for the weight gradient, whose gemms reduce over the tile.
///
/// This and the other sizes here were picked by sweeping them on a ResNet18 and a small CNN
/// on an x86 AVX2 laptop (Core Ultra 7 155H).
const DW_COL_BLOCK_BYTES: usize = 512 << 10;
/// Size of a `dcol` tile for the input gradient.
const DX_COL_BLOCK_BYTES: usize = 512 << 10;
/// Output positions an input-gradient task should cover at least, when images are small.
const DX_MIN_POSITIONS_PER_TASK: usize = 256;

/// The shape and index arithmetic of one convolution, for the gradients' im2col.
struct ConvGeometry {
    in_dims: [usize; 3],
    out_dims: [usize; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    dilation: [usize; 3],
    padding: [usize; 3],
    channels_in: usize,
    channels_out: usize,
    channels_per_group: usize,
    out_channels_per_group: usize,
}

impl ConvGeometry {
    fn spatial_in(&self) -> usize {
        self.in_dims.iter().product()
    }

    fn spatial_out(&self) -> usize {
        self.out_dims.iter().product()
    }

    /// Calls `f(batch, first position in the image, first index in the run, run length)` for
    /// each run of `start..end` (positions numbered across the batch) that stays within one
    /// output row of one image.
    fn for_each_row_run(
        &self,
        start: usize,
        end: usize,
        mut f: impl FnMut(usize, usize, usize, usize),
    ) {
        let spatial_out = self.spatial_out();
        let out_w = self.out_dims[2];
        let mut q = start;
        while q < end {
            let b = q / spatial_out;
            let p = q % spatial_out;
            let len = (out_w - p % out_w).min(end - q);
            f(b, p, q - start, len);
            q += len;
        }
    }
}

/// Write the im2col block of group `g`, input channels `channels` of the group, for output
/// positions `start..end` (numbered across the batch) to `col`: row `(c, kd, kh, kw)` holds,
/// for each position, the input value that kernel tap multiplies, or zero in the padding. Rows
/// are `row_stride` apart.
///
/// Positions are walked a run within one output row at a time, so each kernel tap reads one
/// input row; at stride 1 its valid part is a contiguous copy.
///
/// # Safety
/// `col` must be valid for writes of one row per (channel, kernel tap) of `end - start`
/// elements, `row_stride` apart, and no other thread may write those elements.
#[allow(clippy::too_many_arguments)]
unsafe fn im2col<T: bytemuck::Pod + Copy>(
    col: *mut T,
    row_stride: usize,
    x: &[T],
    conv: &ConvGeometry,
    g: usize,
    channels: core::ops::Range<usize>,
    start: usize,
    end: usize,
) {
    let [in_d, in_h, in_w] = conv.in_dims;
    let [_, out_h, out_w] = conv.out_dims;
    let [kd_n, kh_n, kw_n] = conv.kernel;
    let spatial_in = conv.spatial_in();
    let zero: T = bytemuck::Zeroable::zeroed();
    let sw = conv.stride[2];

    conv.for_each_row_run(start, end, |b, p, offset, len| {
        let od = p / (out_h * out_w);
        let oh = (p / out_w) % out_h;
        let ow0 = p % out_w;
        let x_group = &x[(b * conv.channels_in + g * conv.channels_per_group) * spatial_in..]
            [..conv.channels_per_group * spatial_in];
        let mut row = 0;
        for c in channels.clone() {
            let x_c = &x_group[c * spatial_in..][..spatial_in];
            for kd in 0..kd_n {
                let id = (od * conv.stride[0] + kd * conv.dilation[0]) as isize
                    - conv.padding[0] as isize;
                for kh in 0..kh_n {
                    let ih = (oh * conv.stride[1] + kh * conv.dilation[1]) as isize
                        - conv.padding[1] as isize;
                    let row_valid = id >= 0 && id < in_d as isize && ih >= 0 && ih < in_h as isize;
                    for kw in 0..kw_n {
                        // SAFETY: the caller guarantees `col` covers `col_len` rows of
                        // `end - start` elements; `offset + len <= end - start`.
                        let out = unsafe {
                            core::slice::from_raw_parts_mut(col.add(row * row_stride + offset), len)
                        };
                        row += 1;
                        if !row_valid {
                            out.fill(zero);
                            continue;
                        }
                        let x_row = &x_c[(id as usize * in_h + ih as usize) * in_w..][..in_w];
                        // Input column of output column ow is ow * sw + w_off; find the outputs
                        // whose input column is inside the row.
                        let w_off = (kw * conv.dilation[2]) as isize - conv.padding[2] as isize;
                        let first = if w_off >= 0 {
                            0
                        } else {
                            ((-w_off) as usize).div_ceil(sw)
                        };
                        let last = if (in_w as isize) <= w_off {
                            0
                        } else {
                            ((in_w as isize - 1 - w_off) as usize) / sw + 1
                        };
                        let lo = first.clamp(ow0, ow0 + len) - ow0;
                        let hi = last.clamp(ow0, ow0 + len) - ow0;
                        out[..lo].fill(zero);
                        if hi > lo {
                            out[hi..].fill(zero);
                            let iw0 = ((ow0 + lo) * sw) as isize + w_off;
                            let iw0 = iw0 as usize;
                            if sw == 1 {
                                out[lo..hi].copy_from_slice(&x_row[iw0..iw0 + (hi - lo)]);
                            } else {
                                for (i, v) in out[lo..hi].iter_mut().enumerate() {
                                    *v = x_row[iw0 + i * sw];
                                }
                            }
                        } else {
                            out[lo..].fill(zero);
                        }
                    }
                }
            }
        }
    });
}

fn geometry(x_shape: &[usize], w_shape: &[usize], options: &ConvOptions<3>) -> ConvGeometry {
    let in_dims = [x_shape[2], x_shape[3], x_shape[4]];
    let kernel = [w_shape[2], w_shape[3], w_shape[4]];
    let out_dims = core::array::from_fn(|i| {
        calculate_conv_output_size(
            kernel[i],
            options.stride[i],
            options.padding_begin()[i],
            options.dilation[i],
            in_dims[i],
        )
    });
    ConvGeometry {
        in_dims,
        out_dims,
        kernel,
        stride: options.stride,
        dilation: options.dilation,
        padding: options.padding_begin(),
        channels_in: x_shape[1],
        channels_out: w_shape[0],
        channels_per_group: w_shape[1],
        out_channels_per_group: w_shape[0] / options.groups,
    }
}

/// Splits a group's input channels into chunks of at most `ROWS_PER_CHUNK` im2col rows, and into
/// at least `min_chunks` chunks where that leaves each at least `MIN_ROWS_PER_CHUNK` rows.
fn channel_chunks(conv: &ConvGeometry, min_chunks: usize) -> Vec<core::ops::Range<usize>> {
    let taps: usize = conv.kernel.iter().product();
    let rows = conv.channels_per_group * taps;
    let most = (rows / MIN_ROWS_PER_CHUNK).clamp(1, conv.channels_per_group);
    let n = rows
        .div_ceil(ROWS_PER_CHUNK)
        .max(min_chunks.min(most))
        .max(1);
    let size = conv.channels_per_group.div_ceil(n);
    (0..conv.channels_per_group)
        .step_by(size)
        .map(|c| c..(c + size).min(conv.channels_per_group))
        .collect()
}

fn threads() -> usize {
    #[cfg(feature = "rayon")]
    return rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    1
}

/// Runs `f(task)` for each task, in parallel when rayon is enabled, each with its own scratch
/// from `init`.
fn for_each_task<S, I: Fn() -> S + Send + Sync, F: Fn(&mut S, usize) + Send + Sync>(
    tasks: usize,
    init: I,
    f: F,
) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        (0..tasks).into_par_iter().for_each_init(init, f);
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut scratch = init();
        for task in 0..tasks {
            f(&mut scratch, task);
        }
    }
}

/// Gradient of a 3D convolution with respect to its weight, `[c_out, c_in / groups, kd, kh, kw]`.
///
/// `x` and `options` must have symmetric padding (see `pad_asymmetric_conv_input`).
pub(crate) fn conv3d_weight_backward<T: ConvElement>(
    x: FlexTensor,
    output_grad: FlexTensor,
    weight_shape: &[usize],
    options: &ConvOptions<3>,
    dtype: DType,
) -> FlexTensor {
    let x = x.to_contiguous();
    let dy = output_grad.to_contiguous();
    let conv = geometry(x.layout().shape(), weight_shape, options);
    let x_data: &[T] = x.storage();
    let dy_data: &[T] = dy.storage();
    let zero: T = bytemuck::Zeroable::zeroed();
    let one: T = burn_backend::ElementConversion::elem(1.0f32);

    let groups = options.groups;
    let batch = x.layout().shape()[0];
    let ocpg = conv.out_channels_per_group;
    let taps: usize = conv.kernel.iter().product();
    let spatial_out = conv.spatial_out();
    let positions = batch * spatial_out;
    // Positions are the gemm's reduction dimension here: long tiles amortise reading and writing
    // the accumulator.
    let tile_for = |rows: usize| {
        (DW_COL_BLOCK_BYTES / (rows.max(1) * core::mem::size_of::<T>()))
            .clamp(256, 4096)
            .min(positions.max(1))
    };
    // Tasks come from splitting positions (each split needs its own partial sum) and channels
    // (each chunk redoes the output-gradient gather and makes the gemm narrower). Split
    // positions first; chunk channels further only when there are too few tiles to split.
    let row_chunks = channel_chunks(&conv, 1);
    let tiles_available = positions.div_ceil(tile_for(row_chunks[0].len() * taps));
    let wanted_jobs = (2 * threads()).div_ceil(tiles_available.max(1));
    let chunks = channel_chunks(&conv, wanted_jobs.div_ceil(groups));
    let max_rows = chunks.iter().map(|c| c.len()).max().unwrap_or(0) * taps;
    let tile = tile_for(max_rows);
    let num_tiles = positions.div_ceil(tile);
    // Split positions into ranges so that there are enough tasks to occupy the threads.
    let jobs = groups * chunks.len();
    let splits = (2 * threads()).div_ceil(jobs).clamp(1, num_tiles.max(1));
    let tiles_per_split = num_tiles.div_ceil(splits);
    let tasks = jobs * splits;

    // Partial sums, one `[rows, ocpg]` block per task, rows ordered (channel, tap).
    let mut partials = vec![zero; tasks * max_rows * ocpg];
    let partial_len = max_rows * ocpg;
    let partials_ptr = crate::ops::SendMutPtr::new(partials.as_mut_ptr());

    for_each_task(
        tasks,
        || (vec![zero; max_rows * tile], vec![zero; tile * ocpg]),
        |(col, dyt), task| {
            let split = task % splits;
            let job = task / splits;
            let g = job / chunks.len();
            let channels = chunks[job % chunks.len()].clone();
            let rows = channels.len() * taps;
            // SAFETY: each task owns its own partial block.
            let acc = unsafe { partials_ptr.ptr_add(task * partial_len) };
            let first_tile = split * tiles_per_split;
            let last_tile = ((split + 1) * tiles_per_split).min(num_tiles);
            for t in first_tile..last_tile {
                let start = t * tile;
                let len = tile.min(positions - start);
                // SAFETY: `col` holds `max_rows * tile` elements; this writes `rows * len`.
                unsafe {
                    im2col(
                        col.as_mut_ptr(),
                        len,
                        x_data,
                        &conv,
                        g,
                        channels.clone(),
                        start,
                        start + len,
                    )
                };
                gather_transposed(dyt, dy_data, &conv, g, start, start + len);
                unsafe {
                    gemm::gemm(
                        rows,
                        ocpg,
                        len,
                        acc,
                        1,
                        ocpg as isize,
                        true,
                        col.as_ptr(),
                        1,
                        len as isize,
                        dyt.as_ptr(),
                        1,
                        ocpg as isize,
                        one,
                        one,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
            }
        },
    );

    // Sum the partials and transpose (channel, tap) x m into the weight layout, one output
    // channel per task.
    let col_len = conv.channels_per_group * taps;
    let mut dw = vec![zero; conv.channels_out * col_len];
    let dw_ptr = crate::ops::SendMutPtr::new(dw.as_mut_ptr());
    for_each_task(
        conv.channels_out,
        || (),
        |_, out_channel| {
            let g = out_channel / ocpg;
            let m = out_channel % ocpg;
            // SAFETY: each task writes only its own output channel's row.
            let dw_row = unsafe {
                core::slice::from_raw_parts_mut(dw_ptr.ptr_add(out_channel * col_len), col_len)
            };
            for (ci, channels) in chunks.iter().enumerate() {
                let job = g * chunks.len() + ci;
                let rows = channels.len() * taps;
                let dst = &mut dw_row[channels.start * taps..][..rows];
                for split in 0..splits {
                    let partial = &partials[(job * splits + split) * partial_len..][..rows * ocpg];
                    for (r, o) in dst.iter_mut().enumerate() {
                        *o = o.add(partial[r * ocpg + m]);
                    }
                }
            }
        },
    );

    FlexTensor::new(
        Bytes::from_elems(dw),
        Layout::contiguous(Shape::from(weight_shape.to_vec())),
        dtype,
    )
}

/// Copy the output gradient of group `g` at positions `start..end` (numbered across the batch)
/// into `dyt` as `[position, ocpg]`, the layout the weight-gradient gemm reads unit-strided.
fn gather_transposed<T: Copy>(
    dyt: &mut [T],
    dy: &[T],
    conv: &ConvGeometry,
    g: usize,
    start: usize,
    end: usize,
) {
    let spatial_out = conv.spatial_out();
    let ocpg = conv.out_channels_per_group;
    let mut q = start;
    while q < end {
        let b = q / spatial_out;
        let p = q % spatial_out;
        let len = (spatial_out - p).min(end - q);
        for m in 0..ocpg {
            let src = &dy[(b * conv.channels_out + g * ocpg + m) * spatial_out + p..][..len];
            for (i, &v) in src.iter().enumerate() {
                dyt[(q - start + i) * ocpg + m] = v;
            }
        }
        q += len;
    }
}

/// Copy the output gradient of group `g` at positions `start..end` (numbered across the batch)
/// into `dyb` as `[ocpg, position]`.
fn gather<T: Copy>(
    dyb: &mut [T],
    dy: &[T],
    conv: &ConvGeometry,
    g: usize,
    start: usize,
    end: usize,
) {
    let spatial_out = conv.spatial_out();
    let ocpg = conv.out_channels_per_group;
    let n = end - start;
    let mut q = start;
    while q < end {
        let b = q / spatial_out;
        let p = q % spatial_out;
        let len = (spatial_out - p).min(end - q);
        for m in 0..ocpg {
            let src = &dy[(b * conv.channels_out + g * ocpg + m) * spatial_out + p..][..len];
            dyb[m * n + (q - start)..][..len].copy_from_slice(src);
        }
        q += len;
    }
}

/// Gradient of a 3D convolution with respect to its input, of shape `x_shape`.
///
/// `options` must have symmetric padding; for asymmetric padding, pass the shape of the padded
/// input and crop the result.
pub(crate) fn conv3d_x_backward<T: ConvElement>(
    x_shape: &[usize],
    weight: FlexTensor,
    output_grad: FlexTensor,
    options: &ConvOptions<3>,
    dtype: DType,
) -> FlexTensor {
    let weight = weight.to_contiguous();
    let dy = output_grad.to_contiguous();
    let conv = geometry(x_shape, weight.layout().shape(), options);
    let w_data: &[T] = weight.storage();
    let dy_data: &[T] = dy.storage();
    let zero: T = bytemuck::Zeroable::zeroed();
    let one: T = burn_backend::ElementConversion::elem(1.0f32);

    let groups = options.groups;
    let batch = x_shape[0];
    let ocpg = conv.out_channels_per_group;
    let taps: usize = conv.kernel.iter().product();
    let col_len = conv.channels_per_group * taps;
    let spatial_out = conv.spatial_out();
    let chunks = channel_chunks(&conv, 1);
    let max_rows = chunks.iter().map(|c| c.len()).max().unwrap_or(0) * taps;

    // w^T per group, `[col_len, ocpg]`, so the gemm reads it unit-strided. Transposed in
    // parallel blocks of rows: it is as large as the weights, 9 MiB for a 512->512 3x3 conv.
    const TRANSPOSE_ROWS: usize = 64;
    let mut wt = vec![zero; groups * col_len * ocpg];
    let wt_ptr = crate::ops::SendMutPtr::new(wt.as_mut_ptr());
    let row_blocks = col_len.div_ceil(TRANSPOSE_ROWS);
    for_each_task(
        groups * row_blocks,
        || (),
        |_, task| {
            let g = task / row_blocks;
            let r0 = (task % row_blocks) * TRANSPOSE_ROWS;
            let r1 = (r0 + TRANSPOSE_ROWS).min(col_len);
            // SAFETY: each task writes only rows `r0..r1` of group `g`.
            let dst = unsafe {
                core::slice::from_raw_parts_mut(
                    wt_ptr.ptr_add((g * col_len + r0) * ocpg),
                    (r1 - r0) * ocpg,
                )
            };
            for m in 0..ocpg {
                let src = &w_data[(g * ocpg + m) * col_len + r0..][..r1 - r0];
                for (i, &v) in src.iter().enumerate() {
                    dst[i * ocpg + m] = v;
                }
            }
        },
    );

    // Every gemm packs its chunk of w^T again: wide tiles amortise that.
    let tile =
        (DX_COL_BLOCK_BYTES / (max_rows.max(1) * core::mem::size_of::<T>())).clamp(128, 2048);
    // Each task takes whole images, so that no two tasks add into the same input element, and
    // enough of them that its gemms have a wide N even when images have few output positions.
    let jobs = groups * chunks.len();
    let images_per_task = DX_MIN_POSITIONS_PER_TASK
        .div_ceil(spatial_out.max(1))
        .min((batch * jobs / (2 * threads())).max(1))
        .clamp(1, batch.max(1));
    let image_blocks = batch.div_ceil(images_per_task);
    let tasks = image_blocks * jobs;

    let mut dx = vec![zero; x_shape.iter().product()];
    let dx_ptr = crate::ops::SendMutPtr::new(dx.as_mut_ptr());

    for_each_task(
        tasks,
        || (vec![zero; max_rows * tile], vec![zero; ocpg * tile]),
        |(dcol, dyb), task| {
            let block = task / jobs;
            let job = task % jobs;
            let g = job / chunks.len();
            let channels = chunks[job % chunks.len()].clone();
            let rows = channels.len() * taps;
            let wt_chunk = &wt[(g * col_len + channels.start * taps) * ocpg..][..rows * ocpg];
            let first = block * images_per_task * spatial_out;
            let last = ((block + 1) * images_per_task).min(batch) * spatial_out;
            for start in (first..last).step_by(tile) {
                let len = tile.min(last - start);
                gather(dyb, dy_data, &conv, g, start, start + len);
                unsafe {
                    gemm::gemm(
                        rows,
                        len,
                        ocpg,
                        dcol.as_mut_ptr(),
                        1,
                        len as isize,
                        false,
                        wt_chunk.as_ptr(),
                        1,
                        ocpg as isize,
                        dyb.as_ptr(),
                        1,
                        len as isize,
                        zero,
                        one,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
                // SAFETY: this task is the only one writing these images' channels `channels`
                // of group `g`.
                unsafe {
                    col2im(
                        &dx_ptr,
                        &dcol[..rows * len],
                        len,
                        g,
                        channels.clone(),
                        &conv,
                        start,
                        start + len,
                    )
                };
            }
        },
    );

    FlexTensor::new(
        Bytes::from_elems(dx),
        Layout::contiguous(Shape::from(x_shape.to_vec())),
        dtype,
    )
}

/// Add `dcol` (`[channels * taps, end - start]`, rows `row_stride` apart, for output positions
/// `start..end` numbered across the batch) back onto the input gradient `dx`, channels
/// `channels` of group `g`: the transpose of `im2col`. Taps that fall in the padding are
/// dropped.
///
/// # Safety
/// `dx` must point to the whole input gradient, and no other thread may access the elements of
/// these images and channels.
#[allow(clippy::too_many_arguments)]
unsafe fn col2im<T: ConvElement>(
    dx: &crate::ops::SendMutPtr<T>,
    dcol: &[T],
    row_stride: usize,
    g: usize,
    channels: core::ops::Range<usize>,
    conv: &ConvGeometry,
    start: usize,
    end: usize,
) {
    let [in_d, in_h, in_w] = conv.in_dims;
    let [_, out_h, out_w] = conv.out_dims;
    let [kd_n, kh_n, kw_n] = conv.kernel;
    let spatial_in = conv.spatial_in();
    let sw = conv.stride[2];

    conv.for_each_row_run(start, end, |b, p, offset, len| {
        let od = p / (out_h * out_w);
        let oh = (p / out_w) % out_h;
        let ow0 = p % out_w;
        let first_channel = b * conv.channels_in + g * conv.channels_per_group + channels.start;
        // SAFETY: in bounds of the input gradient; the caller guarantees exclusive access.
        let dx = unsafe {
            core::slice::from_raw_parts_mut(
                dx.ptr_add(first_channel * spatial_in),
                channels.len() * spatial_in,
            )
        };
        let mut row = 0;
        for c in 0..channels.len() {
            let dx_c = &mut dx[c * spatial_in..][..spatial_in];
            for kd in 0..kd_n {
                let id = (od * conv.stride[0] + kd * conv.dilation[0]) as isize
                    - conv.padding[0] as isize;
                for kh in 0..kh_n {
                    let ih = (oh * conv.stride[1] + kh * conv.dilation[1]) as isize
                        - conv.padding[1] as isize;
                    let row_valid = id >= 0 && id < in_d as isize && ih >= 0 && ih < in_h as isize;
                    for kw in 0..kw_n {
                        let src = &dcol[row * row_stride + offset..][..len];
                        row += 1;
                        if !row_valid {
                            continue;
                        }
                        let dx_row = &mut dx_c[(id as usize * in_h + ih as usize) * in_w..][..in_w];
                        let w_off = (kw * conv.dilation[2]) as isize - conv.padding[2] as isize;
                        let first = if w_off >= 0 {
                            0
                        } else {
                            ((-w_off) as usize).div_ceil(sw)
                        };
                        let last = if (in_w as isize) <= w_off {
                            0
                        } else {
                            ((in_w as isize - 1 - w_off) as usize) / sw + 1
                        };
                        let lo = first.clamp(ow0, ow0 + len) - ow0;
                        let hi = last.clamp(ow0, ow0 + len) - ow0;
                        if hi <= lo {
                            continue;
                        }
                        let iw0 = (((ow0 + lo) * sw) as isize + w_off) as usize;
                        if sw == 1 {
                            for (d, &v) in dx_row[iw0..iw0 + (hi - lo)].iter_mut().zip(&src[lo..hi])
                            {
                                *d = d.add(v);
                            }
                        } else {
                            for (i, &v) in src[lo..hi].iter().enumerate() {
                                let d = &mut dx_row[iw0 + i * sw];
                                *d = d.add(v);
                            }
                        }
                    }
                }
            }
        }
    });
}

/// Lift `N`-dimensional convolution options and shapes to 3D, with size-1 leading spatial dims.
fn to_3d<const N: usize>(options: &ConvOptions<N>) -> ConvOptions<3> {
    let lift = |v: &[usize], fill: usize| -> [usize; 3] {
        core::array::from_fn(|i| if i < 3 - N { fill } else { v[i - (3 - N)] })
    };
    ConvOptions::new(
        lift(&options.stride, 1),
        lift(&options.padding_begin(), 0),
        lift(&options.dilation, 1),
        options.groups,
    )
}

fn shape_to_3d(shape: &[usize]) -> Vec<usize> {
    let mut out = shape[..2].to_vec();
    out.extend(core::iter::repeat_n(1, 5 - shape.len()));
    out.extend_from_slice(&shape[2..]);
    out
}

fn reshape(tensor: FlexTensor, shape: &[usize]) -> FlexTensor {
    tensor.reshape(Shape::from(shape.to_vec()))
}

/// Run `f` in the tensors' element type, computing bf16 in f32.
fn dispatch(
    dtype: DType,
    tensors: Vec<FlexTensor>,
    f: impl FnOnce(Vec<FlexTensor>, DType) -> FlexTensor,
) -> FlexTensor {
    use super::module::{cast_from_f32, cast_to_f32};
    use burn_std::bf16;
    match dtype {
        DType::BF16 => {
            let tensors = tensors
                .into_iter()
                .map(|t| cast_to_f32::<bf16>(t, bf16::to_f32))
                .collect();
            cast_from_f32::<bf16>(f(tensors, DType::F32), bf16::from_f32)
        }
        _ => f(tensors, dtype),
    }
}

/// Input gradient of an `N`-dimensional convolution.
pub(crate) fn conv_x_backward<const N: usize>(
    x: FlexTensor,
    weight: FlexTensor,
    output_grad: FlexTensor,
    options: ConvOptions<N>,
) -> FlexTensor {
    let x_shape = x.layout().shape().to_vec();
    // Asymmetric padding: take the gradient of the explicitly padded input, then crop.
    let mut padded_shape = x_shape.clone();
    for (i, (begin, end)) in options.padding.iter().enumerate() {
        if options.is_asymmetric() {
            padded_shape[i + 2] += begin + end;
        }
    }
    let options_3d = if options.is_asymmetric() {
        to_3d(&ConvOptions::new(
            options.stride,
            [0; N],
            options.dilation,
            options.groups,
        ))
    } else {
        to_3d(&options)
    };
    let w_shape = weight.layout().shape().to_vec();
    let dy_shape = output_grad.layout().shape().to_vec();
    let x3 = shape_to_3d(&padded_shape);
    let dtype = x.dtype();
    let grad = dispatch(
        dtype,
        vec![
            reshape(weight, &shape_to_3d(&w_shape)),
            reshape(output_grad, &shape_to_3d(&dy_shape)),
        ],
        |mut t, dtype| {
            let dy = t.pop().unwrap();
            let w = t.pop().unwrap();
            match dtype {
                DType::F32 => conv3d_x_backward::<f32>(&x3, w, dy, &options_3d, dtype),
                DType::F64 => conv3d_x_backward::<f64>(&x3, w, dy, &options_3d, dtype),
                DType::F16 => conv3d_x_backward::<burn_std::f16>(&x3, w, dy, &options_3d, dtype),
                dtype => panic!("conv backward: unsupported dtype {dtype:?}"),
            }
        },
    );
    let grad = reshape(grad, &padded_shape);
    if !options.is_asymmetric() {
        return grad;
    }
    let mut slices: Vec<burn_std::Slice> = x_shape
        .iter()
        .map(|&s| burn_std::Slice::from(0..s))
        .collect();
    for (i, (begin, _)) in options.padding.iter().enumerate() {
        slices[i + 2] = burn_std::Slice::from(*begin..*begin + x_shape[i + 2]);
    }
    <crate::Flex as burn_backend::ops::FloatTensorOps<crate::Flex>>::float_slice(grad, &slices)
}

/// Weight gradient of an `N`-dimensional convolution.
pub(crate) fn conv_weight_backward<const N: usize>(
    x: FlexTensor,
    weight: FlexTensor,
    output_grad: FlexTensor,
    options: ConvOptions<N>,
) -> FlexTensor {
    let (x, options) =
        burn_backend::ops::conv::pad_asymmetric_conv_input::<crate::Flex, N>(x, options);
    let options_3d = to_3d(&options);
    let w_shape = weight.layout().shape().to_vec();
    let w3 = shape_to_3d(&w_shape);
    let x_shape = x.layout().shape().to_vec();
    let dy_shape = output_grad.layout().shape().to_vec();
    let dtype = x.dtype();
    let grad = dispatch(
        dtype,
        vec![
            reshape(x, &shape_to_3d(&x_shape)),
            reshape(output_grad, &shape_to_3d(&dy_shape)),
        ],
        |mut t, dtype| {
            let dy = t.pop().unwrap();
            let x = t.pop().unwrap();
            match dtype {
                DType::F32 => conv3d_weight_backward::<f32>(x, dy, &w3, &options_3d, dtype),
                DType::F64 => conv3d_weight_backward::<f64>(x, dy, &w3, &options_3d, dtype),
                DType::F16 => {
                    conv3d_weight_backward::<burn_std::f16>(x, dy, &w3, &options_3d, dtype)
                }
                dtype => panic!("conv backward: unsupported dtype {dtype:?}"),
            }
        },
    );
    reshape(grad, &w_shape)
}
