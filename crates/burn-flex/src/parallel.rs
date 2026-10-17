//! Splitting elementwise work across threads.
//!
//! A memory-bound loop gains from more threads as long as one core cannot draw the machine's
//! memory bandwidth on its own. On Apple silicon one core comes close, and Flex's elementwise ops
//! ran single-threaded; on an x86 laptop (Core Ultra 7 155H) they ran 5-8x slower than
//! libtorch's threaded loops, ReLU over 32 MiB at about 4 GB/s. Work above `MIN_ELEMS` elements
//! is therefore split into chunks of `CHUNK` elements.
//!
//! Without the `rayon` feature every helper runs on the calling thread.

use core::mem::MaybeUninit;

use alloc::vec::Vec;

/// Fewest elements worth splitting: below this, handing chunks to rayon costs more than it saves.
pub(crate) const MIN_ELEMS: usize = 1 << 15;
/// Elements per task: enough to amortise the task, few enough to balance across cores.
pub(crate) const CHUNK: usize = 1 << 14;

/// Whether `n` elements of work should be split across threads.
#[inline]
pub(crate) fn worth_splitting(n: usize) -> bool {
    cfg!(feature = "rayon") && n >= MIN_ELEMS
}

/// Apply `f` to every element of `data`, in place.
pub(crate) fn for_each_mut<T: Send>(data: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
    #[cfg(feature = "rayon")]
    if worth_splitting(data.len()) {
        use rayon::prelude::*;
        data.par_chunks_mut(CHUNK)
            .for_each(|chunk| chunk.iter_mut().for_each(&f));
        return;
    }
    data.iter_mut().for_each(f);
}

/// `f` applied to each element of `src`, in order.
pub(crate) fn map<S: Sync, D: Send>(src: &[S], f: impl Fn(&S) -> D + Sync + Send) -> Vec<D> {
    fill(src.len(), |start, out| {
        for (o, s) in out.iter_mut().zip(&src[start..]) {
            o.write(f(s));
        }
    })
}

/// A vector of `n` elements, written chunk by chunk by `write(start, chunk)`, which must
/// initialise every element of `chunk` (the elements `start..start + chunk.len()`).
pub(crate) fn fill<D: Send>(
    n: usize,
    write: impl Fn(usize, &mut [MaybeUninit<D>]) + Sync + Send,
) -> Vec<D> {
    let mut out = Vec::with_capacity(n);
    let spare = &mut out.spare_capacity_mut()[..n];
    #[cfg(feature = "rayon")]
    if worth_splitting(n) {
        use rayon::prelude::*;
        spare
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| write(i * CHUNK, chunk));
        // SAFETY: `write` initialised every element of every chunk, which together are `..n`.
        unsafe { out.set_len(n) };
        return out;
    }
    write(0, spare);
    // SAFETY: `write` initialised all of `..n`.
    unsafe { out.set_len(n) };
    out
}

/// Runs `f(first, count)` over `0..n` split into ranges of about `per_task` items, in parallel
/// when there are enough items to be worth it. Without rayon, `f` runs once over all of them.
#[cfg_attr(not(feature = "rayon"), allow(unused_variables))]
pub(crate) fn for_ranges(n: usize, per_task: usize, f: impl Fn(usize, usize) + Sync + Send) {
    let per_task = per_task.max(1);
    #[cfg(feature = "rayon")]
    if n > per_task {
        use rayon::prelude::*;
        (0..n.div_ceil(per_task)).into_par_iter().for_each(|t| {
            let first = t * per_task;
            f(first, per_task.min(n - first));
        });
        return;
    }
    f(0, n);
}

/// Runs `f(i)` for every `i` in `0..n`, in parallel when rayon is enabled.
pub(crate) fn for_each_index(n: usize, f: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "rayon"))]
    (0..n).for_each(f);
}
