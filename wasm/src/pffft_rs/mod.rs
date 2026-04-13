mod vector4;

use std::f32::consts::PI;

use vector4::{
    v4_add, v4_cplx_mul, v4_cplx_mul_conj, v4_interleave2, v4_load, v4_mul, v4_scalar_mul, v4_splat, v4_store, v4_sub,
    v4_swaphl, v4_transpose, v4_uninterleave2,
};

// We use 4-wide vectors
const SIMD_SZ: usize = 4;
const IFAC_MAX_SIZE: usize = 25;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PffftDirection {
    Forward,
    Backward,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PffftTransform {
    Real,
    Complex,
}

// Main setup structure
pub struct PFFFTSetup {
    n: usize,
    ncvec: usize,
    ifac: Vec<usize>,
    transform: PffftTransform,
    data: Vec<f32>,
    e_offset: usize,
    twiddle_offset: usize,
}

impl PFFFTSetup {
    /// Create a new PFFFT setup for transforms of size N. Returns None if N is invalid or allocation fails.
    pub fn new(n: usize, transform: PffftTransform) -> Option<Self> {
        if n == 0 {
            return None;
        }

        if n > (1 << 26) {
            // Would cause integer overflow
            return None;
        }

        // Check that N is a multiple of required SIMD size
        match transform {
            PffftTransform::Real => {
                if n % (2 * SIMD_SZ * SIMD_SZ) != 0 {
                    return None;
                }
            }
            PffftTransform::Complex => {
                if n % (SIMD_SZ * SIMD_SZ) != 0 {
                    return None;
                }
            }
        }

        // Compute Ncvec: number of complex SIMD vectors
        let ncvec = match transform {
            PffftTransform::Real => (n / 2) / SIMD_SZ,
            PffftTransform::Complex => n / SIMD_SZ,
        };

        // Allocate data buffer: 2 * Ncvec * SIMD_SZ floats
        let data_len = 2 * ncvec * SIMD_SZ;
        let mut data = vec![0.0; data_len];

        // Compute offsets for e and twiddle pointers (as in C code)
        // e points to the beginning of data
        let e_offset = 0;

        // twiddle points to: data + (2 * Ncvec * (SIMD_SZ - 1)) / SIMD_SZ * SIMD_SZ
        // In float indices: (2 * ncvec * (SIMD_SZ - 1)) / SIMD_SZ * SIMD_SZ
        let v4sf_offset = (2 * ncvec * (SIMD_SZ - 1)) / SIMD_SZ;
        let twiddle_offset = v4sf_offset * SIMD_SZ;

        // Initialize ifac vector with zeros
        let mut ifac = vec![0; IFAC_MAX_SIZE];

        // Compute e array (twiddle factors)
        let e_slice = data.as_mut_slice();
        for k in 0..ncvec {
            let i = k / SIMD_SZ;
            let j = k % SIMD_SZ;
            for m in 0..(SIMD_SZ - 1) {
                let a = -2.0 * PI * ((m + 1) as f32) * (k as f32) / (n as f32);
                let idx_cos = (2 * (i * 3 + m) + 0) * SIMD_SZ + j;
                let idx_sin = (2 * (i * 3 + m) + 1) * SIMD_SZ + j;
                if idx_cos < data_len && idx_sin < data_len {
                    e_slice[idx_cos] = a.cos();
                    e_slice[idx_sin] = a.sin();
                }
            }
        }

        // Compute twiddle factors using functions from pass module
        let twiddle_slice = &mut data.as_mut_slice()[twiddle_offset..];

        // Call appropriate initialization function
        let n_over_simd = n / SIMD_SZ;
        match transform {
            PffftTransform::Real => {
                // rffti1_ps expects n_over_simd, twiddle pointer, ifac pointer
                unsafe {
                    rffti1_ps(n_over_simd, twiddle_slice.as_mut_ptr(), ifac.as_mut_ptr() as *mut usize);
                }
            }
            PffftTransform::Complex => {
                // cffti1_ps expects n_over_simd, twiddle pointer, ifac pointer
                unsafe {
                    cffti1_ps(n_over_simd, twiddle_slice.as_mut_ptr(), ifac.as_mut_ptr() as *mut usize);
                }
            }
        }

        // Check that N is decomposable with allowed prime factors
        let mut m = 1;
        let n_factors = ifac[1];
        for k in 0..n_factors {
            m *= ifac[2 + k];
        }
        if m != n_over_simd {
            return None;
        }

        Some(Self { n, ncvec, ifac, transform, data, e_offset, twiddle_offset })
    }

    /// Get the size N of the transform
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the transform type
    pub fn transform(&self) -> PffftTransform {
        self.transform
    }

    /// Reorder FFT output/input layout to match PFFFT packed order.
    pub fn zreorder(&self, input: &[f32], output: &mut [f32], direction: PffftDirection) {
        let required_len = match self.transform {
            PffftTransform::Real => self.n,
            PffftTransform::Complex => self.n * 2,
        };
        assert!(input.len() >= required_len, "input slice must have at least required_len elements");
        assert!(output.len() >= required_len, "output slice must have at least required_len elements");

        unsafe {
            pffft_zreorder(self, input.as_ptr(), output.as_mut_ptr(), direction);
        }
    }

    /// Transform using packed (internal) PFFFT layout.
    pub fn transform_packed(&self, input: &[f32], output: &mut [f32], work: Option<&mut [f32]>, direction: PffftDirection) {
        let required_len = match self.transform {
            PffftTransform::Real => self.n,
            PffftTransform::Complex => self.n * 2,
        };
        assert!(input.len() >= required_len, "input slice must have at least required_len elements");
        assert!(output.len() >= required_len, "output slice must have at least required_len elements");

        let work_ptr = match work {
            Some(w) => {
                assert!(w.len() >= required_len, "work slice must have at least required_len elements");
                w.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        };

        unsafe {
            pffft_transform_internal(self, input.as_ptr(), output.as_mut_ptr(), work_ptr, direction, false);
        }
    }

    /// Transform using ordered (interleaved complex) layout, equivalent to C pffft_transform_ordered.
    pub fn transform_ordered(
        &self,
        input: &[f32],
        output: &mut [f32],
        work: Option<&mut [f32]>,
        direction: PffftDirection,
    ) {
        let required_len = match self.transform {
            PffftTransform::Real => self.n,
            PffftTransform::Complex => self.n * 2,
        };
        assert!(input.len() >= required_len, "input slice must have at least required_len elements");
        assert!(output.len() >= required_len, "output slice must have at least required_len elements");

        let work_ptr = match work {
            Some(w) => {
                assert!(w.len() >= required_len, "work slice must have at least required_len elements");
                w.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        };

        unsafe {
            pffft_transform_internal(self, input.as_ptr(), output.as_mut_ptr(), work_ptr, direction, true);
        }
    }

    /// Frequency-domain complex multiply-accumulate in packed layout.
    pub fn zconvolve_accumulate(&self, a: &[f32], b: &[f32], ab: &mut [f32], scaling: f32) {
        let required_len = match self.transform {
            PffftTransform::Real => self.n,
            PffftTransform::Complex => self.n * 2,
        };
        assert!(a.len() >= required_len, "a slice must have at least required_len elements");
        assert!(b.len() >= required_len, "b slice must have at least required_len elements");
        assert!(ab.len() >= required_len, "ab slice must have at least required_len elements");

        unsafe {
            pffft_zconvolve_accumulate(self, a.as_ptr(), b.as_ptr(), ab.as_mut_ptr(), scaling);
        }
    }
}

#[inline(never)]
unsafe fn passf2_ps(ido: usize, l1: usize, mut cc: *const f32, mut ch: *mut f32, wa1: *const f32, fsign: f32) {
    let l1ido = l1 * ido;

    unsafe {
        if ido <= 2 {
            for _ in (0..l1ido).step_by(ido) {
                let cc0 = v4_load(cc, 0);
                let cc_ido0 = v4_load(cc, ido);
                let cc1 = v4_load(cc, 1);
                let cc_ido1 = v4_load(cc, ido + 1);

                v4_store(ch, 0, v4_add(cc0, cc_ido0));
                v4_store(ch, l1ido, v4_sub(cc0, cc_ido0));
                v4_store(ch, 1, v4_add(cc1, cc_ido1));
                v4_store(ch, l1ido + 1, v4_sub(cc1, cc_ido1));

                ch = ch.add(ido * 4);
                cc = cc.add(2 * ido * 4);
            }
        } else {
            for _ in (0..l1ido).step_by(ido) {
                for i in (0..(ido - 1)).step_by(2) {
                    let cc_i0 = v4_load(cc, i);
                    let cc_i_ido0 = v4_load(cc, i + ido);
                    let cc_i1 = v4_load(cc, i + 1);
                    let cc_i_ido1 = v4_load(cc, i + ido + 1);

                    let tr2 = v4_sub(cc_i0, cc_i_ido0);
                    let ti2 = v4_sub(cc_i1, cc_i_ido1);

                    let wr = v4_splat(*wa1.add(i));
                    let wi = v4_mul(v4_splat(fsign), v4_splat(*wa1.add(i + 1)));

                    v4_store(ch, i, v4_add(cc_i0, cc_i_ido0));
                    v4_store(ch, i + 1, v4_add(cc_i1, cc_i_ido1));

                    let (tr2, ti2) = v4_cplx_mul(tr2, ti2, wr, wi);
                    v4_store(ch, i + l1ido, tr2);
                    v4_store(ch, i + l1ido + 1, ti2);
                }
                ch = ch.add(ido * 4);
                cc = cc.add(2 * ido * 4);
            }
        }
    }
}

#[inline(never)]
unsafe fn passf3_ps(
    ido: usize,
    l1: usize,
    mut cc: *const f32,
    mut ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    fsign: f32,
) {
    let taur: f32 = -0.5;
    let taui: f32 = 0.866025403784439 * fsign;
    let l1ido = l1 * ido;

    debug_assert!(ido > 2);

    unsafe {
        for _ in (0..l1ido).step_by(ido) {
            for i in (0..(ido - 1)).step_by(2) {
                let cc_i = v4_load(cc, i);
                let cc_i1 = v4_load(cc, i + 1);
                let cc_i_ido = v4_load(cc, i + ido);
                let cc_i_ido1 = v4_load(cc, i + ido + 1);
                let cc_i_2ido = v4_load(cc, i + 2 * ido);
                let cc_i_2ido1 = v4_load(cc, i + 2 * ido + 1);

                let tr2 = v4_add(cc_i_ido, cc_i_2ido);
                let cr2 = v4_add(cc_i, v4_scalar_mul(taur, tr2));
                v4_store(ch, i, v4_add(cc_i, tr2));

                let ti2 = v4_add(cc_i_ido1, cc_i_2ido1);
                let ci2 = v4_add(cc_i1, v4_scalar_mul(taur, ti2));
                v4_store(ch, i + 1, v4_add(cc_i1, ti2));

                let cr3 = v4_scalar_mul(taui, v4_sub(cc_i_ido, cc_i_2ido));
                let ci3 = v4_scalar_mul(taui, v4_sub(cc_i_ido1, cc_i_2ido1));

                let dr2 = v4_sub(cr2, ci3);
                let dr3 = v4_add(cr2, ci3);
                let di2 = v4_add(ci2, cr3);
                let di3 = v4_sub(ci2, cr3);

                let wr1 = *wa1.add(i);
                let wi1 = fsign * *wa1.add(i + 1);
                let wr2 = *wa2.add(i);
                let wi2 = fsign * *wa2.add(i + 1);

                let (dr2, di2) = v4_cplx_mul(dr2, di2, v4_splat(wr1), v4_splat(wi1));
                v4_store(ch, i + l1ido, dr2);
                v4_store(ch, i + l1ido + 1, di2);

                let (dr3, di3) = v4_cplx_mul(dr3, di3, v4_splat(wr2), v4_splat(wi2));
                v4_store(ch, i + 2 * l1ido, dr3);
                v4_store(ch, i + 2 * l1ido + 1, di3);
            }

            cc = cc.add(3 * ido * 4);
            ch = ch.add(ido * 4);
        }
    }
}

#[inline(never)]
unsafe fn passf4_ps(
    ido: usize,
    l1: usize,
    mut cc: *const f32,
    mut ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    wa3: *const f32,
    fsign: f32,
) {
    let l1ido = l1 * ido;

    unsafe {
        if ido == 2 {
            for _ in (0..l1ido).step_by(ido) {
                let cc0 = v4_load(cc, 0);
                let cc1 = v4_load(cc, 1);
                let cc_ido = v4_load(cc, ido);
                let cc_ido1 = v4_load(cc, ido + 1);
                let cc_2ido = v4_load(cc, 2 * ido);
                let cc_2ido1 = v4_load(cc, 2 * ido + 1);
                let cc_3ido = v4_load(cc, 3 * ido);
                let cc_3ido1 = v4_load(cc, 3 * ido + 1);

                let tr1 = v4_sub(cc0, cc_2ido);
                let tr2 = v4_add(cc0, cc_2ido);
                let ti1 = v4_sub(cc1, cc_2ido1);
                let ti2 = v4_add(cc1, cc_2ido1);
                let ti4 = v4_scalar_mul(fsign, v4_sub(cc_ido, cc_3ido));
                let tr4 = v4_scalar_mul(fsign, v4_sub(cc_3ido1, cc_ido1));
                let tr3 = v4_add(cc_ido, cc_3ido);
                let ti3 = v4_add(cc_ido1, cc_3ido1);

                v4_store(ch, 0, v4_add(tr2, tr3));
                v4_store(ch, 1, v4_add(ti2, ti3));
                v4_store(ch, l1ido, v4_add(tr1, tr4));
                v4_store(ch, l1ido + 1, v4_add(ti1, ti4));
                v4_store(ch, 2 * l1ido, v4_sub(tr2, tr3));
                v4_store(ch, 2 * l1ido + 1, v4_sub(ti2, ti3));
                v4_store(ch, 3 * l1ido, v4_sub(tr1, tr4));
                v4_store(ch, 3 * l1ido + 1, v4_sub(ti1, ti4));

                ch = ch.add(ido * 4);
                cc = cc.add(4 * ido * 4);
            }
        } else {
            for _ in (0..l1ido).step_by(ido) {
                for i in (0..(ido - 1)).step_by(2) {
                    let cc_i = v4_load(cc, i);
                    let cc_i1 = v4_load(cc, i + 1);
                    let cc_i_ido = v4_load(cc, i + ido);
                    let cc_i_ido1 = v4_load(cc, i + ido + 1);
                    let cc_i_2ido = v4_load(cc, i + 2 * ido);
                    let cc_i_2ido1 = v4_load(cc, i + 2 * ido + 1);
                    let cc_i_3ido = v4_load(cc, i + 3 * ido);
                    let cc_i_3ido1 = v4_load(cc, i + 3 * ido + 1);

                    let tr1 = v4_sub(cc_i, cc_i_2ido);
                    let tr2 = v4_add(cc_i, cc_i_2ido);
                    let ti1 = v4_sub(cc_i1, cc_i_2ido1);
                    let ti2 = v4_add(cc_i1, cc_i_2ido1);
                    let tr4 = v4_scalar_mul(fsign, v4_sub(cc_i_3ido1, cc_i_ido1));
                    let ti4 = v4_scalar_mul(fsign, v4_sub(cc_i_ido, cc_i_3ido));
                    let tr3 = v4_add(cc_i_ido, cc_i_3ido);
                    let ti3 = v4_add(cc_i_ido1, cc_i_3ido1);

                    v4_store(ch, i, v4_add(tr2, tr3));
                    let cr3 = v4_sub(tr2, tr3);
                    v4_store(ch, i + 1, v4_add(ti2, ti3));
                    let ci3 = v4_sub(ti2, ti3);

                    let cr2 = v4_add(tr1, tr4);
                    let cr4 = v4_sub(tr1, tr4);
                    let ci2 = v4_add(ti1, ti4);
                    let ci4 = v4_sub(ti1, ti4);

                    let wr1 = *wa1.add(i);
                    let wi1 = fsign * *wa1.add(i + 1);
                    let wr2 = *wa2.add(i);
                    let wi2 = fsign * *wa2.add(i + 1);
                    let wr3 = *wa3.add(i);
                    let wi3 = fsign * *wa3.add(i + 1);

                    let (cr2, ci2) = v4_cplx_mul(cr2, ci2, v4_splat(wr1), v4_splat(wi1));
                    v4_store(ch, i + l1ido, cr2);
                    v4_store(ch, i + l1ido + 1, ci2);

                    let (cr3, ci3) = v4_cplx_mul(cr3, ci3, v4_splat(wr2), v4_splat(wi2));
                    v4_store(ch, i + 2 * l1ido, cr3);
                    v4_store(ch, i + 2 * l1ido + 1, ci3);

                    let (cr4, ci4) = v4_cplx_mul(cr4, ci4, v4_splat(wr3), v4_splat(wi3));
                    v4_store(ch, i + 3 * l1ido, cr4);
                    v4_store(ch, i + 3 * l1ido + 1, ci4);
                }

                ch = ch.add(ido * 4);
                cc = cc.add(4 * ido * 4);
            }
        }
    }
}

#[inline(never)]
unsafe fn passf5_ps(
    ido: usize,
    l1: usize,
    mut cc: *const f32,
    mut ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    wa3: *const f32,
    wa4: *const f32,
    fsign: f32,
) {
    let tr11: f32 = 0.309016994374947;
    let ti11: f32 = 0.951056516295154 * fsign;
    let tr12: f32 = -0.809016994374947;
    let ti12: f32 = 0.587785252292473 * fsign;
    let l1ido = l1 * ido;

    debug_assert!(ido > 2);

    unsafe {
        for _ in (0..l1ido).step_by(ido) {
            for i in (0..(ido - 1)).step_by(2) {
                let cc_i = v4_load(cc, i);
                let cc_i1 = v4_load(cc, i + 1);
                let cc_i_ido = v4_load(cc, i + ido);
                let cc_i_ido1 = v4_load(cc, i + ido + 1);
                let cc_i_2ido = v4_load(cc, i + 2 * ido);
                let cc_i_2ido1 = v4_load(cc, i + 2 * ido + 1);
                let cc_i_3ido = v4_load(cc, i + 3 * ido);
                let cc_i_3ido1 = v4_load(cc, i + 3 * ido + 1);
                let cc_i_4ido = v4_load(cc, i + 4 * ido);
                let cc_i_4ido1 = v4_load(cc, i + 4 * ido + 1);

                let ti5 = v4_sub(cc_i_ido1, cc_i_4ido1);
                let ti2 = v4_add(cc_i_ido1, cc_i_4ido1);
                let ti4 = v4_sub(cc_i_2ido1, cc_i_3ido1);
                let ti3 = v4_add(cc_i_2ido1, cc_i_3ido1);
                let tr5 = v4_sub(cc_i_ido, cc_i_4ido);
                let tr2 = v4_add(cc_i_ido, cc_i_4ido);
                let tr4 = v4_sub(cc_i_2ido, cc_i_3ido);
                let tr3 = v4_add(cc_i_2ido, cc_i_3ido);

                v4_store(ch, i, v4_add(cc_i, v4_add(tr2, tr3)));
                v4_store(ch, i + 1, v4_add(cc_i1, v4_add(ti2, ti3)));

                let cr2 = v4_add(cc_i, v4_add(v4_scalar_mul(tr11, tr2), v4_scalar_mul(tr12, tr3)));
                let ci2 = v4_add(cc_i1, v4_add(v4_scalar_mul(tr11, ti2), v4_scalar_mul(tr12, ti3)));
                let cr3 = v4_add(cc_i, v4_add(v4_scalar_mul(tr12, tr2), v4_scalar_mul(tr11, tr3)));
                let ci3 = v4_add(cc_i1, v4_add(v4_scalar_mul(tr12, ti2), v4_scalar_mul(tr11, ti3)));

                let cr5 = v4_add(v4_scalar_mul(ti11, tr5), v4_scalar_mul(ti12, tr4));
                let ci5 = v4_add(v4_scalar_mul(ti11, ti5), v4_scalar_mul(ti12, ti4));
                let cr4 = v4_sub(v4_scalar_mul(ti12, tr5), v4_scalar_mul(ti11, tr4));
                let ci4 = v4_sub(v4_scalar_mul(ti12, ti5), v4_scalar_mul(ti11, ti4));

                let dr3 = v4_sub(cr3, ci4);
                let dr4 = v4_add(cr3, ci4);
                let di3 = v4_add(ci3, cr4);
                let di4 = v4_sub(ci3, cr4);
                let dr5 = v4_add(cr2, ci5);
                let dr2 = v4_sub(cr2, ci5);
                let di5 = v4_sub(ci2, cr5);
                let di2 = v4_add(ci2, cr5);

                let wr1 = *wa1.add(i);
                let wi1 = fsign * *wa1.add(i + 1);
                let wr2 = *wa2.add(i);
                let wi2 = fsign * *wa2.add(i + 1);
                let wr3 = *wa3.add(i);
                let wi3 = fsign * *wa3.add(i + 1);
                let wr4 = *wa4.add(i);
                let wi4 = fsign * *wa4.add(i + 1);

                let (dr2, di2) = v4_cplx_mul(dr2, di2, v4_splat(wr1), v4_splat(wi1));
                v4_store(ch, i + l1ido, dr2);
                v4_store(ch, i + l1ido + 1, di2);

                let (dr3, di3) = v4_cplx_mul(dr3, di3, v4_splat(wr2), v4_splat(wi2));
                v4_store(ch, i + 2 * l1ido, dr3);
                v4_store(ch, i + 2 * l1ido + 1, di3);

                let (dr4, di4) = v4_cplx_mul(dr4, di4, v4_splat(wr3), v4_splat(wi3));
                v4_store(ch, i + 3 * l1ido, dr4);
                v4_store(ch, i + 3 * l1ido + 1, di4);

                let (dr5, di5) = v4_cplx_mul(dr5, di5, v4_splat(wr4), v4_splat(wi4));
                v4_store(ch, i + 4 * l1ido, dr5);
                v4_store(ch, i + 4 * l1ido + 1, di5);
            }

            cc = cc.add(5 * ido * 4);
            ch = ch.add(ido * 4);
        }
    }
}

#[inline(never)]
unsafe fn radf2_ps(ido: usize, l1: usize, cc: *const f32, ch: *mut f32, wa1: *const f32) {
    let minus_one: f32 = -1.0;
    let l1ido = l1 * ido;

    unsafe {
        for k in (0..l1ido).step_by(ido) {
            let a = v4_load(cc, k);
            let b = v4_load(cc, k + l1ido);
            v4_store(ch, 2 * k, v4_add(a, b));
            v4_store(ch, 2 * (k + ido) - 1, v4_sub(a, b));
        }

        if ido < 2 {
            return;
        }

        if ido != 2 {
            for k in (0..l1ido).step_by(ido) {
                for i in (2..ido).step_by(2) {
                    let tr2 = v4_load(cc, i - 1 + k + l1ido);
                    let ti2 = v4_load(cc, i + k + l1ido);
                    let br = v4_load(cc, i - 1 + k);
                    let bi = v4_load(cc, i + k);

                    let wr = v4_splat(*wa1.add(i - 2));
                    let wi = v4_splat(*wa1.add(i - 1));
                    let (tr2, ti2) = v4_cplx_mul_conj(tr2, ti2, wr, wi);

                    v4_store(ch, i + 2 * k, v4_add(bi, ti2));
                    v4_store(ch, 2 * (k + ido) - i, v4_sub(ti2, bi));
                    v4_store(ch, i - 1 + 2 * k, v4_add(br, tr2));
                    v4_store(ch, 2 * (k + ido) - i - 1, v4_sub(br, tr2));
                }
            }

            if ido % 2 == 1 {
                return;
            }
        }

        for k in (0..l1ido).step_by(ido) {
            let cc_tail_l1 = v4_load(cc, ido - 1 + k + l1ido);
            let cc_tail = v4_load(cc, k + ido - 1);
            v4_store(ch, 2 * k + ido, v4_scalar_mul(minus_one, cc_tail_l1));
            v4_store(ch, 2 * k + ido - 1, cc_tail);
        }
    }
}

#[inline(never)]
unsafe fn radb2_ps(ido: usize, l1: usize, cc: *const f32, ch: *mut f32, wa1: *const f32) {
    let minus_two: f32 = -2.0;
    let l1ido = l1 * ido;

    unsafe {
        for k in (0..l1ido).step_by(ido) {
            let a = v4_load(cc, 2 * k);
            let b = v4_load(cc, 2 * (k + ido) - 1);
            v4_store(ch, k, v4_add(a, b));
            v4_store(ch, k + l1ido, v4_sub(a, b));
        }

        if ido < 2 {
            return;
        }

        if ido != 2 {
            for k in (0..l1ido).step_by(ido) {
                for i in (2..ido).step_by(2) {
                    let a = v4_load(cc, i - 1 + 2 * k);
                    let b = v4_load(cc, 2 * (k + ido) - i - 1);
                    let c = v4_load(cc, i + 2 * k);
                    let d = v4_load(cc, 2 * (k + ido) - i);

                    v4_store(ch, i - 1 + k, v4_add(a, b));
                    let tr2 = v4_sub(a, b);
                    v4_store(ch, i + k, v4_sub(c, d));
                    let ti2 = v4_add(c, d);

                    let wr1 = v4_splat(*wa1.add(i - 2));
                    let wi1 = v4_splat(*wa1.add(i - 1));
                    let (tr2, ti2) = v4_cplx_mul(tr2, ti2, wr1, wi1);
                    v4_store(ch, i - 1 + k + l1ido, tr2);
                    v4_store(ch, i + k + l1ido, ti2);
                }
            }

            if ido % 2 == 1 {
                return;
            }
        }

        for k in (0..l1ido).step_by(ido) {
            let a = v4_load(cc, 2 * k + ido - 1);
            let b = v4_load(cc, 2 * k + ido);
            v4_store(ch, k + ido - 1, v4_add(a, a));
            v4_store(ch, k + ido - 1 + l1ido, v4_scalar_mul(minus_two, b));
        }
    }
}

#[inline(never)]
unsafe fn radf3_ps(ido: usize, l1: usize, cc: *const f32, ch: *mut f32, wa1: *const f32, wa2: *const f32) {
    let taur: f32 = -0.5;
    let taui: f32 = 0.866025403784439;

    unsafe {
        for k in 0..l1 {
            let cc_k = v4_load(cc, k * ido);
            let cc_k_l1 = v4_load(cc, (k + l1) * ido);
            let cc_k_2l1 = v4_load(cc, (k + 2 * l1) * ido);
            let cr2 = v4_add(cc_k_l1, cc_k_2l1);
            v4_store(ch, 3 * k * ido, v4_add(cc_k, cr2));
            v4_store(ch, (3 * k + 2) * ido, v4_scalar_mul(taui, v4_sub(cc_k_2l1, cc_k_l1)));
            v4_store(ch, ido - 1 + (3 * k + 1) * ido, v4_add(cc_k, v4_scalar_mul(taur, cr2)));
        }

        if ido == 1 {
            return;
        }

        for k in 0..l1 {
            for i in (2..ido).step_by(2) {
                let ic = ido - i;
                let wr1 = v4_splat(*wa1.add(i - 2));
                let wi1 = v4_splat(*wa1.add(i - 1));
                let mut dr2 = v4_load(cc, i - 1 + (k + l1) * ido);
                let mut di2 = v4_load(cc, i + (k + l1) * ido);
                (dr2, di2) = v4_cplx_mul_conj(dr2, di2, wr1, wi1);

                let wr2 = v4_splat(*wa2.add(i - 2));
                let wi2 = v4_splat(*wa2.add(i - 1));
                let mut dr3 = v4_load(cc, i - 1 + (k + 2 * l1) * ido);
                let mut di3 = v4_load(cc, i + (k + 2 * l1) * ido);
                (dr3, di3) = v4_cplx_mul_conj(dr3, di3, wr2, wi2);

                let cc_i_k = v4_load(cc, i + k * ido);
                let cc_i1_k = v4_load(cc, i - 1 + k * ido);
                let cr2 = v4_add(dr2, dr3);
                let ci2 = v4_add(di2, di3);
                v4_store(ch, i - 1 + 3 * k * ido, v4_add(cc_i1_k, cr2));
                v4_store(ch, i + 3 * k * ido, v4_add(cc_i_k, ci2));

                let tr2 = v4_add(cc_i1_k, v4_scalar_mul(taur, cr2));
                let ti2 = v4_add(cc_i_k, v4_scalar_mul(taur, ci2));
                let tr3 = v4_scalar_mul(taui, v4_sub(di2, di3));
                let ti3 = v4_scalar_mul(taui, v4_sub(dr3, dr2));

                v4_store(ch, i - 1 + (3 * k + 2) * ido, v4_add(tr2, tr3));
                v4_store(ch, ic - 1 + (3 * k + 1) * ido, v4_sub(tr2, tr3));
                v4_store(ch, i + (3 * k + 2) * ido, v4_add(ti2, ti3));
                v4_store(ch, ic + (3 * k + 1) * ido, v4_sub(ti3, ti2));
            }
        }
    }
}

#[inline(never)]
unsafe fn radb3_ps(ido: usize, l1: usize, cc: *const f32, ch: *mut f32, wa1: *const f32, wa2: *const f32) {
    let taur: f32 = -0.5;
    let taui: f32 = 0.866025403784439;
    let taui_2: f32 = 1.732050807568878;

    unsafe {
        for k in 0..l1 {
            let cc_3k = v4_load(cc, 3 * k * ido);
            let mut tr2 = v4_load(cc, ido - 1 + (3 * k + 1) * ido);
            tr2 = v4_add(tr2, tr2);
            let cr2 = v4_add(cc_3k, v4_scalar_mul(taur, tr2));
            v4_store(ch, k * ido, v4_add(cc_3k, tr2));
            let ci3 = v4_scalar_mul(taui_2, v4_load(cc, (3 * k + 2) * ido));
            v4_store(ch, (k + l1) * ido, v4_sub(cr2, ci3));
            v4_store(ch, (k + 2 * l1) * ido, v4_add(cr2, ci3));
        }

        if ido == 1 {
            return;
        }

        for k in 0..l1 {
            for i in (2..ido).step_by(2) {
                let ic = ido - i;
                let cc_i1_3k = v4_load(cc, i - 1 + 3 * k * ido);
                let cc_i_3k = v4_load(cc, i + 3 * k * ido);
                let cc_i1_3k2 = v4_load(cc, i - 1 + (3 * k + 2) * ido);
                let cc_ic1_3k1 = v4_load(cc, ic - 1 + (3 * k + 1) * ido);
                let cc_i_3k2 = v4_load(cc, i + (3 * k + 2) * ido);
                let cc_ic_3k1 = v4_load(cc, ic + (3 * k + 1) * ido);

                let tr2 = v4_add(cc_i1_3k2, cc_ic1_3k1);
                let cr2 = v4_add(cc_i1_3k, v4_scalar_mul(taur, tr2));
                v4_store(ch, i - 1 + k * ido, v4_add(cc_i1_3k, tr2));
                let ti2 = v4_sub(cc_i_3k2, cc_ic_3k1);
                let ci2 = v4_add(cc_i_3k, v4_scalar_mul(taur, ti2));
                v4_store(ch, i + k * ido, v4_add(cc_i_3k, ti2));

                let cr3 = v4_scalar_mul(taui, v4_sub(cc_i1_3k2, cc_ic1_3k1));
                let ci3 = v4_scalar_mul(taui, v4_add(cc_i_3k2, cc_ic_3k1));
                let dr2 = v4_sub(cr2, ci3);
                let dr3 = v4_add(cr2, ci3);
                let di2 = v4_add(ci2, cr3);
                let di3 = v4_sub(ci2, cr3);

                let wr1 = v4_splat(*wa1.add(i - 2));
                let wi1 = v4_splat(*wa1.add(i - 1));
                let (dr2, di2) = v4_cplx_mul(dr2, di2, wr1, wi1);
                v4_store(ch, i - 1 + (k + l1) * ido, dr2);
                v4_store(ch, i + (k + l1) * ido, di2);

                let wr2 = v4_splat(*wa2.add(i - 2));
                let wi2 = v4_splat(*wa2.add(i - 1));
                let (dr3, di3) = v4_cplx_mul(dr3, di3, wr2, wi2);
                v4_store(ch, i - 1 + (k + 2 * l1) * ido, dr3);
                v4_store(ch, i + (k + 2 * l1) * ido, di3);
            }
        }
    }
}

#[inline(never)]
unsafe fn radf4_ps(
    ido: usize,
    l1: usize,
    mut cc: *const f32,
    mut ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    wa3: *const f32,
) {
    let minus_hsqt2: f32 = -0.7071067811865475;
    let l1ido = l1 * ido;

    unsafe {
        let cc_start = cc;
        let ch_start = ch;
        while cc < cc_start.add(l1ido * 4) {
            let a0 = v4_load(cc, 0);
            let a1 = v4_load(cc, l1ido);
            let a2 = v4_load(cc, 2 * l1ido);
            let a3 = v4_load(cc, 3 * l1ido);
            let tr1 = v4_add(a1, a3);
            let tr2 = v4_add(a0, a2);
            v4_store(ch, 2 * ido - 1, v4_sub(a0, a2));
            v4_store(ch, 2 * ido, v4_sub(a3, a1));
            v4_store(ch, 0, v4_add(tr1, tr2));
            v4_store(ch, 4 * ido - 1, v4_sub(tr2, tr1));
            cc = cc.add(ido * 4);
            ch = ch.add(4 * ido * 4);
        }
        cc = cc_start;
        ch = ch_start;

        if ido < 2 {
            return;
        }

        if ido != 2 {
            for k in (0..l1ido).step_by(ido) {
                for i in (2..ido).step_by(2) {
                    let ic = ido - i;
                    let mut cr2 = v4_load(cc, i - 1 + k + l1ido);
                    let mut ci2 = v4_load(cc, i + k + l1ido);
                    let wr1 = v4_splat(*wa1.add(i - 2));
                    let wi1 = v4_splat(*wa1.add(i - 1));
                    (cr2, ci2) = v4_cplx_mul_conj(cr2, ci2, wr1, wi1);

                    let mut cr3 = v4_load(cc, i - 1 + k + 2 * l1ido);
                    let mut ci3 = v4_load(cc, i + k + 2 * l1ido);
                    let wr2 = v4_splat(*wa2.add(i - 2));
                    let wi2 = v4_splat(*wa2.add(i - 1));
                    (cr3, ci3) = v4_cplx_mul_conj(cr3, ci3, wr2, wi2);

                    let mut cr4 = v4_load(cc, i - 1 + k + 3 * l1ido);
                    let mut ci4 = v4_load(cc, i + k + 3 * l1ido);
                    let wr3 = v4_splat(*wa3.add(i - 2));
                    let wi3 = v4_splat(*wa3.add(i - 1));
                    (cr4, ci4) = v4_cplx_mul_conj(cr4, ci4, wr3, wi3);

                    let cc_i1_k = v4_load(cc, i - 1 + k);
                    let cc_i_k = v4_load(cc, i + k);
                    let tr1 = v4_add(cr2, cr4);
                    let tr4 = v4_sub(cr4, cr2);
                    let tr2 = v4_add(cc_i1_k, cr3);
                    let tr3 = v4_sub(cc_i1_k, cr3);

                    v4_store(ch, i - 1 + 4 * k, v4_add(tr1, tr2));
                    v4_store(ch, ic - 1 + 4 * k + 3 * ido, v4_sub(tr2, tr1));
                    let ti1 = v4_add(ci2, ci4);
                    let ti4 = v4_sub(ci2, ci4);
                    v4_store(ch, i - 1 + 4 * k + 2 * ido, v4_add(ti4, tr3));
                    v4_store(ch, ic - 1 + 4 * k + 1 * ido, v4_sub(tr3, ti4));
                    let ti2 = v4_add(cc_i_k, ci3);
                    let ti3 = v4_sub(cc_i_k, ci3);
                    v4_store(ch, i + 4 * k, v4_add(ti1, ti2));
                    v4_store(ch, ic + 4 * k + 3 * ido, v4_sub(ti1, ti2));
                    v4_store(ch, i + 4 * k + 2 * ido, v4_add(tr4, ti3));
                    v4_store(ch, ic + 4 * k + 1 * ido, v4_sub(tr4, ti3));
                }
            }

            if ido % 2 == 1 {
                return;
            }
        }

        for k in (0..l1ido).step_by(ido) {
            let cc_tail_l1 = v4_load(cc, ido - 1 + k + l1ido);
            let cc_tail_3l1 = v4_load(cc, ido - 1 + k + 3 * l1ido);
            let cc_tail = v4_load(cc, ido - 1 + k);
            let cc_tail_2l1 = v4_load(cc, ido - 1 + k + 2 * l1ido);
            let ti1 = v4_scalar_mul(minus_hsqt2, v4_add(cc_tail_l1, cc_tail_3l1));
            let tr1 = v4_scalar_mul(minus_hsqt2, v4_sub(cc_tail_3l1, cc_tail_l1));
            v4_store(ch, ido - 1 + 4 * k, v4_add(tr1, cc_tail));
            v4_store(ch, ido - 1 + 4 * k + 2 * ido, v4_sub(cc_tail, tr1));
            v4_store(ch, 4 * k + 1 * ido, v4_sub(ti1, cc_tail_2l1));
            v4_store(ch, 4 * k + 3 * ido, v4_add(ti1, cc_tail_2l1));
        }
    }
}

#[inline(never)]
unsafe fn radb4_ps(
    ido: usize,
    l1: usize,
    mut cc: *const f32,
    mut ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    wa3: *const f32,
) {
    let minus_sqrt2: f32 = -1.414213562373095;
    let l1ido = l1 * ido;

    unsafe {
        let cc_start = cc;
        let ch_start = ch;
        while ch < ch_start.add(l1ido * 4) {
            let a = v4_load(cc, 0);
            let b = v4_load(cc, 4 * ido - 1);
            let c = v4_load(cc, 2 * ido);
            let d = v4_load(cc, 2 * ido - 1);
            let tr3 = v4_scalar_mul(2.0, d);
            let tr2 = v4_add(a, b);
            let tr1 = v4_sub(a, b);
            let tr4 = v4_scalar_mul(2.0, c);
            v4_store(ch, 0, v4_add(tr2, tr3));
            v4_store(ch, 2 * l1ido, v4_sub(tr2, tr3));
            v4_store(ch, 1 * l1ido, v4_sub(tr1, tr4));
            v4_store(ch, 3 * l1ido, v4_add(tr1, tr4));
            cc = cc.add(4 * ido * 4);
            ch = ch.add(ido * 4);
        }
        cc = cc_start;
        ch = ch_start;

        if ido < 2 {
            return;
        }

        if ido != 2 {
            for k in (0..l1ido).step_by(ido) {
                let pc = cc.add(4 * k * 4).sub(4);
                let mut ph = ch.add((k + 1) * 4);
                for i in (2..ido).step_by(2) {
                    let pc_i = v4_load(pc, i);
                    let pc_4ido_i = v4_load(pc, 4 * ido - i);
                    let pc_2ido_i = v4_load(pc, 2 * ido + i);
                    let pc_2ido_minus_i = v4_load(pc, 2 * ido - i);
                    let pc_2ido_i1 = v4_load(pc, 2 * ido + i + 1);
                    let pc_2ido_minus_i1 = v4_load(pc, 2 * ido - i + 1);
                    let pc_i1 = v4_load(pc, i + 1);
                    let pc_4ido_i1 = v4_load(pc, 4 * ido - i + 1);

                    let tr1 = v4_sub(pc_i, pc_4ido_i);
                    let tr2 = v4_add(pc_i, pc_4ido_i);
                    let ti4 = v4_sub(pc_2ido_i, pc_2ido_minus_i);
                    let tr3 = v4_add(pc_2ido_i, pc_2ido_minus_i);
                    v4_store(ph, 0, v4_add(tr2, tr3));
                    let cr3 = v4_sub(tr2, tr3);
                    let ti3 = v4_sub(pc_2ido_i1, pc_2ido_minus_i1);
                    let tr4 = v4_add(pc_2ido_i1, pc_2ido_minus_i1);
                    let cr2 = v4_sub(tr1, tr4);
                    let cr4 = v4_add(tr1, tr4);
                    let ti1 = v4_add(pc_i1, pc_4ido_i1);
                    let ti2 = v4_sub(pc_i1, pc_4ido_i1);
                    v4_store(ph, 1, v4_add(ti2, ti3));
                    ph = ph.add(l1ido * 4);
                    let ci3 = v4_sub(ti2, ti3);
                    let ci2 = v4_add(ti1, ti4);
                    let ci4 = v4_sub(ti1, ti4);
                    let wr1 = v4_splat(*wa1.add(i - 2));
                    let wi1 = v4_splat(*wa1.add(i - 1));
                    let (cr2, ci2) = v4_cplx_mul(cr2, ci2, wr1, wi1);
                    v4_store(ph, 0, cr2);
                    v4_store(ph, 1, ci2);
                    ph = ph.add(l1ido * 4);
                    let wr2 = v4_splat(*wa2.add(i - 2));
                    let wi2 = v4_splat(*wa2.add(i - 1));
                    let (cr3, ci3) = v4_cplx_mul(cr3, ci3, wr2, wi2);
                    v4_store(ph, 0, cr3);
                    v4_store(ph, 1, ci3);
                    ph = ph.add(l1ido * 4);
                    let wr3 = v4_splat(*wa3.add(i - 2));
                    let wi3 = v4_splat(*wa3.add(i - 1));
                    let (cr4, ci4) = v4_cplx_mul(cr4, ci4, wr3, wi3);
                    v4_store(ph, 0, cr4);
                    v4_store(ph, 1, ci4);
                    ph = ph.sub(3 * l1ido * 4).add(2 * 4);
                }
            }

            if ido % 2 == 1 {
                return;
            }
        }

        for k in (0..l1ido).step_by(ido) {
            let i0 = 4 * k + ido;
            let c = v4_load(cc, i0 - 1);
            let d = v4_load(cc, i0 + 2 * ido - 1);
            let a = v4_load(cc, i0);
            let b = v4_load(cc, i0 + 2 * ido);
            let tr1 = v4_sub(c, d);
            let tr2 = v4_add(c, d);
            let ti1 = v4_add(b, a);
            let ti2 = v4_sub(b, a);
            v4_store(ch, ido - 1 + k + 0 * l1ido, v4_add(tr2, tr2));
            v4_store(ch, ido - 1 + k + 1 * l1ido, v4_scalar_mul(minus_sqrt2, v4_sub(ti1, tr1)));
            v4_store(ch, ido - 1 + k + 2 * l1ido, v4_add(ti2, ti2));
            v4_store(ch, ido - 1 + k + 3 * l1ido, v4_scalar_mul(minus_sqrt2, v4_add(ti1, tr1)));
        }
    }
}

#[inline(never)]
unsafe fn radf5_ps(
    ido: usize,
    l1: usize,
    cc: *const f32,
    ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    wa3: *const f32,
    wa4: *const f32,
) {
    let tr11: f32 = 0.309016994374947;
    let ti11: f32 = 0.951056516295154;
    let tr12: f32 = -0.809016994374947;
    let ti12: f32 = 0.587785252292473;

    unsafe {
        for k in 0..l1 {
            let cc_1_k_5 = v4_load(cc, (4 * l1 + k) * ido);
            let cc_1_k_2 = v4_load(cc, (l1 + k) * ido);
            let cc_1_k_4 = v4_load(cc, (3 * l1 + k) * ido);
            let cc_1_k_3 = v4_load(cc, (2 * l1 + k) * ido);
            let cc_1_k_1 = v4_load(cc, k * ido);

            let cr2 = v4_add(cc_1_k_5, cc_1_k_2);
            let ci5 = v4_sub(cc_1_k_5, cc_1_k_2);
            let cr3 = v4_add(cc_1_k_4, cc_1_k_3);
            let ci4 = v4_sub(cc_1_k_4, cc_1_k_3);

            v4_store(ch, 5 * k * ido, v4_add(cc_1_k_1, v4_add(cr2, cr3)));
            v4_store(
                ch,
                (5 * k + 1) * ido + ido - 1,
                v4_add(cc_1_k_1, v4_add(v4_scalar_mul(tr11, cr2), v4_scalar_mul(tr12, cr3))),
            );
            v4_store(ch, (5 * k + 2) * ido, v4_add(v4_scalar_mul(ti11, ci5), v4_scalar_mul(ti12, ci4)));
            v4_store(
                ch,
                (5 * k + 3) * ido + ido - 1,
                v4_add(cc_1_k_1, v4_add(v4_scalar_mul(tr12, cr2), v4_scalar_mul(tr11, cr3))),
            );
            v4_store(ch, (5 * k + 4) * ido, v4_sub(v4_scalar_mul(ti12, ci5), v4_scalar_mul(ti11, ci4)));
        }

        if ido == 1 {
            return;
        }

        let idp2 = ido + 2;

        for k in 0..l1 {
            for i in (3..=ido).step_by(2) {
                let ic = idp2 - i;

                let mut dr2 = v4_splat(*wa1.add(i - 3));
                let mut di2 = v4_splat(*wa1.add(i - 2));
                let mut dr3 = v4_splat(*wa2.add(i - 3));
                let mut di3 = v4_splat(*wa2.add(i - 2));
                let mut dr4 = v4_splat(*wa3.add(i - 3));
                let mut di4 = v4_splat(*wa3.add(i - 2));
                let mut dr5 = v4_splat(*wa4.add(i - 3));
                let mut di5 = v4_splat(*wa4.add(i - 2));

                (dr2, di2) = v4_cplx_mul_conj(
                    dr2,
                    di2,
                    v4_load(cc, (l1 + k) * ido + i - 2),
                    v4_load(cc, (l1 + k) * ido + i - 1),
                );
                (dr3, di3) = v4_cplx_mul_conj(
                    dr3,
                    di3,
                    v4_load(cc, (2 * l1 + k) * ido + i - 2),
                    v4_load(cc, (2 * l1 + k) * ido + i - 1),
                );
                (dr4, di4) = v4_cplx_mul_conj(
                    dr4,
                    di4,
                    v4_load(cc, (3 * l1 + k) * ido + i - 2),
                    v4_load(cc, (3 * l1 + k) * ido + i - 1),
                );
                (dr5, di5) = v4_cplx_mul_conj(
                    dr5,
                    di5,
                    v4_load(cc, (4 * l1 + k) * ido + i - 2),
                    v4_load(cc, (4 * l1 + k) * ido + i - 1),
                );

                let cr2 = v4_add(dr2, dr5);
                let ci5 = v4_sub(dr5, dr2);
                let cr5 = v4_sub(di2, di5);
                let ci2 = v4_add(di2, di5);
                let cr3 = v4_add(dr3, dr4);
                let ci4 = v4_sub(dr4, dr3);
                let cr4 = v4_sub(di3, di4);
                let ci3 = v4_add(di3, di4);

                let cc_i1_k_1 = v4_load(cc, k * ido + i - 2);
                let cc_i_k_1 = v4_load(cc, k * ido + i - 1);

                v4_store(ch, 5 * k * ido + i - 2, v4_add(cc_i1_k_1, v4_add(cr2, cr3)));
                v4_store(ch, 5 * k * ido + i - 1, v4_sub(cc_i_k_1, v4_add(ci2, ci3)));

                let tr2 = v4_add(cc_i1_k_1, v4_add(v4_scalar_mul(tr11, cr2), v4_scalar_mul(tr12, cr3)));
                let ti2 = v4_sub(cc_i_k_1, v4_add(v4_scalar_mul(tr11, ci2), v4_scalar_mul(tr12, ci3)));
                let tr3 = v4_add(cc_i1_k_1, v4_add(v4_scalar_mul(tr12, cr2), v4_scalar_mul(tr11, cr3)));
                let ti3 = v4_sub(cc_i_k_1, v4_add(v4_scalar_mul(tr12, ci2), v4_scalar_mul(tr11, ci3)));
                let tr5 = v4_add(v4_scalar_mul(ti11, cr5), v4_scalar_mul(ti12, cr4));
                let ti5 = v4_add(v4_scalar_mul(ti11, ci5), v4_scalar_mul(ti12, ci4));
                let tr4 = v4_sub(v4_scalar_mul(ti12, cr5), v4_scalar_mul(ti11, cr4));
                let ti4 = v4_sub(v4_scalar_mul(ti12, ci5), v4_scalar_mul(ti11, ci4));

                v4_store(ch, (5 * k + 2) * ido + i - 2, v4_sub(tr2, tr5));
                v4_store(ch, (5 * k + 1) * ido + ic - 2, v4_add(tr2, tr5));
                v4_store(ch, (5 * k + 2) * ido + i - 1, v4_add(ti2, ti5));
                v4_store(ch, (5 * k + 1) * ido + ic - 1, v4_sub(ti5, ti2));
                v4_store(ch, (5 * k + 4) * ido + i - 2, v4_sub(tr3, tr4));
                v4_store(ch, (5 * k + 3) * ido + ic - 2, v4_add(tr3, tr4));
                v4_store(ch, (5 * k + 4) * ido + i - 1, v4_add(ti3, ti4));
                v4_store(ch, (5 * k + 3) * ido + ic - 1, v4_sub(ti4, ti3));
            }
        }
    }
}

#[inline(never)]
unsafe fn radb5_ps(
    ido: usize,
    l1: usize,
    cc: *const f32,
    ch: *mut f32,
    wa1: *const f32,
    wa2: *const f32,
    wa3: *const f32,
    wa4: *const f32,
) {
    let tr11: f32 = 0.309016994374947;
    let ti11: f32 = 0.951056516295154;
    let tr12: f32 = -0.809016994374947;
    let ti12: f32 = 0.587785252292473;

    unsafe {
        for k in 0..l1 {
            let cc_1_3_k = v4_load(cc, (5 * k + 2) * ido);
            let cc_1_5_k = v4_load(cc, (5 * k + 4) * ido);
            let cc_ido_2_k = v4_load(cc, (5 * k + 1) * ido + ido - 1);
            let cc_ido_4_k = v4_load(cc, (5 * k + 3) * ido + ido - 1);
            let cc_1_1_k = v4_load(cc, 5 * k * ido);

            let ti5 = v4_add(cc_1_3_k, cc_1_3_k);
            let ti4 = v4_add(cc_1_5_k, cc_1_5_k);
            let tr2 = v4_add(cc_ido_2_k, cc_ido_2_k);
            let tr3 = v4_add(cc_ido_4_k, cc_ido_4_k);

            v4_store(ch, k * ido, v4_add(cc_1_1_k, v4_add(tr2, tr3)));

            let cr2 = v4_add(cc_1_1_k, v4_add(v4_scalar_mul(tr11, tr2), v4_scalar_mul(tr12, tr3)));
            let cr3 = v4_add(cc_1_1_k, v4_add(v4_scalar_mul(tr12, tr2), v4_scalar_mul(tr11, tr3)));
            let ci5 = v4_add(v4_scalar_mul(ti11, ti5), v4_scalar_mul(ti12, ti4));
            let ci4 = v4_sub(v4_scalar_mul(ti12, ti5), v4_scalar_mul(ti11, ti4));

            v4_store(ch, (l1 + k) * ido, v4_sub(cr2, ci5));
            v4_store(ch, (2 * l1 + k) * ido, v4_sub(cr3, ci4));
            v4_store(ch, (3 * l1 + k) * ido, v4_add(cr3, ci4));
            v4_store(ch, (4 * l1 + k) * ido, v4_add(cr2, ci5));
        }

        if ido == 1 {
            return;
        }

        let idp2 = ido + 2;

        for k in 0..l1 {
            for i in (3..=ido).step_by(2) {
                let ic = idp2 - i;

                let cc_i_3_k = v4_load(cc, (5 * k + 2) * ido + i - 1);
                let cc_ic_2_k = v4_load(cc, (5 * k + 1) * ido + ic - 1);
                let cc_i_5_k = v4_load(cc, (5 * k + 4) * ido + i - 1);
                let cc_ic_4_k = v4_load(cc, (5 * k + 3) * ido + ic - 1);
                let cc_i1_3_k = v4_load(cc, (5 * k + 2) * ido + i - 2);
                let cc_ic1_2_k = v4_load(cc, (5 * k + 1) * ido + ic - 2);
                let cc_i1_5_k = v4_load(cc, (5 * k + 4) * ido + i - 2);
                let cc_ic1_4_k = v4_load(cc, (5 * k + 3) * ido + ic - 2);
                let cc_i1_1_k = v4_load(cc, 5 * k * ido + i - 2);
                let cc_i_1_k = v4_load(cc, 5 * k * ido + i - 1);

                let ti5 = v4_add(cc_i_3_k, cc_ic_2_k);
                let ti2 = v4_sub(cc_i_3_k, cc_ic_2_k);
                let ti4 = v4_add(cc_i_5_k, cc_ic_4_k);
                let ti3 = v4_sub(cc_i_5_k, cc_ic_4_k);
                let tr5 = v4_sub(cc_i1_3_k, cc_ic1_2_k);
                let tr2 = v4_add(cc_i1_3_k, cc_ic1_2_k);
                let tr4 = v4_sub(cc_i1_5_k, cc_ic1_4_k);
                let tr3 = v4_add(cc_i1_5_k, cc_ic1_4_k);

                v4_store(ch, k * ido + i - 2, v4_add(cc_i1_1_k, v4_add(tr2, tr3)));
                v4_store(ch, k * ido + i - 1, v4_add(cc_i_1_k, v4_add(ti2, ti3)));

                let cr2 = v4_add(cc_i1_1_k, v4_add(v4_scalar_mul(tr11, tr2), v4_scalar_mul(tr12, tr3)));
                let ci2 = v4_add(cc_i_1_k, v4_add(v4_scalar_mul(tr11, ti2), v4_scalar_mul(tr12, ti3)));
                let cr3 = v4_add(cc_i1_1_k, v4_add(v4_scalar_mul(tr12, tr2), v4_scalar_mul(tr11, tr3)));
                let ci3 = v4_add(cc_i_1_k, v4_add(v4_scalar_mul(tr12, ti2), v4_scalar_mul(tr11, ti3)));
                let cr5 = v4_add(v4_scalar_mul(ti11, tr5), v4_scalar_mul(ti12, tr4));
                let ci5 = v4_add(v4_scalar_mul(ti11, ti5), v4_scalar_mul(ti12, ti4));
                let cr4 = v4_sub(v4_scalar_mul(ti12, tr5), v4_scalar_mul(ti11, tr4));
                let ci4 = v4_sub(v4_scalar_mul(ti12, ti5), v4_scalar_mul(ti11, ti4));

                let dr3 = v4_sub(cr3, ci4);
                let dr4 = v4_add(cr3, ci4);
                let di3 = v4_add(ci3, cr4);
                let di4 = v4_sub(ci3, cr4);
                let dr5 = v4_add(cr2, ci5);
                let dr2 = v4_sub(cr2, ci5);
                let di5 = v4_sub(ci2, cr5);
                let di2 = v4_add(ci2, cr5);

                let (dr2, di2) = v4_cplx_mul(dr2, di2, v4_splat(*wa1.add(i - 3)), v4_splat(*wa1.add(i - 2)));
                let (dr3, di3) = v4_cplx_mul(dr3, di3, v4_splat(*wa2.add(i - 3)), v4_splat(*wa2.add(i - 2)));
                let (dr4, di4) = v4_cplx_mul(dr4, di4, v4_splat(*wa3.add(i - 3)), v4_splat(*wa3.add(i - 2)));
                let (dr5, di5) = v4_cplx_mul(dr5, di5, v4_splat(*wa4.add(i - 3)), v4_splat(*wa4.add(i - 2)));

                v4_store(ch, (l1 + k) * ido + i - 2, dr2);
                v4_store(ch, (l1 + k) * ido + i - 1, di2);
                v4_store(ch, (2 * l1 + k) * ido + i - 2, dr3);
                v4_store(ch, (2 * l1 + k) * ido + i - 1, di3);
                v4_store(ch, (3 * l1 + k) * ido + i - 2, dr4);
                v4_store(ch, (3 * l1 + k) * ido + i - 1, di4);
                v4_store(ch, (4 * l1 + k) * ido + i - 2, dr5);
                v4_store(ch, (4 * l1 + k) * ido + i - 1, di5);
            }
        }
    }
}

#[inline(never)]
unsafe fn rfftf1_ps(
    n: usize,
    input_readonly: *const f32,
    work1: *mut f32,
    work2: *mut f32,
    wa: *const f32,
    ifac: *const usize,
) -> *mut f32 {
    unsafe {
        let mut in_ptr = input_readonly as *mut f32;
        let mut out_ptr = if in_ptr == work2 { work1 } else { work2 };
        let nf = *ifac.add(1);
        let mut l2 = n;
        let mut iw = n as isize - 1;
        assert!(in_ptr != out_ptr && work1 != work2);
        for k1 in 1..=nf {
            let kh = nf - k1;
            let ip = *ifac.add(kh + 2);
            let l1 = l2 / ip;
            let ido = n / l2;
            iw -= (ip as isize - 1) * ido as isize;
            match ip {
                5 => {
                    let ix2 = iw + ido as isize;
                    let ix3 = ix2 + ido as isize;
                    let ix4 = ix3 + ido as isize;
                    radf5_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2), wa.offset(ix3), wa.offset(ix4));
                }
                4 => {
                    let ix2 = iw + ido as isize;
                    let ix3 = ix2 + ido as isize;
                    radf4_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2), wa.offset(ix3));
                }
                3 => {
                    let ix2 = iw + ido as isize;
                    radf3_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2));
                }
                2 => {
                    radf2_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw));
                }
                _ => {
                    unreachable!();
                }
            }
            l2 = l1;
            if out_ptr == work2 {
                out_ptr = work1;
                in_ptr = work2;
            } else {
                out_ptr = work2;
                in_ptr = work1;
            }
        }
        in_ptr
    }
}

#[inline(never)]
unsafe fn rfftb1_ps(
    n: usize,
    input_readonly: *const f32,
    work1: *mut f32,
    work2: *mut f32,
    wa: *const f32,
    ifac: *const usize,
) -> *mut f32 {
    unsafe {
        let mut in_ptr = input_readonly as *mut f32;
        let mut out_ptr = if in_ptr == work2 { work1 } else { work2 };
        let nf = *ifac.add(1);
        let mut l1 = 1;
        let mut iw = 0;
        assert!(in_ptr != out_ptr);
        for k1 in 1..=nf {
            let ip = *ifac.add(k1 + 1);
            let l2 = ip * l1;
            let ido = n / l2;
            match ip {
                5 => {
                    let ix2 = iw + ido as isize;
                    let ix3 = ix2 + ido as isize;
                    let ix4 = ix3 + ido as isize;
                    radb5_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2), wa.offset(ix3), wa.offset(ix4));
                }
                4 => {
                    let ix2 = iw + ido as isize;
                    let ix3 = ix2 + ido as isize;
                    radb4_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2), wa.offset(ix3));
                }
                3 => {
                    let ix2 = iw + ido as isize;
                    radb3_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2));
                }
                2 => {
                    radb2_ps(ido, l1, in_ptr, out_ptr, wa.offset(iw));
                }
                _ => {
                    unreachable!();
                }
            }
            l1 = l2;
            iw += (ip as isize - 1) * ido as isize;
            if out_ptr == work2 {
                out_ptr = work1;
                in_ptr = work2;
            } else {
                out_ptr = work2;
                in_ptr = work1;
            }
        }
        in_ptr
    }
}

#[inline(never)]
unsafe fn reversed_copy(n: usize, mut input: *const f32, in_stride: isize, mut out: *mut f32) {
    unsafe {
        let (g0, mut g1) = v4_interleave2(v4_load(input, 0), v4_load(input, 1));
        input = input.offset(in_stride * 4);

        out = out.sub(4);
        v4_store(out, 0, v4_swaphl(g0, g1));

        for _ in 1..n {
            let (h0_raw, h1) = v4_interleave2(v4_load(input, 0), v4_load(input, 1));
            input = input.offset(in_stride * 4);

            out = out.sub(4);
            v4_store(out, 0, v4_swaphl(g1, h0_raw));
            out = out.sub(4);
            v4_store(out, 0, v4_swaphl(h0_raw, h1));
            g1 = h1;
        }

        out = out.sub(4);
        v4_store(out, 0, v4_swaphl(g1, g0));
    }
}

#[inline(never)]
unsafe fn unreversed_copy(n: usize, mut input: *const f32, mut out: *mut f32, out_stride: isize) {
    unsafe {
        let g0 = v4_load(input, 0);
        let mut g1 = g0;
        input = input.add(4);

        for _ in 1..n {
            let mut h0 = v4_load(input, 0);
            let h1 = v4_load(input, 1);
            input = input.add(8);

            g1 = v4_swaphl(g1, h0);
            h0 = v4_swaphl(h0, h1);

            let (out0, out1) = v4_uninterleave2(h0, g1);
            v4_store(out, 0, out0);
            v4_store(out, 1, out1);
            out = out.offset(out_stride * 4);

            g1 = h1;
        }

        let mut h0 = v4_load(input, 0);
        let h1 = g0;

        g1 = v4_swaphl(g1, h0);
        h0 = v4_swaphl(h0, h1);

        let (out0, out1) = v4_uninterleave2(h0, g1);
        v4_store(out, 0, out0);
        v4_store(out, 1, out1);
    }
}

#[inline(never)]
unsafe fn pffft_cplx_finalize(ncvec: usize, input: *const f32, output: *mut f32, e: *const f32) {
    unsafe {
        let dk = ncvec / SIMD_SZ;
        assert!(input != output as *const f32);

        for k in 0..dk {
            let mut r0 = v4_load(input, 8 * k);
            let mut i0 = v4_load(input, 8 * k + 1);
            let mut r1 = v4_load(input, 8 * k + 2);
            let mut i1 = v4_load(input, 8 * k + 3);
            let mut r2 = v4_load(input, 8 * k + 4);
            let mut i2 = v4_load(input, 8 * k + 5);
            let mut r3 = v4_load(input, 8 * k + 6);
            let mut i3 = v4_load(input, 8 * k + 7);

            (r0, r1, r2, r3) = v4_transpose(r0, r1, r2, r3);
            (i0, i1, i2, i3) = v4_transpose(i0, i1, i2, i3);

            let (nr1, ni1) = v4_cplx_mul(r1, i1, v4_load(e, k * 6), v4_load(e, k * 6 + 1));
            r1 = nr1;
            i1 = ni1;
            let (nr2, ni2) = v4_cplx_mul(r2, i2, v4_load(e, k * 6 + 2), v4_load(e, k * 6 + 3));
            r2 = nr2;
            i2 = ni2;
            let (nr3, ni3) = v4_cplx_mul(r3, i3, v4_load(e, k * 6 + 4), v4_load(e, k * 6 + 5));
            r3 = nr3;
            i3 = ni3;

            let sr0 = v4_add(r0, r2);
            let dr0 = v4_sub(r0, r2);
            let sr1 = v4_add(r1, r3);
            let dr1 = v4_sub(r1, r3);
            let si0 = v4_add(i0, i2);
            let di0 = v4_sub(i0, i2);
            let si1 = v4_add(i1, i3);
            let di1 = v4_sub(i1, i3);

            r0 = v4_add(sr0, sr1);
            i0 = v4_add(si0, si1);
            r1 = v4_add(dr0, di1);
            i1 = v4_sub(di0, dr1);
            r2 = v4_sub(sr0, sr1);
            i2 = v4_sub(si0, si1);
            r3 = v4_sub(dr0, di1);
            i3 = v4_add(di0, dr1);

            v4_store(output, 8 * k, r0);
            v4_store(output, 8 * k + 1, i0);
            v4_store(output, 8 * k + 2, r1);
            v4_store(output, 8 * k + 3, i1);
            v4_store(output, 8 * k + 4, r2);
            v4_store(output, 8 * k + 5, i2);
            v4_store(output, 8 * k + 6, r3);
            v4_store(output, 8 * k + 7, i3);
        }
    }
}

#[inline(never)]
unsafe fn pffft_cplx_preprocess(ncvec: usize, input: *const f32, output: *mut f32, e: *const f32) {
    unsafe {
        let dk = ncvec / SIMD_SZ;
        assert!(input != output as *const f32);

        for k in 0..dk {
            let mut r0 = v4_load(input, 8 * k);
            let mut i0 = v4_load(input, 8 * k + 1);
            let mut r1 = v4_load(input, 8 * k + 2);
            let mut i1 = v4_load(input, 8 * k + 3);
            let mut r2 = v4_load(input, 8 * k + 4);
            let mut i2 = v4_load(input, 8 * k + 5);
            let mut r3 = v4_load(input, 8 * k + 6);
            let mut i3 = v4_load(input, 8 * k + 7);

            let sr0 = v4_add(r0, r2);
            let dr0 = v4_sub(r0, r2);
            let sr1 = v4_add(r1, r3);
            let dr1 = v4_sub(r1, r3);
            let si0 = v4_add(i0, i2);
            let di0 = v4_sub(i0, i2);
            let si1 = v4_add(i1, i3);
            let di1 = v4_sub(i1, i3);

            r0 = v4_add(sr0, sr1);
            i0 = v4_add(si0, si1);
            r1 = v4_sub(dr0, di1);
            i1 = v4_add(di0, dr1);
            r2 = v4_sub(sr0, sr1);
            i2 = v4_sub(si0, si1);
            r3 = v4_add(dr0, di1);
            i3 = v4_sub(di0, dr1);

            let (nr1, ni1) = v4_cplx_mul_conj(r1, i1, v4_load(e, k * 6), v4_load(e, k * 6 + 1));
            r1 = nr1;
            i1 = ni1;
            let (nr2, ni2) = v4_cplx_mul_conj(r2, i2, v4_load(e, k * 6 + 2), v4_load(e, k * 6 + 3));
            r2 = nr2;
            i2 = ni2;
            let (nr3, ni3) = v4_cplx_mul_conj(r3, i3, v4_load(e, k * 6 + 4), v4_load(e, k * 6 + 5));
            r3 = nr3;
            i3 = ni3;

            (r0, r1, r2, r3) = v4_transpose(r0, r1, r2, r3);
            (i0, i1, i2, i3) = v4_transpose(i0, i1, i2, i3);

            v4_store(output, 8 * k, r0);
            v4_store(output, 8 * k + 1, i0);
            v4_store(output, 8 * k + 2, r1);
            v4_store(output, 8 * k + 3, i1);
            v4_store(output, 8 * k + 4, r2);
            v4_store(output, 8 * k + 5, i2);
            v4_store(output, 8 * k + 6, r3);
            v4_store(output, 8 * k + 7, i3);
        }
    }
}

#[inline(never)]
unsafe fn pffft_real_finalize_4x4(in0: vector4::V4sf, in1: vector4::V4sf, input: *const f32, e: *const f32, output: *mut f32) {
    unsafe {
        let mut r0 = in0;
        let mut i0 = in1;
        let mut r1 = v4_load(input, 0);
        let mut i1 = v4_load(input, 1);
        let mut r2 = v4_load(input, 2);
        let mut i2 = v4_load(input, 3);
        let mut r3 = v4_load(input, 4);
        let mut i3 = v4_load(input, 5);

        (r0, r1, r2, r3) = v4_transpose(r0, r1, r2, r3);
        (i0, i1, i2, i3) = v4_transpose(i0, i1, i2, i3);

        let (nr1, ni1) = v4_cplx_mul(r1, i1, v4_load(e, 0), v4_load(e, 1));
        r1 = nr1;
        i1 = ni1;
        let (nr2, ni2) = v4_cplx_mul(r2, i2, v4_load(e, 2), v4_load(e, 3));
        r2 = nr2;
        i2 = ni2;
        let (nr3, ni3) = v4_cplx_mul(r3, i3, v4_load(e, 4), v4_load(e, 5));
        r3 = nr3;
        i3 = ni3;

        let sr0 = v4_add(r0, r2);
        let dr0 = v4_sub(r0, r2);
        let sr1 = v4_add(r1, r3);
        let dr1 = v4_sub(r3, r1);
        let si0 = v4_add(i0, i2);
        let di0 = v4_sub(i0, i2);
        let si1 = v4_add(i1, i3);
        let di1 = v4_sub(i3, i1);

        r0 = v4_add(sr0, sr1);
        r3 = v4_sub(sr0, sr1);
        i0 = v4_add(si0, si1);
        i3 = v4_sub(si1, si0);
        r1 = v4_add(dr0, di1);
        r2 = v4_sub(dr0, di1);
        i1 = v4_sub(dr1, di0);
        i2 = v4_add(dr1, di0);

        v4_store(output, 0, r0);
        v4_store(output, 1, i0);
        v4_store(output, 2, r1);
        v4_store(output, 3, i1);
        v4_store(output, 4, r2);
        v4_store(output, 5, i2);
        v4_store(output, 6, r3);
        v4_store(output, 7, i3);
    }
}

#[inline(never)]
unsafe fn pffft_real_finalize(ncvec: usize, input: *const f32, output: *mut f32, e: *const f32) {
    unsafe {
        let dk = ncvec / SIMD_SZ;
        assert!(input != output as *const f32);

        let mut save = v4_load(input, 7);
        let zero = v4_splat(0.0);
        pffft_real_finalize_4x4(zero, zero, input.add(4), e, output);

        let mut cr = [0.0f32; 4];
        let mut ci = [0.0f32; 4];
        let mut xr = [0.0f32; 4];
        let mut xi = [0.0f32; 4];
        let s = std::f32::consts::FRAC_1_SQRT_2;

        v4_store(cr.as_mut_ptr(), 0, v4_load(input, 0));
        v4_store(ci.as_mut_ptr(), 0, v4_load(input, ncvec * 2 - 1));

        xr[0] = (cr[0] + cr[2]) + (cr[1] + cr[3]);
        xi[0] = (cr[0] + cr[2]) - (cr[1] + cr[3]);
        xr[2] = cr[0] - cr[2];
        xi[2] = cr[3] - cr[1];
        xr[1] = ci[0] + s * (ci[1] - ci[3]);
        xi[1] = -ci[2] - s * (ci[1] + ci[3]);
        xr[3] = ci[0] - s * (ci[1] - ci[3]);
        xi[3] = ci[2] - s * (ci[1] + ci[3]);

        let mut out_scalar = output as *mut f32;
        *out_scalar.add(0) = xr[0];
        *out_scalar.add(4) = xi[0];
        *out_scalar.add(16) = xr[2];
        *out_scalar.add(20) = xi[2];
        *out_scalar.add(8) = xr[1];
        *out_scalar.add(12) = xi[1];
        *out_scalar.add(24) = xr[3];
        *out_scalar.add(28) = xi[3];

        for k in 1..dk {
            let save_next = v4_load(input, 8 * k + 7);
            pffft_real_finalize_4x4(
                save,
                v4_load(input, 8 * k),
                input.add((8 * k + 1) * 4),
                e.add((k * 6) * 4),
                output.add((k * 8) * 4),
            );
            save = save_next;
        }
    }
}

#[inline(never)]
unsafe fn pffft_real_preprocess_4x4(input: *const f32, e: *const f32, output: *mut f32, first: bool) {
    unsafe {
        let mut r0 = v4_load(input, 0);
        let mut i0 = v4_load(input, 1);
        let mut r1 = v4_load(input, 2);
        let mut i1 = v4_load(input, 3);
        let mut r2 = v4_load(input, 4);
        let mut i2 = v4_load(input, 5);
        let mut r3 = v4_load(input, 6);
        let mut i3 = v4_load(input, 7);

        let sr0 = v4_add(r0, r3);
        let dr0 = v4_sub(r0, r3);
        let sr1 = v4_add(r1, r2);
        let dr1 = v4_sub(r1, r2);
        let si0 = v4_add(i0, i3);
        let di0 = v4_sub(i0, i3);
        let si1 = v4_add(i1, i2);
        let di1 = v4_sub(i1, i2);

        r0 = v4_add(sr0, sr1);
        r2 = v4_sub(sr0, sr1);
        r1 = v4_sub(dr0, si1);
        r3 = v4_add(dr0, si1);
        i0 = v4_sub(di0, di1);
        i2 = v4_add(di0, di1);
        i1 = v4_sub(si0, dr1);
        i3 = v4_add(si0, dr1);

        let (nr1, ni1) = v4_cplx_mul_conj(r1, i1, v4_load(e, 0), v4_load(e, 1));
        r1 = nr1;
        i1 = ni1;
        let (nr2, ni2) = v4_cplx_mul_conj(r2, i2, v4_load(e, 2), v4_load(e, 3));
        r2 = nr2;
        i2 = ni2;
        let (nr3, ni3) = v4_cplx_mul_conj(r3, i3, v4_load(e, 4), v4_load(e, 5));
        r3 = nr3;
        i3 = ni3;

        (r0, r1, r2, r3) = v4_transpose(r0, r1, r2, r3);
        (i0, i1, i2, i3) = v4_transpose(i0, i1, i2, i3);

        let mut out = output;
        if !first {
            v4_store(out, 0, r0);
            v4_store(out, 1, i0);
            out = out.add(8);
        }
        v4_store(out, 0, r1);
        v4_store(out, 1, i1);
        v4_store(out, 2, r2);
        v4_store(out, 3, i2);
        v4_store(out, 4, r3);
        v4_store(out, 5, i3);
    }
}

#[inline(never)]
unsafe fn pffft_real_preprocess(ncvec: usize, input: *const f32, output: *mut f32, e: *const f32) {
    unsafe {
        let dk = ncvec / SIMD_SZ;
        assert!(input != output as *const f32);

        let mut xr = [0.0f32; 4];
        let mut xi = [0.0f32; 4];
        let mut cr = [0.0f32; 4];
        let mut ci = [0.0f32; 4];
        let s = std::f32::consts::SQRT_2;

        for k in 0..4 {
            xr[k] = *input.add(8 * k);
            xi[k] = *input.add(8 * k + 4);
        }

        pffft_real_preprocess_4x4(input, e, output.add(4), true);
        for k in 1..dk {
            pffft_real_preprocess_4x4(
                input.add((8 * k) * 4),
                e.add((k * 6) * 4),
                output.add(((k * 8) - 1) * 4),
                false,
            );
        }

        cr[0] = (xr[0] + xi[0]) + 2.0 * xr[2];
        cr[1] = (xr[0] - xi[0]) - 2.0 * xi[2];
        cr[2] = (xr[0] + xi[0]) - 2.0 * xr[2];
        cr[3] = (xr[0] - xi[0]) + 2.0 * xi[2];
        ci[0] = 2.0 * (xr[1] + xr[3]);
        ci[1] = s * (xr[1] - xr[3]) - s * (xi[1] + xi[3]);
        ci[2] = 2.0 * (xi[3] - xi[1]);
        ci[3] = -s * (xr[1] - xr[3]) - s * (xi[1] + xi[3]);

        v4_store(output, 0, v4_load(cr.as_ptr(), 0));
        v4_store(output, 2 * ncvec - 1, v4_load(ci.as_ptr(), 0));
    }
}

#[inline(never)]
pub unsafe fn pffft_transform_internal(
    setup: &PFFFTSetup,
    finput: *const f32,
    foutput: *mut f32,
    scratch: *mut f32,
    direction: PffftDirection,
    ordered: bool,
) {
    unsafe {
        let ncvec = setup.ncvec;
        let nf_odd = (setup.ifac[1] & 1) != 0;

        assert!(finput as usize != 0);
        assert!(foutput as usize != 0);

        let mut local_scratch = vec![0.0f32; ncvec * 2 * SIMD_SZ];
        let scratch_ptr = if scratch.is_null() { local_scratch.as_mut_ptr() } else { scratch };

        let mut ib = if nf_odd ^ ordered { 1usize } else { 0usize };
        let mut vinput = finput;

        let mut buff0 = foutput;
        let mut buff1 = scratch_ptr;

        if direction == PffftDirection::Forward {
            ib ^= 1;
            if setup.transform == PffftTransform::Real {
                let r = rfftf1_ps(
                    ncvec * 2,
                    vinput,
                    if ib == 0 { buff0 } else { buff1 },
                    if ib == 0 { buff1 } else { buff0 },
                    setup.data.as_ptr().add(setup.twiddle_offset),
                    setup.ifac.as_ptr(),
                );
                ib = if r == buff0 { 0 } else { 1 };
                pffft_real_finalize(
                    ncvec,
                    if ib == 0 { buff0 } else { buff1 },
                    if ib == 0 { buff1 } else { buff0 },
                    setup.data.as_ptr().add(setup.e_offset),
                );
            } else {
                let tmp = if ib == 0 { buff0 } else { buff1 };
                for k in 0..ncvec {
                    let (a, b) = v4_uninterleave2(v4_load(vinput, k * 2), v4_load(vinput, k * 2 + 1));
                    v4_store(tmp, k * 2, a);
                    v4_store(tmp, k * 2 + 1, b);
                }
                let r = cfftf1_ps(
                    ncvec,
                    if ib == 0 { buff0 } else { buff1 },
                    if ib == 0 { buff1 } else { buff0 },
                    if ib == 0 { buff0 } else { buff1 },
                    setup.data.as_ptr().add(setup.twiddle_offset),
                    setup.ifac.as_ptr(),
                    -1.0,
                );
                ib = if r == buff0 { 0 } else { 1 };
                pffft_cplx_finalize(
                    ncvec,
                    if ib == 0 { buff0 } else { buff1 },
                    if ib == 0 { buff1 } else { buff0 },
                    setup.data.as_ptr().add(setup.e_offset),
                );
            }
            if ordered {
                pffft_zreorder(
                    setup,
                    if ib == 0 { buff1 } else { buff0 },
                    if ib == 0 { buff0 } else { buff1 },
                    PffftDirection::Forward,
                );
            } else {
                ib ^= 1;
            }
        } else {
            if vinput == if ib == 0 { buff0 } else { buff1 } {
                ib ^= 1;
            }
            if ordered {
                pffft_zreorder(
                    setup,
                    vinput,
                    if ib == 0 { buff0 } else { buff1 },
                    PffftDirection::Backward,
                );
                vinput = if ib == 0 { buff0 } else { buff1 };
                ib ^= 1;
            }
            if setup.transform == PffftTransform::Real {
                pffft_real_preprocess(
                    ncvec,
                    vinput,
                    if ib == 0 { buff0 } else { buff1 },
                    setup.data.as_ptr().add(setup.e_offset),
                );
                let r = rfftb1_ps(
                    ncvec * 2,
                    if ib == 0 { buff0 } else { buff1 },
                    buff0,
                    buff1,
                    setup.data.as_ptr().add(setup.twiddle_offset),
                    setup.ifac.as_ptr(),
                );
                ib = if r == buff0 { 0 } else { 1 };
            } else {
                pffft_cplx_preprocess(
                    ncvec,
                    vinput,
                    if ib == 0 { buff0 } else { buff1 },
                    setup.data.as_ptr().add(setup.e_offset),
                );
                let r = cfftf1_ps(
                    ncvec,
                    if ib == 0 { buff0 } else { buff1 },
                    buff0,
                    buff1,
                    setup.data.as_ptr().add(setup.twiddle_offset),
                    setup.ifac.as_ptr(),
                    1.0,
                );
                ib = if r == buff0 { 0 } else { 1 };
                let cur = if ib == 0 { buff0 } else { buff1 };
                for k in 0..ncvec {
                    let (a, b) = v4_interleave2(v4_load(cur, k * 2), v4_load(cur, k * 2 + 1));
                    v4_store(cur, k * 2, a);
                    v4_store(cur, k * 2 + 1, b);
                }
            }
        }

        let cur = if ib == 0 { buff0 } else { buff1 };
        if cur != foutput {
            for k in 0..ncvec {
                v4_store(foutput, 2 * k, v4_load(cur, 2 * k));
                v4_store(foutput, 2 * k + 1, v4_load(cur, 2 * k + 1));
            }
        }
    }
}

#[inline(never)]
pub unsafe fn pffft_zconvolve_accumulate(
    setup: &PFFFTSetup,
    a: *const f32,
    b: *const f32,
    ab: *mut f32,
    scaling: f32,
) {
    let ncvec = setup.ncvec;
    let vscal = v4_splat(scaling);

    let ar0 = unsafe { *a.add(0) };
    let ai0 = unsafe { *a.add(4) };
    let br0 = unsafe { *b.add(0) };
    let bi0 = unsafe { *b.add(4) };
    let abr0 = unsafe { *ab.add(0) };
    let abi0 = unsafe { *ab.add(4) };

    for i in (0..ncvec).step_by(2) {
        unsafe {
            let mut ar = v4_load(a, 2 * i);
            let mut ai = v4_load(a, 2 * i + 1);
            let br = v4_load(b, 2 * i);
            let bi = v4_load(b, 2 * i + 1);
            (ar, ai) = v4_cplx_mul(ar, ai, br, bi);
            v4_store(ab, 2 * i, v4_add(v4_mul(ar, vscal), v4_load(ab, 2 * i)));
            v4_store(ab, 2 * i + 1, v4_add(v4_mul(ai, vscal), v4_load(ab, 2 * i + 1)));

            let mut ar2 = v4_load(a, 2 * i + 2);
            let mut ai2 = v4_load(a, 2 * i + 3);
            let br2 = v4_load(b, 2 * i + 2);
            let bi2 = v4_load(b, 2 * i + 3);
            (ar2, ai2) = v4_cplx_mul(ar2, ai2, br2, bi2);
            v4_store(ab, 2 * i + 2, v4_add(v4_mul(ar2, vscal), v4_load(ab, 2 * i + 2)));
            v4_store(ab, 2 * i + 3, v4_add(v4_mul(ai2, vscal), v4_load(ab, 2 * i + 3)));
        }
    }

    if setup.transform == PffftTransform::Real {
        unsafe {
            *ab.add(0) = abr0 + ar0 * br0 * scaling;
            *ab.add(4) = abi0 + ai0 * bi0 * scaling;
        }
    }
}

#[inline(never)]
pub unsafe fn pffft_transform(
    setup: &PFFFTSetup,
    input: *const f32,
    output: *mut f32,
    work: *mut f32,
    direction: PffftDirection,
) {
    unsafe {
        pffft_transform_internal(setup, input, output, work, direction, false);
    }
}

#[inline(never)]
pub unsafe fn pffft_transform_ordered(
    setup: &PFFFTSetup,
    input: *const f32,
    output: *mut f32,
    work: *mut f32,
    direction: PffftDirection,
) {
    unsafe {
        pffft_transform_internal(setup, input, output, work, direction, true);
    }
}

pub unsafe fn pffft_zreorder(setup: &PFFFTSetup, input: *const f32, output: *mut f32, direction: PffftDirection) {
    unsafe {
        assert!(input != output as *const f32);
        let n = setup.n;
        let ncvec = setup.ncvec;

        match setup.transform {
            PffftTransform::Real => {
                let dk = n / 32;

                if direction == PffftDirection::Forward {
                    for k in 0..dk {
                        let in0 = v4_load(input, k * 8);
                        let in1 = v4_load(input, k * 8 + 1);
                        let (o0, o1) = v4_interleave2(in0, in1);
                        v4_store(output, 2 * k, o0);
                        v4_store(output, 2 * k + 1, o1);

                        let in4 = v4_load(input, k * 8 + 4);
                        let in5 = v4_load(input, k * 8 + 5);
                        let (o4, o5) = v4_interleave2(in4, in5);
                        v4_store(output, 2 * (2 * dk + k), o4);
                        v4_store(output, 2 * (2 * dk + k) + 1, o5);
                    }

                    reversed_copy(dk, input.add(2 * SIMD_SZ), 8, output.add(n / 2));
                    reversed_copy(dk, input.add(6 * SIMD_SZ), 8, output.add(n));
                } else {
                    for k in 0..dk {
                        let in0 = v4_load(input, 2 * k);
                        let in1 = v4_load(input, 2 * k + 1);
                        let (o0, o1) = v4_uninterleave2(in0, in1);
                        v4_store(output, k * 8, o0);
                        v4_store(output, k * 8 + 1, o1);

                        let in4 = v4_load(input, 2 * (2 * dk + k));
                        let in5 = v4_load(input, 2 * (2 * dk + k) + 1);
                        let (o4, o5) = v4_uninterleave2(in4, in5);
                        v4_store(output, k * 8 + 4, o4);
                        v4_store(output, k * 8 + 5, o5);
                    }

                    unreversed_copy(dk, input.add(n / 4), output.add(n - 6 * SIMD_SZ), -8);
                    unreversed_copy(dk, input.add(3 * n / 4), output.add(n - 2 * SIMD_SZ), -8);
                }
            }
            PffftTransform::Complex => {
                if direction == PffftDirection::Forward {
                    for k in 0..ncvec {
                        let kk = (k / 4) + (k % 4) * (ncvec / 4);
                        let inr = v4_load(input, k * 2);
                        let ini = v4_load(input, k * 2 + 1);
                        let (outr, outi) = v4_interleave2(inr, ini);
                        v4_store(output, kk * 2, outr);
                        v4_store(output, kk * 2 + 1, outi);
                    }
                } else {
                    for k in 0..ncvec {
                        let kk = (k / 4) + (k % 4) * (ncvec / 4);
                        let inr = v4_load(input, kk * 2);
                        let ini = v4_load(input, kk * 2 + 1);
                        let (outr, outi) = v4_uninterleave2(inr, ini);
                        v4_store(output, k * 2, outr);
                        v4_store(output, k * 2 + 1, outi);
                    }
                }
            }
        }
    }
}

pub(crate) unsafe fn decompose(n: usize, ifac: *mut usize, ntryh: *const usize) -> usize {
    unsafe {
        let mut nl = n;
        let mut nf: usize = 0;
        let mut j: usize = 0;
        while *ntryh.add(j) != 0 {
            let ntry = *ntryh.add(j);
            while nl != 1 {
                let nq = nl / ntry;
                let nr = nl - ntry * nq;
                if nr == 0 {
                    assert!(2 + nf < IFAC_MAX_SIZE);
                    *ifac.add(2 + nf) = ntry;
                    nf += 1;
                    nl = nq;
                    if ntry == 2 && nf != 1 {
                        for i in 2..=nf {
                            let ib = nf - i + 2;
                            *ifac.add(ib + 1) = *ifac.add(ib);
                        }
                        *ifac.add(2) = 2;
                    }
                } else {
                    break;
                }
            }
            j += 1;
        }
        *ifac.add(0) = n;
        *ifac.add(1) = nf;
        nf
    }
}

pub(crate) unsafe fn rffti1_ps(n: usize, wa: *mut f32, ifac: *mut usize) {
    unsafe {
        static NTRYH: [usize; 5] = [4, 2, 3, 5, 0];
        let nf = decompose(n, ifac, NTRYH.as_ptr());
        let argh = (2.0 * std::f32::consts::PI) / n as f32;
        let mut is = 0;
        let nfm1 = nf - 1;
        let mut l1 = 1;
        for k1 in 1..=nfm1 {
            let ip = *ifac.add(k1 + 1);
            let mut ld = 0;
            let l2 = l1 * ip;
            let ido = n / l2;
            let ipm = ip - 1;
            for _ in 1..=ipm {
                let mut i = is;
                let mut fi = 0;
                ld += l1;
                let argld = (ld as f32) * argh;
                for _ in (3..=ido).step_by(2) {
                    i += 2;
                    fi += 1;
                    *wa.add(i - 2) = (fi as f32 * argld).cos();
                    *wa.add(i - 1) = (fi as f32 * argld).sin();
                }
                is += ido;
            }
            l1 = l2;
        }
    }
}

pub(crate) unsafe fn cffti1_ps(n: usize, wa: *mut f32, ifac: *mut usize) {
    unsafe {
        static NTRYH: [usize; 5] = [5, 3, 4, 2, 0];
        let nf = decompose(n, ifac, NTRYH.as_ptr());
        let argh = (2.0 * std::f32::consts::PI) / n as f32;
        let mut i = 1;
        let mut l1 = 1;
        for k1 in 1..=nf {
            let ip = *ifac.add(k1 + 1);
            let mut ld = 0;
            let l2 = l1 * ip;
            let ido = n / l2;
            let idot = ido + ido + 2;
            let ipm = ip - 1;
            for _ in 1..=ipm {
                let i1 = i;
                let mut fi = 0;
                *wa.add(i - 1) = 1.0;
                *wa.add(i) = 0.0;
                ld += l1;
                let argld = (ld as f32) * argh;
                for _ in (4..=idot).step_by(2) {
                    i += 2;
                    fi += 1;
                    *wa.add(i - 1) = (fi as f32 * argld).cos();
                    *wa.add(i) = (fi as f32 * argld).sin();
                }
                if ip > 5 {
                    *wa.add(i1 - 1) = *wa.add(i - 1);
                    *wa.add(i1) = *wa.add(i);
                }
            }
            l1 = l2;
        }
    }
}

unsafe fn cfftf1_ps(
    n: usize,
    input_readonly: *const f32,
    work1: *mut f32,
    work2: *mut f32,
    wa: *const f32,
    ifac: *const usize,
    isign: f32,
) -> *mut f32 {
    unsafe {
        let mut in_ptr = input_readonly as *mut f32;
        let mut out_ptr = if in_ptr == work2 { work1 } else { work2 };
        let nf = *ifac.add(1);
        let mut l1 = 1;
        let mut iw: isize = 0;
        assert!(in_ptr != out_ptr && work1 != work2);
        for k1 in 2..=nf + 1 {
            let ip = *ifac.add(k1);
            let l2 = ip * l1;
            let ido = n / l2;
            let idot = ido + ido;
            match ip {
                5 => {
                    let ix2 = iw + idot as isize;
                    let ix3 = ix2 + idot as isize;
                    let ix4 = ix3 + idot as isize;
                    passf5_ps(
                        idot,
                        l1,
                        in_ptr,
                        out_ptr,
                        wa.offset(iw),
                        wa.offset(ix2),
                        wa.offset(ix3),
                        wa.offset(ix4),
                        isign,
                    );
                }
                4 => {
                    let ix2 = iw + idot as isize;
                    let ix3 = ix2 + idot as isize;
                    passf4_ps(idot, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2), wa.offset(ix3), isign);
                }
                2 => {
                    passf2_ps(idot, l1, in_ptr, out_ptr, wa.offset(iw), isign);
                }
                3 => {
                    let ix2 = iw + idot as isize;
                    passf3_ps(idot, l1, in_ptr, out_ptr, wa.offset(iw), wa.offset(ix2), isign);
                }
                _ => {
                    unreachable!();
                }
            }
            l1 = l2;
            iw += (ip as isize - 1) * idot as isize;
            if out_ptr == work2 {
                out_ptr = work1;
                in_ptr = work2;
            } else {
                out_ptr = work2;
                in_ptr = work1;
            }
        }
        in_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[cfg(feature = "pffft")]
    use std::ffi::c_int;
    #[cfg(feature = "pffft")]
    use std::os::raw::c_void;

    #[cfg(feature = "pffft")]
    #[repr(C)]
    struct CPffftSetup {
        _private: [u8; 0],
    }

    #[cfg(feature = "pffft")]
    #[repr(C)]
    #[derive(Copy, Clone)]
    enum CPffftDirection {
        Forward = 0,
        Backward = 1,
    }

    #[cfg(feature = "pffft")]
    #[repr(C)]
    #[derive(Copy, Clone)]
    enum CPffftTransform {
        Real = 0,
        Complex = 1,
    }

    #[cfg(feature = "pffft")]
    #[link(name = "pffft", kind = "static")]
    unsafe extern "C" {
        fn pffft_new_setup(n: c_int, transform: CPffftTransform) -> *mut CPffftSetup;
        fn pffft_destroy_setup(setup: *mut CPffftSetup);
        fn pffft_zreorder(setup: *mut CPffftSetup, input: *const f32, output: *mut f32, direction: CPffftDirection);
        fn pffft_transform_ordered(
            setup: *mut CPffftSetup,
            input: *const f32,
            output: *mut f32,
            work: *mut f32,
            direction: CPffftDirection,
        );
        fn pffft_zconvolve_accumulate(
            setup: *mut CPffftSetup,
            a: *const f32,
            b: *const f32,
            ab: *mut f32,
            scaling: f32,
        );
        fn pffft_aligned_malloc(nb_bytes: usize) -> *mut c_void;
        fn pffft_aligned_free(ptr: *mut c_void);
    }

    #[cfg(feature = "pffft")]
    struct CAlignedBuffer {
        ptr: *mut f32,
        len: usize,
    }

    #[cfg(feature = "pffft")]
    impl CAlignedBuffer {
        fn new(len: usize) -> Option<Self> {
            let bytes = len * std::mem::size_of::<f32>();
            let ptr = unsafe { pffft_aligned_malloc(bytes) as *mut f32 };
            if ptr.is_null() { None } else { Some(Self { ptr, len }) }
        }

        fn as_slice_mut(&mut self) -> &mut [f32] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }

        fn as_slice(&self) -> &[f32] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    #[cfg(feature = "pffft")]
    impl Drop for CAlignedBuffer {
        fn drop(&mut self) {
            unsafe { pffft_aligned_free(self.ptr as *mut c_void) };
        }
    }

    #[cfg(feature = "pffft")]
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[cfg(feature = "pffft")]
    fn fill_test_signal(buf: &mut [f32]) {
        for (i, v) in buf.iter_mut().enumerate() {
            let t = i as f32;
            *v = (0.013 * t).sin() * 0.7 + (0.031 * t).cos() * 0.2 + ((i % 17) as f32 - 8.0) * 0.001;
        }
    }

    #[cfg(feature = "pffft")]
    fn compare_zreorder_real(n: usize, direction: PffftDirection) {
        let rs_setup = PFFFTSetup::new(n, PffftTransform::Real).expect("Rust setup failed");
        let c_setup = unsafe { pffft_new_setup(n as c_int, CPffftTransform::Real) };
        assert!(!c_setup.is_null(), "C setup failed");

        let len = n;
        let mut input = CAlignedBuffer::new(len).expect("aligned input allocation failed");
        let mut c_output = CAlignedBuffer::new(len).expect("aligned output allocation failed");
        let mut rs_output = vec![0.0f32; len];

        fill_test_signal(input.as_slice_mut());

        unsafe {
            pffft_zreorder(
                c_setup,
                input.as_slice().as_ptr(),
                c_output.as_slice_mut().as_mut_ptr(),
                match direction {
                    PffftDirection::Forward => CPffftDirection::Forward,
                    PffftDirection::Backward => CPffftDirection::Backward,
                },
            );
        }

        rs_setup.zreorder(input.as_slice(), &mut rs_output, direction);

        let diff = max_abs_diff(c_output.as_slice(), &rs_output);
        assert!(
            diff < 1e-6,
            "real zreorder mismatch n={}, direction={:?}, max_abs_diff={}",
            n,
            direction,
            diff
        );

        unsafe { pffft_destroy_setup(c_setup) };
    }

    #[cfg(feature = "pffft")]
    fn compare_zreorder_complex(n: usize, direction: PffftDirection) {
        let rs_setup = PFFFTSetup::new(n, PffftTransform::Complex).expect("Rust setup failed");
        let c_setup = unsafe { pffft_new_setup(n as c_int, CPffftTransform::Complex) };
        assert!(!c_setup.is_null(), "C setup failed");

        let len = 2 * n;
        let mut input = CAlignedBuffer::new(len).expect("aligned input allocation failed");
        let mut c_output = CAlignedBuffer::new(len).expect("aligned output allocation failed");
        let mut rs_output = vec![0.0f32; len];

        fill_test_signal(input.as_slice_mut());

        unsafe {
            pffft_zreorder(
                c_setup,
                input.as_slice().as_ptr(),
                c_output.as_slice_mut().as_mut_ptr(),
                match direction {
                    PffftDirection::Forward => CPffftDirection::Forward,
                    PffftDirection::Backward => CPffftDirection::Backward,
                },
            );
        }

        rs_setup.zreorder(input.as_slice(), &mut rs_output, direction);

        let diff = max_abs_diff(c_output.as_slice(), &rs_output);
        assert!(
            diff < 1e-6,
            "complex zreorder mismatch n={}, direction={:?}, max_abs_diff={}",
            n,
            direction,
            diff
        );

        unsafe { pffft_destroy_setup(c_setup) };
    }

    #[test]
    fn test_new_real_setup() {
        // Real transform requires N multiple of 32
        let n = 32;
        let setup = PFFFTSetup::new(n, PffftTransform::Real);
        assert!(setup.is_some());
        let setup = setup.unwrap();
        assert_eq!(setup.n(), n as usize);
        assert_eq!(setup.transform(), PffftTransform::Real);
        // Check ifac array has valid decomposition
        let ifac = &setup.ifac;
        assert!(ifac[0] > 0); // n
        assert!(ifac[1] > 0); // number of factors
    }

    #[test]
    fn test_new_complex_setup() {
        // Complex transform requires N multiple of 16
        let n = 16;
        let setup = PFFFTSetup::new(n, PffftTransform::Complex);
        assert!(setup.is_some());
        let setup = setup.unwrap();
        assert_eq!(setup.n(), n as usize);
        assert_eq!(setup.transform(), PffftTransform::Complex);
    }

    #[test]
    fn test_invalid_n() {
        // N must be positive
        assert!(PFFFTSetup::new(0, PffftTransform::Real).is_none());
        // N too large (beyond 1<<26)
        assert!(PFFFTSetup::new(1 << 27, PffftTransform::Real).is_none());
        // Real transform requires N multiple of 32
        assert!(PFFFTSetup::new(31, PffftTransform::Real).is_none());
        assert!(PFFFTSetup::new(33, PffftTransform::Real).is_none());
        // Complex transform requires N multiple of 16
        assert!(PFFFTSetup::new(15, PffftTransform::Complex).is_none());
        assert!(PFFFTSetup::new(17, PffftTransform::Complex).is_none());
    }

    // ========================================================================
    // Standalone round-trip and correctness tests (no C library required)
    // ========================================================================

    const REAL_SIZES: [usize; 6] = [32, 64, 128, 256, 512, 1024];
    const COMPLEX_SIZES: [usize; 6] = [16, 32, 64, 128, 256, 512];

    fn generate_test_signal(n: usize, seed: u32) -> Vec<f32> {
        // Deterministic pseudo-random signal using a simple LCG
        let mut state: u64 = seed as u64 ^ 0xDEADBEEF;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = ((state >> 33) as u32) as f32 / u32::MAX as f32;
                bits * 10.0 - 5.0
            })
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, context: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", context, a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff < tol,
                "{}: mismatch at index {}: {} vs {}, diff={} (tol={})",
                context,
                i,
                x,
                y,
                diff,
                tol
            );
        }
    }

    // --- Real transform ordered round-trip ---

    #[test]
    fn test_real_ordered_roundtrip() {
        for &n in &REAL_SIZES {
            let input = generate_test_signal(n, n as u32);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; n];
            setup.transform_ordered(&spectrum, &mut output, None, PffftDirection::Backward);

            // PFFFT backward is unnormalized: output = n * original
            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-4, &format!("real ordered roundtrip n={}", n));
        }
    }

    // --- Complex transform ordered round-trip ---

    #[test]
    fn test_complex_ordered_roundtrip() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 1000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; len];
            setup.transform_ordered(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-4, &format!("complex ordered roundtrip n={}", n));
        }
    }

    // --- Packed transform round-trip ---

    #[test]
    fn test_real_packed_roundtrip() {
        for &n in &REAL_SIZES {
            let input = generate_test_signal(n, n as u32 + 2000);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_packed(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; n];
            setup.transform_packed(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-4, &format!("real packed roundtrip n={}", n));
        }
    }

    #[test]
    fn test_complex_packed_roundtrip() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 3000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; len];
            setup.transform_packed(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; len];
            setup.transform_packed(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-4, &format!("complex packed roundtrip n={}", n));
        }
    }

    // --- zreorder round-trip ---

    #[test]
    fn test_zreorder_real_roundtrip() {
        for &n in &REAL_SIZES {
            let input = generate_test_signal(n, n as u32 + 4000);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut reordered = vec![0.0f32; n];
            setup.zreorder(&input, &mut reordered, PffftDirection::Forward);

            let mut recovered = vec![0.0f32; n];
            setup.zreorder(&reordered, &mut recovered, PffftDirection::Backward);

            assert_close(&input, &recovered, 1e-6, &format!("real zreorder roundtrip n={}", n));
        }
    }

    #[test]
    fn test_zreorder_complex_roundtrip() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 5000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut reordered = vec![0.0f32; len];
            setup.zreorder(&input, &mut reordered, PffftDirection::Forward);

            let mut recovered = vec![0.0f32; len];
            setup.zreorder(&reordered, &mut recovered, PffftDirection::Backward);

            assert_close(&input, &recovered, 1e-6, &format!("complex zreorder roundtrip n={}", n));
        }
    }

    // --- Ordered vs packed+zreorder equivalence ---

    #[test]
    fn test_real_ordered_equals_packed_then_zreorder() {
        for &n in &REAL_SIZES {
            let input = generate_test_signal(n, n as u32 + 6000);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut ordered = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut ordered, None, PffftDirection::Forward);

            let mut packed = vec![0.0f32; n];
            setup.transform_packed(&input, &mut packed, None, PffftDirection::Forward);
            let mut reordered = vec![0.0f32; n];
            setup.zreorder(&packed, &mut reordered, PffftDirection::Forward);

            assert_close(
                &ordered,
                &reordered,
                1e-5,
                &format!("real ordered vs packed+zreorder n={}", n),
            );
        }
    }

    #[test]
    fn test_complex_ordered_equals_packed_then_zreorder() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 7000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut ordered = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut ordered, None, PffftDirection::Forward);

            let mut packed = vec![0.0f32; len];
            setup.transform_packed(&input, &mut packed, None, PffftDirection::Forward);
            let mut reordered = vec![0.0f32; len];
            setup.zreorder(&packed, &mut reordered, PffftDirection::Forward);

            assert_close(
                &ordered,
                &reordered,
                1e-5,
                &format!("complex ordered vs packed+zreorder n={}", n),
            );
        }
    }

    // --- DC signal: constant input should concentrate energy at bin 0 ---

    #[test]
    fn test_real_dc_signal() {
        for &n in &REAL_SIZES {
            let dc_val = 3.5f32;
            let input = vec![dc_val; n];
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // Ordered real layout: [DC_re, Nyquist_re, bin1_re, bin1_im, bin2_re, bin2_im, ...]
            let dc_re = spectrum[0];
            let expected_dc = dc_val * n as f32;
            assert!(
                (dc_re - expected_dc).abs() < 1e-3,
                "real DC n={}: expected DC={}, got {}",
                n,
                expected_dc,
                dc_re
            );

            // Nyquist should be zero for a constant signal (even n)
            let nyquist_re = spectrum[1];
            assert!(
                nyquist_re.abs() < 1e-3,
                "real DC n={}: Nyquist should be ~0, got {}",
                n,
                nyquist_re
            );

            // All other bins should be ~0
            for i in 1..n / 2 {
                let re = spectrum[2 * i];
                let im = spectrum[2 * i + 1];
                let mag = (re * re + im * im).sqrt();
                assert!(
                    mag < 1e-3,
                    "real DC n={}: bin {} magnitude should be ~0, got {} (re={}, im={})",
                    n,
                    i,
                    mag,
                    re,
                    im
                );
            }
        }
    }

    #[test]
    fn test_complex_dc_signal() {
        for &n in &COMPLEX_SIZES {
            let dc_val = 2.0f32;
            // Interleaved complex: all real parts = dc_val, all imag parts = 0
            let mut input = vec![0.0f32; 2 * n];
            for i in 0..n {
                input[2 * i] = dc_val;
            }
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; 2 * n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // Bin 0 should be (dc_val * n, 0)
            let bin0_re = spectrum[0];
            let bin0_im = spectrum[1];
            let expected_dc = dc_val * n as f32;
            assert!(
                (bin0_re - expected_dc).abs() < 1e-3,
                "complex DC n={}: expected bin0 re={}, got {}",
                n,
                expected_dc,
                bin0_re
            );
            assert!(
                bin0_im.abs() < 1e-3,
                "complex DC n={}: expected bin0 im=0, got {}",
                n,
                bin0_im
            );

            // All other bins should be ~0
            for k in 1..n {
                let re = spectrum[2 * k];
                let im = spectrum[2 * k + 1];
                let mag = (re * re + im * im).sqrt();
                assert!(
                    mag < 1e-3,
                    "complex DC n={}: bin {} should be ~0, got mag={} (re={}, im={})",
                    n,
                    k,
                    mag,
                    re,
                    im
                );
            }
        }
    }

    // --- Impulse: delta[0]=1 should produce flat spectrum ---

    #[test]
    fn test_real_impulse() {
        for &n in &REAL_SIZES {
            let mut input = vec![0.0f32; n];
            input[0] = 1.0;
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // DC bin should be 1.0
            assert!(
                (spectrum[0] - 1.0).abs() < 1e-5,
                "real impulse n={}: DC expected 1.0, got {}",
                n,
                spectrum[0]
            );
            // Nyquist bin should be 1.0
            assert!(
                (spectrum[1] - 1.0).abs() < 1e-5,
                "real impulse n={}: Nyquist expected 1.0, got {}",
                n,
                spectrum[1]
            );
            // All other bins: re=1, im=0
            for i in 1..n / 2 {
                let re = spectrum[2 * i];
                let im = spectrum[2 * i + 1];
                assert!(
                    (re - 1.0).abs() < 1e-5,
                    "real impulse n={}: bin {} re expected 1.0, got {}",
                    n,
                    i,
                    re
                );
                assert!(
                    im.abs() < 1e-5,
                    "real impulse n={}: bin {} im expected 0.0, got {}",
                    n,
                    i,
                    im
                );
            }
        }
    }

    #[test]
    fn test_complex_impulse() {
        for &n in &COMPLEX_SIZES {
            let mut input = vec![0.0f32; 2 * n];
            input[0] = 1.0; // real part of sample 0
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; 2 * n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // Every bin should be (1, 0)
            for k in 0..n {
                let re = spectrum[2 * k];
                let im = spectrum[2 * k + 1];
                assert!(
                    (re - 1.0).abs() < 1e-5,
                    "complex impulse n={}: bin {} re expected 1.0, got {}",
                    n,
                    k,
                    re
                );
                assert!(
                    im.abs() < 1e-5,
                    "complex impulse n={}: bin {} im expected 0.0, got {}",
                    n,
                    k,
                    im
                );
            }
        }
    }

    // --- Pure sinusoid: energy should appear at the correct bin ---

    #[test]
    fn test_real_pure_sinusoid() {
        for &n in &REAL_SIZES {
            let bin_index = 3usize;
            let input: Vec<f32> = (0..n)
                .map(|i| (2.0 * PI * bin_index as f32 * i as f32 / n as f32).cos())
                .collect();
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // For cos at bin k: spectrum[k] = n/2 (real part), all others ~0
            // DC and Nyquist are real-only
            let expected_mag = n as f32 / 2.0;
            for i in 0..n / 2 {
                let mag = if i == 0 {
                    spectrum[0].abs()
                } else if i == n / 2 {
                    spectrum[1].abs()
                } else {
                    let re = spectrum[2 * i];
                    let im = spectrum[2 * i + 1];
                    (re * re + im * im).sqrt()
                };

                if i == bin_index {
                    assert!(
                        (mag - expected_mag).abs() < 1e-2,
                        "real sinusoid n={}: bin {} mag expected {}, got {}",
                        n,
                        i,
                        expected_mag,
                        mag
                    );
                } else {
                    assert!(
                        mag < 1e-2,
                        "real sinusoid n={}: bin {} should be ~0, got {}",
                        n,
                        i,
                        mag
                    );
                }
            }
        }
    }

    #[test]
    fn test_complex_pure_sinusoid() {
        for &n in &COMPLEX_SIZES {
            let bin_index = 2usize;
            // Complex exponential e^(j*2*pi*k*i/n) puts all energy in bin k
            let mut input = vec![0.0f32; 2 * n];
            for i in 0..n {
                let angle = 2.0 * PI * bin_index as f32 * i as f32 / n as f32;
                input[2 * i] = angle.cos();
                input[2 * i + 1] = angle.sin();
            }
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; 2 * n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            for k in 0..n {
                let re = spectrum[2 * k];
                let im = spectrum[2 * k + 1];
                let mag = (re * re + im * im).sqrt();
                if k == bin_index {
                    // Should be ~n (re ~= n, im ~= 0 for positive frequency exponential)
                    assert!(
                        (mag - n as f32).abs() < 1e-2,
                        "complex sinusoid n={}: bin {} mag expected {}, got {}",
                        n,
                        k,
                        n,
                        mag
                    );
                } else {
                    assert!(
                        mag < 1e-2,
                        "complex sinusoid n={}: bin {} should be ~0, got {}",
                        n,
                        k,
                        mag
                    );
                }
            }
        }
    }

    // --- Parseval's theorem: energy in time domain == energy in frequency domain (scaled) ---

    #[test]
    fn test_real_parsevals_theorem() {
        for &n in &REAL_SIZES {
            let input = generate_test_signal(n, n as u32 + 8000);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let time_energy: f32 = input.iter().map(|x| x * x).sum();

            // Frequency energy for real FFT: DC^2 + Nyquist^2 + 2*sum(|bin_k|^2) for k=1..n/2-1
            // all divided by n
            let dc_energy = spectrum[0] * spectrum[0];
            let nyquist_energy = spectrum[1] * spectrum[1];
            let mut bin_energy = 0.0f32;
            for i in 1..n / 2 {
                let re = spectrum[2 * i];
                let im = spectrum[2 * i + 1];
                bin_energy += re * re + im * im;
            }
            let freq_energy = (dc_energy + nyquist_energy + 2.0 * bin_energy) / n as f32;

            let relative_err = (time_energy - freq_energy).abs() / time_energy.max(1e-10);
            assert!(
                relative_err < 1e-4,
                "real Parseval n={}: time_energy={}, freq_energy={}, rel_err={}",
                n,
                time_energy,
                freq_energy,
                relative_err
            );
        }
    }

    #[test]
    fn test_complex_parsevals_theorem() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 9000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // Time-domain energy (sum of |x[i]|^2 where x[i] is complex)
            let time_energy: f32 = input.chunks_exact(2).map(|c| c[0] * c[0] + c[1] * c[1]).sum();

            // Frequency-domain energy: sum(|X[k]|^2) / n
            let freq_energy: f32 =
                spectrum.chunks_exact(2).map(|c| c[0] * c[0] + c[1] * c[1]).sum::<f32>() / n as f32;

            let relative_err = (time_energy - freq_energy).abs() / time_energy.max(1e-10);
            assert!(
                relative_err < 1e-4,
                "complex Parseval n={}: time_energy={}, freq_energy={}, rel_err={}",
                n,
                time_energy,
                freq_energy,
                relative_err
            );
        }
    }

    // --- Linearity: FFT(a*x + b*y) == a*FFT(x) + b*FFT(y) ---

    #[test]
    fn test_real_linearity() {
        for &n in &REAL_SIZES {
            let x = generate_test_signal(n, n as u32 + 10000);
            let y = generate_test_signal(n, n as u32 + 11000);
            let a = 0.7f32;
            let b = -1.3f32;
            let combined: Vec<f32> = x.iter().zip(y.iter()).map(|(xi, yi)| a * xi + b * yi).collect();

            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut fft_x = vec![0.0f32; n];
            setup.transform_ordered(&x, &mut fft_x, None, PffftDirection::Forward);

            let mut fft_y = vec![0.0f32; n];
            setup.transform_ordered(&y, &mut fft_y, None, PffftDirection::Forward);

            let mut fft_combined = vec![0.0f32; n];
            setup.transform_ordered(&combined, &mut fft_combined, None, PffftDirection::Forward);

            let expected: Vec<f32> = fft_x.iter().zip(fft_y.iter()).map(|(fx, fy)| a * fx + b * fy).collect();

            assert_close(&fft_combined, &expected, 1e-3, &format!("real linearity n={}", n));
        }
    }

    #[test]
    fn test_complex_linearity() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let x = generate_test_signal(len, n as u32 + 12000);
            let y = generate_test_signal(len, n as u32 + 13000);
            let a = 0.5f32;
            let b = 2.1f32;
            let combined: Vec<f32> = x.iter().zip(y.iter()).map(|(xi, yi)| a * xi + b * yi).collect();

            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut fft_x = vec![0.0f32; len];
            setup.transform_ordered(&x, &mut fft_x, None, PffftDirection::Forward);

            let mut fft_y = vec![0.0f32; len];
            setup.transform_ordered(&y, &mut fft_y, None, PffftDirection::Forward);

            let mut fft_combined = vec![0.0f32; len];
            setup.transform_ordered(&combined, &mut fft_combined, None, PffftDirection::Forward);

            let expected: Vec<f32> = fft_x.iter().zip(fft_y.iter()).map(|(fx, fy)| a * fx + b * fy).collect();

            assert_close(&fft_combined, &expected, 1e-3, &format!("complex linearity n={}", n));
        }
    }

    // --- Shifted impulse: delta at position d should give e^(-j*2*pi*k*d/n) spectrum ---

    #[test]
    fn test_real_shifted_impulse() {
        for &n in &REAL_SIZES {
            let delay = 5usize;
            let mut input = vec![0.0f32; n];
            input[delay] = 1.0;
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            // All bins should have magnitude 1
            // DC (purely real)
            assert!(
                (spectrum[0].abs() - 1.0).abs() < 1e-4,
                "real shifted impulse n={}: DC magnitude expected 1, got {}",
                n,
                spectrum[0].abs()
            );
            // Nyquist (purely real)
            assert!(
                (spectrum[1].abs() - 1.0).abs() < 1e-4,
                "real shifted impulse n={}: Nyquist magnitude expected 1, got {}",
                n,
                spectrum[1].abs()
            );
            // Other bins
            for i in 1..n / 2 {
                let re = spectrum[2 * i];
                let im = spectrum[2 * i + 1];
                let mag = (re * re + im * im).sqrt();
                assert!(
                    (mag - 1.0).abs() < 1e-4,
                    "real shifted impulse n={}: bin {} magnitude expected 1, got {}",
                    n,
                    i,
                    mag
                );
                // Phase should match the delay: angle = -2*pi*k*delay/n
                let expected_re = (2.0 * PI * i as f32 * delay as f32 / n as f32).cos();
                let expected_im = -(2.0 * PI * i as f32 * delay as f32 / n as f32).sin();
                assert!(
                    (re - expected_re).abs() < 1e-4,
                    "real shifted impulse n={}: bin {} re expected {}, got {}",
                    n,
                    i,
                    expected_re,
                    re
                );
                assert!(
                    (im - expected_im).abs() < 1e-4,
                    "real shifted impulse n={}: bin {} im expected {}, got {}",
                    n,
                    i,
                    expected_im,
                    im
                );
            }
        }
    }

    // --- Convolution via multiply in frequency domain ---

    #[test]
    fn test_real_convolution_via_frequency_domain() {
        for &n in &REAL_SIZES {
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let x = generate_test_signal(n, n as u32 + 14000);
            let y = generate_test_signal(n, n as u32 + 15000);

            // Transform both to packed layout and use zconvolve_accumulate
            let mut fx = vec![0.0f32; n];
            setup.transform_packed(&x, &mut fx, None, PffftDirection::Forward);

            let mut fy = vec![0.0f32; n];
            setup.transform_packed(&y, &mut fy, None, PffftDirection::Forward);

            let mut product = vec![0.0f32; n];
            setup.zconvolve_accumulate(&fx, &fy, &mut product, 1.0);

            let mut time_result = vec![0.0f32; n];
            setup.transform_packed(&product, &mut time_result, None, PffftDirection::Backward);

            // Compute circular convolution directly
            let mut direct_conv = vec![0.0f32; n];
            for i in 0..n {
                let mut sum = 0.0f32;
                for j in 0..n {
                    let idx = (i + n - j) % n;
                    sum += x[j] * y[idx];
                }
                direct_conv[i] = sum;
            }

            // time_result should equal n * direct_conv (unnormalized FFT)
            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = time_result.iter().map(|v| v * scale).collect();
            assert_close(&scaled, &direct_conv, 1e-1, &format!("real convolution n={}", n));
        }
    }

    // --- zconvolve_accumulate actually accumulates ---

    #[test]
    fn test_zconvolve_accumulate_adds_to_existing() {
        for &n in &REAL_SIZES {
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();
            let sig = generate_test_signal(n, n as u32 + 16000);

            let mut packed = vec![0.0f32; n];
            setup.transform_packed(&sig, &mut packed, None, PffftDirection::Forward);

            // First call with zeroed accumulator
            let mut accum = vec![0.0f32; n];
            setup.zconvolve_accumulate(&packed, &packed, &mut accum, 1.0);
            let first = accum.clone();

            // Second call should add to existing accumulator
            setup.zconvolve_accumulate(&packed, &packed, &mut accum, 1.0);

            // Result should be 2x the first call
            let doubled: Vec<f32> = first.iter().map(|v| v * 2.0).collect();
            assert_close(&accum, &doubled, 1e-4, &format!("zconvolve accumulates n={}", n));
        }
    }

    // --- zconvolve_accumulate scaling parameter ---

    #[test]
    fn test_zconvolve_accumulate_scaling() {
        for &n in &REAL_SIZES {
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();
            let sig = generate_test_signal(n, n as u32 + 17000);

            let mut packed = vec![0.0f32; n];
            setup.transform_packed(&sig, &mut packed, None, PffftDirection::Forward);

            let mut result_scale1 = vec![0.0f32; n];
            setup.zconvolve_accumulate(&packed, &packed, &mut result_scale1, 1.0);

            let mut result_scale_half = vec![0.0f32; n];
            setup.zconvolve_accumulate(&packed, &packed, &mut result_scale_half, 0.5);

            let halved: Vec<f32> = result_scale1.iter().map(|v| v * 0.5).collect();
            assert_close(
                &result_scale_half,
                &halved,
                1e-4,
                &format!("zconvolve scaling n={}", n),
            );
        }
    }

    // --- Roundtrip with work buffer (non-null scratch) ---

    #[test]
    fn test_real_roundtrip_with_work_buffer() {
        for &n in &REAL_SIZES {
            let input = generate_test_signal(n, n as u32 + 18000);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut work = vec![0.0f32; n];
            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, Some(&mut work), PffftDirection::Forward);

            let mut output = vec![0.0f32; n];
            setup.transform_ordered(&spectrum, &mut output, Some(&mut work), PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-4, &format!("real roundtrip with work n={}", n));
        }
    }

    #[test]
    fn test_complex_roundtrip_with_work_buffer() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 19000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut work = vec![0.0f32; len];
            let mut spectrum = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut spectrum, Some(&mut work), PffftDirection::Forward);

            let mut output = vec![0.0f32; len];
            setup.transform_ordered(&spectrum, &mut output, Some(&mut work), PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-4, &format!("complex roundtrip with work n={}", n));
        }
    }

    // --- Larger FFT sizes ---

    #[test]
    fn test_real_roundtrip_large_sizes() {
        for n in [2048usize, 4096, 8192] {
            let input = generate_test_signal(n, n as u32 + 20000);
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; n];
            setup.transform_ordered(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-3, &format!("real roundtrip large n={}", n));
        }
    }

    #[test]
    fn test_complex_roundtrip_large_sizes() {
        for n in [1024usize, 2048, 4096] {
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 21000);
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; len];
            setup.transform_ordered(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-3, &format!("complex roundtrip large n={}", n));
        }
    }

    // --- Non-power-of-two composite sizes ---

    #[test]
    fn test_real_roundtrip_composite_sizes() {
        // PFFFT supports composite sizes that are multiples of 32 with factors 2,3,5
        for n in [96usize, 160, 192, 288, 320, 480, 960] {
            let setup = match PFFFTSetup::new(n, PffftTransform::Real) {
                Some(s) => s,
                None => continue,
            };
            let input = generate_test_signal(n, n as u32 + 22000);

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; n];
            setup.transform_ordered(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-3, &format!("real roundtrip composite n={}", n));
        }
    }

    #[test]
    fn test_complex_roundtrip_composite_sizes() {
        for n in [48usize, 80, 96, 144, 160, 240, 480] {
            let setup = match PFFFTSetup::new(n, PffftTransform::Complex) {
                Some(s) => s,
                None => continue,
            };
            let len = 2 * n;
            let input = generate_test_signal(len, n as u32 + 23000);

            let mut spectrum = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let mut output = vec![0.0f32; len];
            setup.transform_ordered(&spectrum, &mut output, None, PffftDirection::Backward);

            let scale = 1.0 / n as f32;
            let scaled: Vec<f32> = output.iter().map(|x| x * scale).collect();
            assert_close(&input, &scaled, 1e-3, &format!("complex roundtrip composite n={}", n));
        }
    }

    // --- Forward transform of all-zeros should produce all-zeros ---

    #[test]
    fn test_real_zero_input() {
        for &n in &REAL_SIZES {
            let input = vec![0.0f32; n];
            let setup = PFFFTSetup::new(n, PffftTransform::Real).unwrap();

            let mut spectrum = vec![0.0f32; n];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let zeros = vec![0.0f32; n];
            assert_close(&spectrum, &zeros, 1e-7, &format!("real zero input n={}", n));
        }
    }

    #[test]
    fn test_complex_zero_input() {
        for &n in &COMPLEX_SIZES {
            let len = 2 * n;
            let input = vec![0.0f32; len];
            let setup = PFFFTSetup::new(n, PffftTransform::Complex).unwrap();

            let mut spectrum = vec![0.0f32; len];
            setup.transform_ordered(&input, &mut spectrum, None, PffftDirection::Forward);

            let zeros = vec![0.0f32; len];
            assert_close(&spectrum, &zeros, 1e-7, &format!("complex zero input n={}", n));
        }
    }

    // ========================================================================
    // C library parity tests (require pffft feature)
    // ========================================================================

    #[cfg(feature = "pffft")]
    fn compare_transform_ordered_real(n: usize, direction: PffftDirection) {
        let rs_setup = PFFFTSetup::new(n, PffftTransform::Real).expect("Rust setup failed");
        let c_setup = unsafe { pffft_new_setup(n as c_int, CPffftTransform::Real) };
        assert!(!c_setup.is_null(), "C setup failed");

        let len = n;
        let mut input = CAlignedBuffer::new(len).expect("aligned input allocation failed");
        let mut c_output = CAlignedBuffer::new(len).expect("aligned output allocation failed");
        let mut rs_output = vec![0.0f32; len];

        fill_test_signal(input.as_slice_mut());

        let transform_input = if direction == PffftDirection::Backward {
            let mut spectrum = CAlignedBuffer::new(len).expect("aligned spectrum allocation failed");
            unsafe {
                pffft_transform_ordered(
                    c_setup,
                    input.as_slice().as_ptr(),
                    spectrum.as_slice_mut().as_mut_ptr(),
                    std::ptr::null_mut(),
                    CPffftDirection::Forward,
                );
            }
            spectrum
        } else {
            let mut passthrough = CAlignedBuffer::new(len).expect("aligned passthrough allocation failed");
            passthrough.as_slice_mut().copy_from_slice(input.as_slice());
            passthrough
        };

        unsafe {
            pffft_transform_ordered(
                c_setup,
                transform_input.as_slice().as_ptr(),
                c_output.as_slice_mut().as_mut_ptr(),
                std::ptr::null_mut(),
                match direction {
                    PffftDirection::Forward => CPffftDirection::Forward,
                    PffftDirection::Backward => CPffftDirection::Backward,
                },
            );
        }

        rs_setup.transform_ordered(transform_input.as_slice(), &mut rs_output, None, direction);

        let diff = max_abs_diff(c_output.as_slice(), &rs_output);
        assert!(
            diff < 2e-5,
            "real transform_ordered mismatch n={}, direction={:?}, max_abs_diff={}",
            n,
            direction,
            diff
        );

        unsafe { pffft_destroy_setup(c_setup) };
    }

    #[cfg(feature = "pffft")]
    fn compare_transform_ordered_complex(n: usize, direction: PffftDirection) {
        let rs_setup = PFFFTSetup::new(n, PffftTransform::Complex).expect("Rust setup failed");
        let c_setup = unsafe { pffft_new_setup(n as c_int, CPffftTransform::Complex) };
        assert!(!c_setup.is_null(), "C setup failed");

        let len = 2 * n;
        let mut input = CAlignedBuffer::new(len).expect("aligned input allocation failed");
        let mut c_output = CAlignedBuffer::new(len).expect("aligned output allocation failed");
        let mut rs_output = vec![0.0f32; len];

        fill_test_signal(input.as_slice_mut());

        unsafe {
            pffft_transform_ordered(
                c_setup,
                input.as_slice().as_ptr(),
                c_output.as_slice_mut().as_mut_ptr(),
                std::ptr::null_mut(),
                match direction {
                    PffftDirection::Forward => CPffftDirection::Forward,
                    PffftDirection::Backward => CPffftDirection::Backward,
                },
            );
        }

        rs_setup.transform_ordered(input.as_slice(), &mut rs_output, None, direction);

        let diff = max_abs_diff(c_output.as_slice(), &rs_output);
        assert!(
            diff < 1e-5,
            "complex transform_ordered mismatch n={}, direction={:?}, max_abs_diff={}",
            n,
            direction,
            diff
        );

        unsafe { pffft_destroy_setup(c_setup) };
    }

    #[cfg(feature = "pffft")]
    fn compare_zconvolve_accumulate_real(n: usize, scaling: f32) {
        let rs_setup = PFFFTSetup::new(n, PffftTransform::Real).expect("Rust setup failed");
        let c_setup = unsafe { pffft_new_setup(n as c_int, CPffftTransform::Real) };
        assert!(!c_setup.is_null(), "C setup failed");

        let len = n;
        let mut a = CAlignedBuffer::new(len).expect("aligned a allocation failed");
        let mut b = CAlignedBuffer::new(len).expect("aligned b allocation failed");
        let mut c_ab = CAlignedBuffer::new(len).expect("aligned ab allocation failed");
        let mut rs_ab = vec![0.0f32; len];

        fill_test_signal(a.as_slice_mut());
        fill_test_signal(b.as_slice_mut());
        fill_test_signal(c_ab.as_slice_mut());
        rs_ab.copy_from_slice(c_ab.as_slice());

        unsafe {
            pffft_zconvolve_accumulate(
                c_setup,
                a.as_slice().as_ptr(),
                b.as_slice().as_ptr(),
                c_ab.as_slice_mut().as_mut_ptr(),
                scaling,
            );
        }

        rs_setup.zconvolve_accumulate(a.as_slice(), b.as_slice(), &mut rs_ab, scaling);

        let diff = max_abs_diff(c_ab.as_slice(), &rs_ab);
        assert!(diff < 1e-5, "real zconvolve_accumulate mismatch n={}, max_abs_diff={}", n, diff);

        unsafe { pffft_destroy_setup(c_setup) };
    }

    #[cfg(feature = "pffft")]
    fn compare_zconvolve_accumulate_complex(n: usize, scaling: f32) {
        let rs_setup = PFFFTSetup::new(n, PffftTransform::Complex).expect("Rust setup failed");
        let c_setup = unsafe { pffft_new_setup(n as c_int, CPffftTransform::Complex) };
        assert!(!c_setup.is_null(), "C setup failed");

        let len = 2 * n;
        let mut a = CAlignedBuffer::new(len).expect("aligned a allocation failed");
        let mut b = CAlignedBuffer::new(len).expect("aligned b allocation failed");
        let mut c_ab = CAlignedBuffer::new(len).expect("aligned ab allocation failed");
        let mut rs_ab = vec![0.0f32; len];

        fill_test_signal(a.as_slice_mut());
        fill_test_signal(b.as_slice_mut());
        fill_test_signal(c_ab.as_slice_mut());
        rs_ab.copy_from_slice(c_ab.as_slice());

        unsafe {
            pffft_zconvolve_accumulate(
                c_setup,
                a.as_slice().as_ptr(),
                b.as_slice().as_ptr(),
                c_ab.as_slice_mut().as_mut_ptr(),
                scaling,
            );
        }

        rs_setup.zconvolve_accumulate(a.as_slice(), b.as_slice(), &mut rs_ab, scaling);

        let diff = max_abs_diff(c_ab.as_slice(), &rs_ab);
        assert!(diff < 1e-5, "complex zconvolve_accumulate mismatch n={}, max_abs_diff={}", n, diff);

        unsafe { pffft_destroy_setup(c_setup) };
    }

    #[cfg(feature = "pffft")]
    #[test]
    fn test_zreorder_real_parity_with_c() {
        for n in [32usize, 64, 128, 256] {
            compare_zreorder_real(n, PffftDirection::Forward);
            compare_zreorder_real(n, PffftDirection::Backward);
        }
    }

    #[cfg(feature = "pffft")]
    #[test]
    fn test_zreorder_complex_parity_with_c() {
        for n in [16usize, 32, 64, 128, 256] {
            compare_zreorder_complex(n, PffftDirection::Forward);
            compare_zreorder_complex(n, PffftDirection::Backward);
        }
    }

    #[cfg(feature = "pffft")]
    #[test]
    fn test_transform_ordered_real_parity_with_c() {
        for n in [32usize, 64, 128, 256] {
            compare_transform_ordered_real(n, PffftDirection::Forward);
            compare_transform_ordered_real(n, PffftDirection::Backward);
        }
    }

    #[cfg(feature = "pffft")]
    #[test]
    fn test_transform_ordered_complex_parity_with_c() {
        for n in [16usize, 32, 64, 128, 256] {
            compare_transform_ordered_complex(n, PffftDirection::Forward);
            compare_transform_ordered_complex(n, PffftDirection::Backward);
        }
    }

    #[cfg(feature = "pffft")]
    #[test]
    fn test_zconvolve_accumulate_real_parity_with_c() {
        for n in [32usize, 64, 128, 256] {
            compare_zconvolve_accumulate_real(n, 0.5);
            compare_zconvolve_accumulate_real(n, 1.0);
        }
    }

    #[cfg(feature = "pffft")]
    #[test]
    fn test_zconvolve_accumulate_complex_parity_with_c() {
        for n in [16usize, 32, 64, 128, 256] {
            compare_zconvolve_accumulate_complex(n, 0.5);
            compare_zconvolve_accumulate_complex(n, 1.0);
        }
    }
}
