mod vector4;

use std::f32::consts::PI;

use vector4::{
    v4_add, v4_cplx_mul, v4_cplx_mul_conj, v4_interleave2, v4_load, v4_mul, v4_scalar_mul, v4_splat, v4_store, v4_sub,
    v4_swaphl, v4_uninterleave2,
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
                let pc = cc.add((4 * k - 1) * 4);
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
}
