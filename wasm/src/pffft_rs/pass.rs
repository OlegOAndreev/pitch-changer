#[allow(dead_code)]
use super::vector4::{v4_add, v4_cplx_mul, v4_load, v4_mul, v4_scalar_mul, v4_splat, v4_store, v4_sub};

#[inline(never)]
pub unsafe fn passf2_ps(ido: usize, l1: usize, mut cc: *const f32, mut ch: *mut f32, wa1: *const f32, fsign: f32) {
    let l1ido = l1 * ido;

    unsafe {
        if ido <= 2 {
            for _ in 0..l1 {
                let cc0 = v4_load(cc, 0);
                let cc_ido0 = v4_load(cc, ido);
                let cc1 = v4_load(cc, 1);
                let cc_ido1 = v4_load(cc, ido + 1);

                // Store at ch[0], ch[l1ido], ch[1], ch[l1ido+1]
                v4_store(ch, 0, v4_add(cc0, cc_ido0));
                v4_store(ch, l1ido, v4_sub(cc0, cc_ido0));
                v4_store(ch, 1, v4_add(cc1, cc_ido1));
                v4_store(ch, l1ido + 1, v4_sub(cc1, cc_ido1));

                ch = ch.add(ido * 4);
                cc = cc.add(2 * ido * 4);
            }
        } else {
            for _ in 0..l1 {
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
pub unsafe fn passf3_ps(
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
        for _ in 0..l1 {
            for i in (0..(ido - 1)).step_by(2) {
                let tr2 = v4_add(v4_load(cc, i + ido), v4_load(cc, i + 2 * ido));
                let cr2 = v4_add(v4_load(cc, i), v4_scalar_mul(taur, tr2));
                v4_store(ch, i, v4_add(v4_load(cc, i), tr2));

                let ti2 = v4_add(v4_load(cc, i + ido + 1), v4_load(cc, i + 2 * ido + 1));
                let ci2 = v4_add(v4_load(cc, i + 1), v4_scalar_mul(taur, ti2));
                v4_store(ch, i + 1, v4_add(v4_load(cc, i + 1), ti2));

                let cr3 = v4_scalar_mul(taui, v4_sub(v4_load(cc, i + ido), v4_load(cc, i + 2 * ido)));
                let ci3 = v4_scalar_mul(taui, v4_sub(v4_load(cc, i + ido + 1), v4_load(cc, i + 2 * ido + 1)));

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
pub unsafe fn passf4_ps(
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
            for _ in 0..l1 {
                let tr1 = v4_sub(v4_load(cc, 0), v4_load(cc, 2 * ido));
                let tr2 = v4_add(v4_load(cc, 0), v4_load(cc, 2 * ido));
                let ti1 = v4_sub(v4_load(cc, 1), v4_load(cc, 2 * ido + 1));
                let ti2 = v4_add(v4_load(cc, 1), v4_load(cc, 2 * ido + 1));
                let ti4 = v4_scalar_mul(fsign, v4_sub(v4_load(cc, ido), v4_load(cc, 3 * ido)));
                let tr4 = v4_scalar_mul(fsign, v4_sub(v4_load(cc, 3 * ido + 1), v4_load(cc, ido + 1)));
                let tr3 = v4_add(v4_load(cc, ido), v4_load(cc, 3 * ido));
                let ti3 = v4_add(v4_load(cc, ido + 1), v4_load(cc, 3 * ido + 1));

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
            for _ in 0..l1 {
                for i in (0..(ido - 1)).step_by(2) {
                    let tr1 = v4_sub(v4_load(cc, i), v4_load(cc, i + 2 * ido));
                    let tr2 = v4_add(v4_load(cc, i), v4_load(cc, i + 2 * ido));
                    let ti1 = v4_sub(v4_load(cc, i + 1), v4_load(cc, i + 2 * ido + 1));
                    let ti2 = v4_add(v4_load(cc, i + 1), v4_load(cc, i + 2 * ido + 1));
                    let tr4 = v4_scalar_mul(fsign, v4_sub(v4_load(cc, i + 3 * ido + 1), v4_load(cc, i + ido + 1)));
                    let ti4 = v4_scalar_mul(fsign, v4_sub(v4_load(cc, i + ido), v4_load(cc, i + 3 * ido)));
                    let tr3 = v4_add(v4_load(cc, i + ido), v4_load(cc, i + 3 * ido));
                    let ti3 = v4_add(v4_load(cc, i + ido + 1), v4_load(cc, i + 3 * ido + 1));

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
pub unsafe fn passf5_ps(
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
        for _ in 0..l1 {
            for i in (0..(ido - 1)).step_by(2) {
                let ti5 = v4_sub(v4_load(cc, i + ido + 1), v4_load(cc, i + 4 * ido + 1));
                let ti2 = v4_add(v4_load(cc, i + ido + 1), v4_load(cc, i + 4 * ido + 1));
                let ti4 = v4_sub(v4_load(cc, i + 2 * ido + 1), v4_load(cc, i + 3 * ido + 1));
                let ti3 = v4_add(v4_load(cc, i + 2 * ido + 1), v4_load(cc, i + 3 * ido + 1));
                let tr5 = v4_sub(v4_load(cc, i + ido), v4_load(cc, i + 4 * ido));
                let tr2 = v4_add(v4_load(cc, i + ido), v4_load(cc, i + 4 * ido));
                let tr4 = v4_sub(v4_load(cc, i + 2 * ido), v4_load(cc, i + 3 * ido));
                let tr3 = v4_add(v4_load(cc, i + 2 * ido), v4_load(cc, i + 3 * ido));

                v4_store(ch, i, v4_add(v4_load(cc, i), v4_add(tr2, tr3)));
                v4_store(ch, i + 1, v4_add(v4_load(cc, i + 1), v4_add(ti2, ti3)));

                let cr2 = v4_add(v4_load(cc, i), v4_add(v4_scalar_mul(tr11, tr2), v4_scalar_mul(tr12, tr3)));
                let ci2 = v4_add(v4_load(cc, i + 1), v4_add(v4_scalar_mul(tr11, ti2), v4_scalar_mul(tr12, ti3)));
                let cr3 = v4_add(v4_load(cc, i), v4_add(v4_scalar_mul(tr12, tr2), v4_scalar_mul(tr11, tr3)));
                let ci3 = v4_add(v4_load(cc, i + 1), v4_add(v4_scalar_mul(tr12, ti2), v4_scalar_mul(tr11, ti3)));

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
