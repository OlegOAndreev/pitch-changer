#[allow(dead_code)]

// Common trait for 4-element SIMD float vectors. Each platform then defines a V4sf implementation (the platforms
// without SIMD use ScalarVector4). For better ergonomics, all V4sf functions are re-exported with prefix v4_.
pub(crate) trait Vector4: Copy {
    /// Zero vector
    fn zero() -> Self;

    /// Element-wise multiplication
    fn mul(a: Self, b: Self) -> Self;

    /// Element-wise addition
    fn add(a: Self, b: Self) -> Self;

    /// Fused multiply-add: a * b + c
    fn fma(a: Self, b: Self, c: Self) -> Self;

    /// Element-wise subtraction
    fn sub(a: Self, b: Self) -> Self;

    /// Broadcast scalar to all lanes
    fn splat(f: f32) -> Self;

    /// Interleave two vectors (zips them). Returns (interleaved_low, interleaved_high)
    fn interleave2(a: Self, b: Self) -> (Self, Self);

    /// Uninterleave two vectors (unzips them). Returns (even_elements, odd_elements)
    fn uninterleave2(a: Self, b: Self) -> (Self, Self);

    /// Transpose 4 vectors (matrix transpose). Returns (x0', x1', x2', x3')
    fn transpose(a: Self, b: Self, c: Self, d: Self) -> (Self, Self, Self, Self);

    /// Swap high and low halves: result[0..2] = b[0..2], result[2..4] = a[2..4]
    fn swaphl(a: Self, b: Self) -> Self;

    /// Unaligned load from memory
    unsafe fn load(ptr: *const f32) -> Self;

    /// Unaligned store to memory
    unsafe fn store(ptr: *mut f32, value: Self);

    /// Safe load from an unaligned slice
    #[inline(always)]
    fn load_from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= 4, "slice must have at least 4 elements for Vector4 load");
        unsafe { Self::load(slice.as_ptr()) }
    }

    /// Safe store to an unaligned slice
    #[inline(always)]
    fn store_to_slice(self, slice: &mut [f32]) {
        assert!(slice.len() >= 4, "slice must have at least 4 elements for Vector4 store");
        unsafe { Self::store(slice.as_mut_ptr(), self) }
    }

    fn scalar_mul(f: f32, v: Self) -> Self {
        Self::mul(Self::splat(f), v)
    }

    /// Complex multiplication: (ar + i*ai) * (br + i*bi) -> (ar', ai')
    #[inline(always)]
    fn cplx_mul(ar: Self, ai: Self, br: Self, bi: Self) -> (Self, Self) {
        let ar_result = Self::sub(Self::mul(ar, br), Self::mul(ai, bi));
        let ai_result = Self::add(Self::mul(ai, br), Self::mul(ar, bi));
        (ar_result, ai_result)
    }

    /// Complex multiplication with conjugate: (ar + i*ai) * (br - i*bi) -> (ar', ai')
    #[inline(always)]
    fn cplx_mul_conj(ar: Self, ai: Self, br: Self, bi: Self) -> (Self, Self) {
        let ar_result = Self::add(Self::mul(ar, br), Self::mul(ai, bi));
        let ai_result = Self::sub(Self::mul(ai, br), Self::mul(ar, bi));
        (ar_result, ai_result)
    }
}

// SSE2 implementation
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod sse2 {
    use super::*;
    use core::arch::x86_64::__m128;
    use core::arch::x86_64::{
        _MM_TRANSPOSE4_PS, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_shuffle_ps,
        _mm_storeu_ps, _mm_sub_ps, _mm_unpackhi_ps, _mm_unpacklo_ps,
    };

    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub(crate) struct SSE2Vector4(__m128);

    impl Vector4 for SSE2Vector4 {
        #[inline(always)]
        fn zero() -> Self {
            unsafe { SSE2Vector4(_mm_setzero_ps()) }
        }

        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            unsafe { SSE2Vector4(_mm_mul_ps(a.0, b.0)) }
        }

        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            unsafe { SSE2Vector4(_mm_add_ps(a.0, b.0)) }
        }

        #[inline(always)]
        fn fma(a: Self, b: Self, c: Self) -> Self {
            unsafe { SSE2Vector4(_mm_add_ps(_mm_mul_ps(a.0, b.0), c.0)) }
        }

        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            unsafe { SSE2Vector4(_mm_sub_ps(a.0, b.0)) }
        }

        #[inline(always)]
        fn splat(f: f32) -> Self {
            unsafe { SSE2Vector4(_mm_set1_ps(f)) }
        }

        #[inline(always)]
        fn interleave2(a: Self, b: Self) -> (Self, Self) {
            unsafe { (SSE2Vector4(_mm_unpacklo_ps(a.0, b.0)), SSE2Vector4(_mm_unpackhi_ps(a.0, b.0))) }
        }

        #[inline(always)]
        fn uninterleave2(a: Self, b: Self) -> (Self, Self) {
            unsafe {
                // _MM_SHUFFLE(2,0,2,0): result = [a0, a2, b0, b2]
                let shuf1 = _mm_shuffle_ps(a.0, b.0, 0b10001000);
                // _MM_SHUFFLE(3,1,3,1): result = [a1, a3, b1, b3]
                let shuf2 = _mm_shuffle_ps(a.0, b.0, 0b11011101);
                (SSE2Vector4(shuf1), SSE2Vector4(shuf2))
            }
        }

        #[inline(always)]
        fn transpose(mut x0: Self, mut x1: Self, mut x2: Self, mut x3: Self) -> (Self, Self, Self, Self) {
            unsafe {
                _MM_TRANSPOSE4_PS(&mut x0.0, &mut x1.0, &mut x2.0, &mut x3.0);
                (x0, x1, x2, x3)
            }
        }

        #[inline(always)]
        fn swaphl(a: Self, b: Self) -> Self {
            unsafe {
                // _MM_SHUFFLE(3,2,1,0): result = [b0, b1, a2, a3] (matches swaphl)
                SSE2Vector4(_mm_shuffle_ps(b.0, a.0, 0b11100100))
            }
        }

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            unsafe { SSE2Vector4(_mm_loadu_ps(ptr)) }
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, value: Self) {
            unsafe {
                _mm_storeu_ps(ptr, value.0);
            }
        }
    }
}

// NEON implementation
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::*;
    use core::arch::aarch64::float32x4_t;
    use core::arch::aarch64::{
        vaddq_f32, vcombine_f32, vdupq_n_f32, vget_high_f32, vget_low_f32, vld1q_dup_f32, vld1q_f32, vmlaq_f32,
        vmulq_f32, vst1q_f32, vsubq_f32, vuzpq_f32, vzipq_f32,
    };

    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub(crate) struct NEONVector4(float32x4_t);

    impl Vector4 for NEONVector4 {
        #[inline(always)]
        fn zero() -> Self {
            unsafe { NEONVector4(vdupq_n_f32(0.0)) }
        }

        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            unsafe { NEONVector4(vmulq_f32(a.0, b.0)) }
        }

        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            unsafe { NEONVector4(vaddq_f32(a.0, b.0)) }
        }

        #[inline(always)]
        fn fma(a: Self, b: Self, c: Self) -> Self {
            unsafe { NEONVector4(vmlaq_f32(c.0, a.0, b.0)) }
        }

        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            unsafe { NEONVector4(vsubq_f32(a.0, b.0)) }
        }

        #[inline(always)]
        fn splat(f: f32) -> Self {
            unsafe { NEONVector4(vld1q_dup_f32(&f)) }
        }

        #[inline(always)]
        fn interleave2(a: Self, b: Self) -> (Self, Self) {
            unsafe {
                let zipped = vzipq_f32(a.0, b.0);
                (NEONVector4(zipped.0), NEONVector4(zipped.1))
            }
        }

        #[inline(always)]
        fn uninterleave2(a: Self, b: Self) -> (Self, Self) {
            unsafe {
                let unzipped = vuzpq_f32(a.0, b.0);
                (NEONVector4(unzipped.0), NEONVector4(unzipped.1))
            }
        }

        #[inline(always)]
        fn transpose(a: Self, b: Self, c: Self, d: Self) -> (Self, Self, Self, Self) {
            unsafe {
                let t0 = vzipq_f32(a.0, c.0);
                let t1 = vzipq_f32(b.0, d.0);
                let u0 = vzipq_f32(t0.0, t1.0);
                let u1 = vzipq_f32(t0.1, t1.1);
                (NEONVector4(u0.0), NEONVector4(u0.1), NEONVector4(u1.0), NEONVector4(u1.1))
            }
        }

        #[inline(always)]
        fn swaphl(a: Self, b: Self) -> Self {
            unsafe {
                // Combine low half of b with high half of a
                NEONVector4(vcombine_f32(vget_low_f32(b.0), vget_high_f32(a.0)))
            }
        }

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            unsafe { NEONVector4(vld1q_f32(ptr)) }
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, value: Self) {
            unsafe { vst1q_f32(ptr, value.0) }
        }
    }
}

// WASM SIMD128 implementation
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use super::*;
    use core::arch::wasm32::v128;
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, f32x4_sub, i32x4_shuffle, v128_load, v128_store};

    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub(crate) struct WASMVector4(v128);

    impl Vector4 for WASMVector4 {
        #[inline(always)]
        fn zero() -> Self {
            WASMVector4(f32x4_splat(0.0))
        }

        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            WASMVector4(f32x4_mul(a.0, b.0))
        }

        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            WASMVector4(f32x4_add(a.0, b.0))
        }

        #[inline(always)]
        fn fma(a: Self, b: Self, c: Self) -> Self {
            WASMVector4(f32x4_add(f32x4_mul(a.0, b.0), c.0))
        }

        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            WASMVector4(f32x4_sub(a.0, b.0))
        }

        #[inline(always)]
        fn splat(f: f32) -> Self {
            WASMVector4(f32x4_splat(f))
        }

        #[inline(always)]
        fn interleave2(a: Self, b: Self) -> (Self, Self) {
            // low: [a0, b0, a1, b1]
            let low = i32x4_shuffle::<0, 4, 1, 5>(a.0, b.0);
            // high: [a2, b2, a3, b3]
            let high = i32x4_shuffle::<2, 6, 3, 7>(a.0, b.0);
            (WASMVector4(low), WASMVector4(high))
        }

        #[inline(always)]
        fn uninterleave2(a: Self, b: Self) -> (Self, Self) {
            // even: [a0, a2, b0, b2]
            let even = i32x4_shuffle::<0, 2, 4, 6>(a.0, b.0);
            // odd: [a1, a3, b1, b3]
            let odd = i32x4_shuffle::<1, 3, 5, 7>(a.0, b.0);
            (WASMVector4(even), WASMVector4(odd))
        }

        #[inline(always)]
        fn transpose(a: Self, b: Self, c: Self, d: Self) -> (Self, Self, Self, Self) {
            // Use interleave2 to simulate the NEON implementation
            let (t00, t01) = Self::interleave2(a, c); // t00 = [a0, c0, a1, c1], t01 = [a2, c2, a3, c3]
            let (t10, t11) = Self::interleave2(b, d); // t10 = [b0, d0, b1, d1], t11 = [b2, d2, b3, d3]
            let (col0, col1) = Self::interleave2(t00, t10); // col0 = [a0, b0, c0, d0], col1 = [a1, b1, c1, d1]
            let (col2, col3) = Self::interleave2(t01, t11); // col2 = [a2, b2, c2, d2], col3 = [a3, b3, c3, d3]
            (col0, col1, col2, col3)
        }

        #[inline(always)]
        fn swaphl(a: Self, b: Self) -> Self {
            // result: [b0, b1, a2, a3]
            WASMVector4(i32x4_shuffle::<4, 5, 2, 3>(a.0, b.0))
        }

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            unsafe { WASMVector4(v128_load(ptr as *const v128)) }
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, value: Self) {
            unsafe { v128_store(ptr as *mut v128, value.0) }
        }
    }
}

// Scalar fallback implementation (when SIMD is not available)
#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct ScalarVector4([f32; 4]);

impl Vector4 for ScalarVector4 {
    #[inline(always)]
    fn zero() -> Self {
        ScalarVector4([0.0; 4])
    }

    #[inline(always)]
    fn mul(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] * b.0[i];
        }
        ScalarVector4(result)
    }

    #[inline(always)]
    fn add(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] + b.0[i];
        }
        ScalarVector4(result)
    }

    #[inline(always)]
    fn fma(a: Self, b: Self, c: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] * b.0[i] + c.0[i];
        }
        ScalarVector4(result)
    }

    #[inline(always)]
    fn sub(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] - b.0[i];
        }
        ScalarVector4(result)
    }

    #[inline(always)]
    fn splat(f: f32) -> Self {
        ScalarVector4([f; 4])
    }

    #[inline(always)]
    fn interleave2(a: Self, b: Self) -> (Self, Self) {
        let mut low = [0.0; 4];
        let mut high = [0.0; 4];

        low[0] = a.0[0];
        low[1] = b.0[0];
        low[2] = a.0[1];
        low[3] = b.0[1];

        high[0] = a.0[2];
        high[1] = b.0[2];
        high[2] = a.0[3];
        high[3] = b.0[3];

        (ScalarVector4(low), ScalarVector4(high))
    }

    #[inline(always)]
    fn uninterleave2(a: Self, b: Self) -> (Self, Self) {
        let mut even = [0.0; 4];
        let mut odd = [0.0; 4];

        even[0] = a.0[0];
        even[1] = a.0[2];
        even[2] = b.0[0];
        even[3] = b.0[2];

        odd[0] = a.0[1];
        odd[1] = a.0[3];
        odd[2] = b.0[1];
        odd[3] = b.0[3];

        (ScalarVector4(even), ScalarVector4(odd))
    }

    #[inline(always)]
    fn transpose(a: Self, b: Self, c: Self, d: Self) -> (Self, Self, Self, Self) {
        let mut x0 = [0.0; 4];
        let mut x1 = [0.0; 4];
        let mut x2 = [0.0; 4];
        let mut x3 = [0.0; 4];

        x0[0] = a.0[0];
        x0[1] = b.0[0];
        x0[2] = c.0[0];
        x0[3] = d.0[0];

        x1[0] = a.0[1];
        x1[1] = b.0[1];
        x1[2] = c.0[1];
        x1[3] = d.0[1];

        x2[0] = a.0[2];
        x2[1] = b.0[2];
        x2[2] = c.0[2];
        x2[3] = d.0[2];

        x3[0] = a.0[3];
        x3[1] = b.0[3];
        x3[2] = c.0[3];
        x3[3] = d.0[3];

        (ScalarVector4(x0), ScalarVector4(x1), ScalarVector4(x2), ScalarVector4(x3))
    }

    #[inline(always)]
    fn swaphl(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        result[0] = b.0[0];
        result[1] = b.0[1];
        result[2] = a.0[2];
        result[3] = a.0[3];
        ScalarVector4(result)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        // Note: scalar load doesn't care about alignment
        let mut arr = [0.0; 4];
        unsafe { std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 4) }
        ScalarVector4(arr)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f32, value: Self) {
        unsafe { std::ptr::copy_nonoverlapping(value.0.as_ptr(), ptr, 4) }
    }
}

// Export the appropriate type based on target architecture
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(crate) use sse2::SSE2Vector4 as V4sf;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) use neon::NEONVector4 as V4sf;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm_simd::WASMVector4 as V4sf;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub(crate) use ScalarVector4 as V4sf;

// Re-export all functions in V4sf as v4_*

#[inline(always)]
pub(crate) fn v4_zero() -> V4sf {
    V4sf::zero()
}

#[inline(always)]
pub(crate) fn v4_mul(a: V4sf, b: V4sf) -> V4sf {
    V4sf::mul(a, b)
}

#[inline(always)]
pub(crate) fn v4_add(a: V4sf, b: V4sf) -> V4sf {
    V4sf::add(a, b)
}

#[inline(always)]
pub(crate) fn v4_fma(a: V4sf, b: V4sf, c: V4sf) -> V4sf {
    V4sf::fma(a, b, c)
}

#[inline(always)]
pub(crate) fn v4_sub(a: V4sf, b: V4sf) -> V4sf {
    V4sf::sub(a, b)
}

#[inline(always)]
pub(crate) fn v4_splat(f: f32) -> V4sf {
    V4sf::splat(f)
}

#[inline(always)]
pub(crate) fn v4_interleave2(a: V4sf, b: V4sf) -> (V4sf, V4sf) {
    V4sf::interleave2(a, b)
}

#[inline(always)]
pub(crate) fn v4_uninterleave2(a: V4sf, b: V4sf) -> (V4sf, V4sf) {
    V4sf::uninterleave2(a, b)
}

#[inline(always)]
pub(crate) fn v4_transpose(a: V4sf, b: V4sf, c: V4sf, d: V4sf) -> (V4sf, V4sf, V4sf, V4sf) {
    V4sf::transpose(a, b, c, d)
}

#[inline(always)]
pub(crate) fn v4_swaphl(a: V4sf, b: V4sf) -> V4sf {
    V4sf::swaphl(a, b)
}

// offset is in V4sf values (i.e. 16 bytes)
#[inline(always)]
pub(crate) unsafe fn v4_load(ptr: *const f32, offset: usize) -> V4sf {
    unsafe { V4sf::load(ptr.add(offset * 4)) }
}

// offset is in V4sf values (i.e. 16 bytes)
#[inline(always)]
pub(crate) unsafe fn v4_store(ptr: *mut f32, offset: usize, value: V4sf) {
    unsafe { V4sf::store(ptr.add(offset * 4), value) }
}

#[inline(always)]
pub(crate) fn v4_scalar_mul(f: f32, v: V4sf) -> V4sf {
    V4sf::scalar_mul(f, v)
}

#[inline(always)]
pub(crate) fn v4_cplx_mul(ar: V4sf, ai: V4sf, br: V4sf, bi: V4sf) -> (V4sf, V4sf) {
    V4sf::cplx_mul(ar, ai, br, bi)
}

#[inline(always)]
pub(crate) fn v4_cplx_mul_conj(ar: V4sf, ai: V4sf, br: V4sf, bi: V4sf) -> (V4sf, V4sf) {
    V4sf::cplx_mul_conj(ar, ai, br, bi)
}

#[cfg(test)]
mod tests {
    use super::{ScalarVector4, V4sf, Vector4};

    fn to_array<T: Vector4>(v: T) -> [f32; 4] {
        let mut arr = [0.0; 4];
        T::store_to_slice(v, &mut arr);
        arr
    }

    #[test]
    fn validate_simd_operations() {
        // Data from the C test
        let f: [f32; 16] = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let a0 = V4sf::load_from_slice(&f[0..]);
        let a1 = V4sf::load_from_slice(&f[4..]);
        let a2 = V4sf::load_from_slice(&f[8..]);
        let a3 = V4sf::load_from_slice(&f[12..]);

        // Test zero
        let t = V4sf::zero();
        let t_arr = to_array(t);
        assert_eq!(t_arr, [0., 0., 0., 0.], "zero failed");

        // Test add
        let t = V4sf::add(a1, a2);
        let t_arr = to_array(t);
        let expected = [12., 14., 16., 18.];
        for (i, (a, b)) in t_arr.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "add failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test mul
        let t = V4sf::mul(a1, a2);
        let t_arr = to_array(t);
        let expected = [32., 45., 60., 77.];
        for (i, (a, b)) in t_arr.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mul failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test fma
        let t = V4sf::fma(a1, a2, a0);
        let t_arr = to_array(t);
        let expected = [32., 46., 62., 80.];
        for (i, (a, b)) in t_arr.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "fma failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test interleave2
        let (t, u) = V4sf::interleave2(a1, a2);
        let t_arr = to_array(t);
        let u_arr = to_array(u);
        let expected_t = [4., 8., 5., 9.];
        let expected_u = [6., 10., 7., 11.];
        for (i, (a, b)) in t_arr.iter().zip(expected_t.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "interleave2 first output failed at index {}: expected {}, got {}", i, b, a);
        }
        for (i, (a, b)) in u_arr.iter().zip(expected_u.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "interleave2 second output failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test uninterleave2
        let (t, u) = V4sf::uninterleave2(a1, a2);
        let t_arr = to_array(t);
        let u_arr = to_array(u);
        let expected_t = [4., 6., 8., 10.];
        let expected_u = [5., 7., 9., 11.];
        for (i, (a, b)) in t_arr.iter().zip(expected_t.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "uninterleave2 first output failed at index {}: expected {}, got {}",
                i,
                b,
                a
            );
        }
        for (i, (a, b)) in u_arr.iter().zip(expected_u.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "uninterleave2 second output failed at index {}: expected {}, got {}",
                i,
                b,
                a
            );
        }

        // Test splat (broadcast)
        let t = V4sf::splat(15.);
        let t_arr = to_array(t);
        let expected = [15., 15., 15., 15.];
        for (i, (a, b)) in t_arr.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "splat failed at index {}: expected {}, got {}", i, b, a);
        }

        let t = V4sf::scalar_mul(2.0, a1);
        let t_arr = to_array(t);
        let expected = [8., 10., 12., 14.]; // a1 is [4.,5.,6.,7.] * 2
        for (i, (a, b)) in t_arr.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "scalar_mul failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test swaphl
        let t = V4sf::swaphl(a1, a2);
        let t_arr = to_array(t);
        let expected = [8., 9., 6., 7.];
        for (i, (a, b)) in t_arr.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "swaphl failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test transpose
        let (x0, x1, x2, x3) = V4sf::transpose(a0, a1, a2, a3);
        let x0_arr = to_array(x0);
        let x1_arr = to_array(x1);
        let x2_arr = to_array(x2);
        let x3_arr = to_array(x3);
        let expected_x0 = [0., 4., 8., 12.];
        let expected_x1 = [1., 5., 9., 13.];
        let expected_x2 = [2., 6., 10., 14.];
        let expected_x3 = [3., 7., 11., 15.];
        for (i, (a, b)) in x0_arr.iter().zip(expected_x0.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "transpose x0 failed at index {}: expected {}, got {}", i, b, a);
        }
        for (i, (a, b)) in x1_arr.iter().zip(expected_x1.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "transpose x1 failed at index {}: expected {}, got {}", i, b, a);
        }
        for (i, (a, b)) in x2_arr.iter().zip(expected_x2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "transpose x2 failed at index {}: expected {}, got {}", i, b, a);
        }
        for (i, (a, b)) in x3_arr.iter().zip(expected_x3.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "transpose x3 failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test cplx_mul
        let (ar, ai) = V4sf::cplx_mul(a1, a2, a3, a0);
        let ar_arr = to_array(ar);
        let ai_arr = to_array(ai);
        let expected_ar = [48., 56., 64., 72.];
        let expected_ai = [96., 122., 152., 186.];
        for (i, (a, b)) in ar_arr.iter().zip(expected_ar.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "cplx_mul real part failed at index {}: expected {}, got {}", i, b, a);
        }
        for (i, (a, b)) in ai_arr.iter().zip(expected_ai.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "cplx_mul imag part failed at index {}: expected {}, got {}", i, b, a);
        }

        // Test cplx_mul_conj
        let (ar, ai) = V4sf::cplx_mul_conj(a1, a2, a3, a0);
        let ar_arr = to_array(ar);
        let ai_arr = to_array(ai);
        let expected_ar = [48., 74., 104., 138.];
        let expected_ai = [96., 112., 128., 144.];
        for (i, (a, b)) in ar_arr.iter().zip(expected_ar.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "cplx_mul_conj real part failed at index {}: expected {}, got {}", i, b, a);
        }
        for (i, (a, b)) in ai_arr.iter().zip(expected_ai.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "cplx_mul_conj imag part failed at index {}: expected {}, got {}", i, b, a);
        }
    }

    #[test]
    #[should_panic(expected = "slice must have at least 4 elements for Vector4 load")]
    fn test_load_from_slice_too_short() {
        let data = [1.0, 2.0, 3.0];
        let _ = V4sf::load_from_slice(&data);
    }

    #[test]
    #[should_panic(expected = "slice must have at least 4 elements for Vector4 store")]
    fn test_store_to_slice_too_short() {
        let v = V4sf::zero();
        let mut out = [0.0; 3];
        v.store_to_slice(&mut out);
    }

    #[test]
    fn test_simd_scalar_equivalence() {
        // Test that V4sf and ScalarVector4 produce identical results for all operations
        use super::Vector4;

        // Test data
        let f: [f32; 16] = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let a0_arr = [f[0], f[1], f[2], f[3]];
        let a1_arr = [f[4], f[5], f[6], f[7]];
        let a2_arr = [f[8], f[9], f[10], f[11]];
        let a3_arr = [f[12], f[13], f[14], f[15]];

        let simd_a0 = V4sf::load_from_slice(&a0_arr);
        let simd_a1 = V4sf::load_from_slice(&a1_arr);
        let simd_a2 = V4sf::load_from_slice(&a2_arr);
        let simd_a3 = V4sf::load_from_slice(&a3_arr);

        let scalar_a0 = ScalarVector4::load_from_slice(&a0_arr);
        let scalar_a1 = ScalarVector4::load_from_slice(&a1_arr);
        let scalar_a2 = ScalarVector4::load_from_slice(&a2_arr);
        let scalar_a3 = ScalarVector4::load_from_slice(&a3_arr);

        // Helper to compare results
        fn compare<T: Vector4, U: Vector4>(simd: T, scalar: U, operation: &str) {
            let simd_arr = to_array(simd);
            let scalar_arr = to_array(scalar);
            for i in 0..4 {
                assert!(
                    (simd_arr[i] - scalar_arr[i]).abs() < 1e-5,
                    "{} mismatch at index {}: simd={}, scalar={}",
                    operation,
                    i,
                    simd_arr[i],
                    scalar_arr[i]
                );
            }
        }

        // Test each operation
        compare(V4sf::zero(), ScalarVector4::zero(), "zero");
        compare(V4sf::add(simd_a1, simd_a2), ScalarVector4::add(scalar_a1, scalar_a2), "add");
        compare(V4sf::mul(simd_a1, simd_a2), ScalarVector4::mul(scalar_a1, scalar_a2), "mul");
        compare(V4sf::fma(simd_a1, simd_a2, simd_a0), ScalarVector4::fma(scalar_a1, scalar_a2, scalar_a0), "fma");
        compare(V4sf::sub(simd_a1, simd_a2), ScalarVector4::sub(scalar_a1, scalar_a2), "sub");
        compare(V4sf::splat(15.0), ScalarVector4::splat(15.0), "vldps1");
        compare(V4sf::scalar_mul(2.0, simd_a1), ScalarVector4::scalar_mul(2.0, scalar_a1), "scalar_mul");

        // Interleave2
        let (simd_t, simd_u) = V4sf::interleave2(simd_a1, simd_a2);
        let (scalar_t, scalar_u) = ScalarVector4::interleave2(scalar_a1, scalar_a2);
        compare(simd_t, scalar_t, "interleave2 first");
        compare(simd_u, scalar_u, "interleave2 second");

        // Uninterleave2
        let (simd_t, simd_u) = V4sf::uninterleave2(simd_a1, simd_a2);
        let (scalar_t, scalar_u) = ScalarVector4::uninterleave2(scalar_a1, scalar_a2);
        compare(simd_t, scalar_t, "uninterleave2 first");
        compare(simd_u, scalar_u, "uninterleave2 second");

        // swaphl
        compare(V4sf::swaphl(simd_a1, simd_a2), ScalarVector4::swaphl(scalar_a1, scalar_a2), "swaphl");

        // Transpose
        let (simd_x0, simd_x1, simd_x2, simd_x3) = V4sf::transpose(simd_a0, simd_a1, simd_a2, simd_a3);
        let (scalar_x0, scalar_x1, scalar_x2, scalar_x3) =
            ScalarVector4::transpose(scalar_a0, scalar_a1, scalar_a2, scalar_a3);
        compare(simd_x0, scalar_x0, "transpose x0");
        compare(simd_x1, scalar_x1, "transpose x1");
        compare(simd_x2, scalar_x2, "transpose x2");
        compare(simd_x3, scalar_x3, "transpose x3");
    }
}
