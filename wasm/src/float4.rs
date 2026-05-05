#[allow(dead_code)]

// Common trait for 4-element SIMD f32 vectors. Each platform then defines a SimdFloat4 implementation (the platforms
// without SIMD use ScalarFloat4).
pub(crate) trait Float4: Copy {
    /// Element-wise multiplication
    fn mul(a: Self, b: Self) -> Self;

    /// Element-wise addition
    fn add(a: Self, b: Self) -> Self;

    /// Element-wise subtraction
    fn sub(a: Self, b: Self) -> Self;

    /// Broadcast scalar to all lanes
    fn splat(f: f32) -> Self;

    /// Unaligned load from memory
    unsafe fn load(ptr: *const f32) -> Self;

    /// Unaligned store to memory
    unsafe fn store(ptr: *mut f32, value: Self);

    /// Unaligned load from memory of two vectors with deinterleaving, returns (even, odd)
    unsafe fn load_deinterleave2(ptr: *const f32) -> (Self, Self);

    /// Unaligned load from memory of two vectors with deinterleaving and reversing, returns (reverse(even),
    /// reverse(odd)), where reverse(a) = [a[3], a[2], a[1], a[0]]
    unsafe fn load_deinterleave2_rev(ptr: *const f32) -> (Self, Self);

    /// Unaligned store to memory of two vectors with interleaving
    unsafe fn store_interleave2(ptr: *mut f32, even: Self, odd: Self);

    /// Unaligned store to memory of two vectors with reversing and interleaving
    unsafe fn store_interleave2_rev(ptr: *mut f32, even: Self, odd: Self);

    /// Safe load from an unaligned slice
    #[inline(always)]
    fn load_from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= 4, "slice must have at least 4 elements for Float4 load");
        unsafe { Self::load(slice.as_ptr()) }
    }

    /// Safe store to an unaligned slice
    #[inline(always)]
    fn store_to_slice(self, slice: &mut [f32]) {
        assert!(slice.len() >= 4, "slice must have at least 4 elements for Float4 store");
        unsafe { Self::store(slice.as_mut_ptr(), self) }
    }
}

// SSE2 implementation
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod sse2 {
    use super::*;
    use core::arch::x86_64::__m128;
    use core::arch::x86_64::{
        _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_shuffle_ps, _mm_storeu_ps, _mm_sub_ps, _mm_unpackhi_ps,
        _mm_unpacklo_ps,
    };

    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub(crate) struct SSE2Float4(__m128);

    impl Float4 for SSE2Float4 {
        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            unsafe { SSE2Float4(_mm_mul_ps(a.0, b.0)) }
        }

        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            unsafe { SSE2Float4(_mm_add_ps(a.0, b.0)) }
        }

        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            unsafe { SSE2Float4(_mm_sub_ps(a.0, b.0)) }
        }

        #[inline(always)]
        fn splat(f: f32) -> Self {
            unsafe { SSE2Float4(_mm_set1_ps(f)) }
        }

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            unsafe { SSE2Float4(_mm_loadu_ps(ptr)) }
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, value: Self) {
            unsafe {
                _mm_storeu_ps(ptr, value.0);
            }
        }

        #[inline(always)]
        unsafe fn load_deinterleave2(ptr: *const f32) -> (Self, Self) {
            unsafe {
                let low = _mm_loadu_ps(ptr);
                let high = _mm_loadu_ps(ptr.add(4));
                // _MM_SHUFFLE(2,0,2,0): result = [a0, a2, b0, b2]
                let even = _mm_shuffle_ps(low, high, 0b10001000);
                // _MM_SHUFFLE(3,1,3,1): result = [a1, a3, b1, b3]
                let odd = _mm_shuffle_ps(low, high, 0b11011101);
                (SSE2Float4(even), SSE2Float4(odd))
            }
        }

        #[inline(always)]
        unsafe fn store_interleave2(ptr: *mut f32, even: Self, odd: Self) {
            unsafe {
                let shuf0 = _mm_unpacklo_ps(even.0, odd.0);
                let shuf1 = _mm_unpackhi_ps(even.0, odd.0);
                _mm_storeu_ps(ptr, shuf0);
                _mm_storeu_ps(ptr.add(4), shuf1);
            }
        }

        #[inline(always)]
        unsafe fn load_deinterleave2_rev(ptr: *const f32) -> (Self, Self) {
            unsafe {
                let low = _mm_loadu_ps(ptr);
                let high = _mm_loadu_ps(ptr.add(4));
                // _MM_SHUFFLE(0,2,0,2): result = [b2, b0, a2, a0]
                let even = _mm_shuffle_ps(high, low, 0b100010);
                // _MM_SHUFFLE(1,3,1,3): result = [b3, b1, a3, a1]
                let odd = _mm_shuffle_ps(high, low, 0b01110111);
                (SSE2Float4(even), SSE2Float4(odd))
            }
        }

        #[inline(always)]
        unsafe fn store_interleave2_rev(ptr: *mut f32, even: Self, odd: Self) {
            unsafe {
                // _MM_SHUFFLE(3,2,1,0): rev0 = [e3, e2, e1, e0]
                let rev0 = _mm_shuffle_ps(even.0, even.0, 0b00011011);
                // _MM_SHUFFLE(3,2,1,0): rev1 = [o3, o2, o1, o0]
                let rev1 = _mm_shuffle_ps(odd.0, odd.0, 0b00011011);
                let high = _mm_unpacklo_ps(rev0, rev1);
                let low = _mm_unpackhi_ps(rev0, rev1);
                _mm_storeu_ps(ptr, low);
                _mm_storeu_ps(ptr.add(4), high);
            }
        }
    }
}

// NEON implementation
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::*;
    use core::arch::aarch64::{
        float32x4_t, float32x4x2_t, vaddq_f32, vextq_f32, vld1q_dup_f32, vld1q_f32, vld1q_f32_x2, vld2q_f32, vmulq_f32,
        vst1q_f32, vst1q_f32_x2, vst2q_f32, vsubq_f32, vuzpq_f32, vzipq_f32,
    };

    #[repr(transparent)]
    #[derive(Copy, Clone)]
    pub(crate) struct NEONFloat4(float32x4_t);

    impl Float4 for NEONFloat4 {
        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            unsafe { NEONFloat4(vmulq_f32(a.0, b.0)) }
        }

        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            unsafe { NEONFloat4(vaddq_f32(a.0, b.0)) }
        }

        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            unsafe { NEONFloat4(vsubq_f32(a.0, b.0)) }
        }

        #[inline(always)]
        fn splat(f: f32) -> Self {
            unsafe { NEONFloat4(vld1q_dup_f32(&f)) }
        }

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            unsafe { NEONFloat4(vld1q_f32(ptr)) }
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, value: Self) {
            unsafe { vst1q_f32(ptr, value.0) }
        }

        #[inline(always)]
        unsafe fn load_deinterleave2(ptr: *const f32) -> (Self, Self) {
            unsafe {
                let dbl = vld2q_f32(ptr);
                (NEONFloat4(dbl.0), NEONFloat4(dbl.1))
            }
        }

        #[inline(always)]
        unsafe fn load_deinterleave2_rev(ptr: *const f32) -> (Self, Self) {
            unsafe {
                // pair = [e0 o0 e1 o1] [e2 o2 e3 o3]
                let pair = vld1q_f32_x2(ptr);
                // rev0 = [e1 o1 e0 o0]
                let rev0 = vextq_f32(pair.0, pair.0, 2);
                // rev1 = [e3 o3 e2 o2]
                let rev1 = vextq_f32(pair.1, pair.1, 2);
                // unzipped = [e3 e2 e1 e0] [o3 o2 o1 o0]
                let unzipped = vuzpq_f32(rev1, rev0);
                (NEONFloat4(unzipped.0), NEONFloat4(unzipped.1))
            }
        }

        #[inline(always)]
        unsafe fn store_interleave2(ptr: *mut f32, even: Self, odd: Self) {
            unsafe { vst2q_f32(ptr, float32x4x2_t(even.0, odd.0)) }
        }

        #[inline(always)]
        unsafe fn store_interleave2_rev(ptr: *mut f32, even: Self, odd: Self) {
            unsafe {
                // zipped = [e0 o0 e1 o0] [e2 o2 e3 o3]
                let zipped = vzipq_f32(even.0, odd.0);
                // rev0 = [e1 o1 e0 o0]
                let rev0 = vextq_f32(zipped.0, zipped.0, 2);
                // rev1 = [e3 o3 e2 o2]
                let rev1 = vextq_f32(zipped.1, zipped.1, 2);
                vst1q_f32_x2(ptr, float32x4x2_t(rev1, rev0))
            }
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
    pub(crate) struct WASMFloat4(v128);

    impl Float4 for WASMFloat4 {
        #[inline(always)]
        fn mul(a: Self, b: Self) -> Self {
            WASMFloat4(f32x4_mul(a.0, b.0))
        }

        #[inline(always)]
        fn add(a: Self, b: Self) -> Self {
            WASMFloat4(f32x4_add(a.0, b.0))
        }

        #[inline(always)]
        fn sub(a: Self, b: Self) -> Self {
            WASMFloat4(f32x4_sub(a.0, b.0))
        }

        #[inline(always)]
        fn splat(f: f32) -> Self {
            WASMFloat4(f32x4_splat(f))
        }

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self {
            unsafe { WASMFloat4(v128_load(ptr as *const v128)) }
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, value: Self) {
            unsafe { v128_store(ptr as *mut v128, value.0) }
        }

        #[inline(always)]
        unsafe fn load_deinterleave2(ptr: *const f32) -> (Self, Self) {
            let low = unsafe { v128_load(ptr as *const v128) };
            let high = unsafe { v128_load(ptr.add(4) as *const v128) };
            // even: [low0, low2, high0, high2]
            let even = i32x4_shuffle::<0, 2, 4, 6>(low, high);
            // odd: [low1, low3, high1, high3]
            let odd = i32x4_shuffle::<1, 3, 5, 7>(low, high);
            (WASMFloat4(even), WASMFloat4(odd))
        }

        #[inline(always)]
        unsafe fn store_interleave2(ptr: *mut f32, even: Self, odd: Self) {
            // low: [even0, odd0, even1, odd1]
            let low = i32x4_shuffle::<0, 4, 1, 5>(even.0, odd.0);
            // high: [even2, odd2, even3, odd3]
            let high = i32x4_shuffle::<2, 6, 3, 7>(even.0, odd.0);
            unsafe {
                v128_store(ptr as *mut v128, low);
                v128_store(ptr.add(4) as *mut v128, high);
            }
        }

        #[inline(always)]
        unsafe fn load_deinterleave2_rev(ptr: *const f32) -> (Self, Self) {
            let low = unsafe { v128_load(ptr as *const v128) };
            let high = unsafe { v128_load(ptr.add(4) as *const v128) };
            // even: [high2, high0, low2, low0]
            let even = i32x4_shuffle::<6, 4, 2, 0>(low, high);
            // odd: [high3, high1, low3, low1]
            let odd = i32x4_shuffle::<7, 5, 3, 1>(low, high);
            (WASMFloat4(even), WASMFloat4(odd))
        }

        #[inline(always)]
        unsafe fn store_interleave2_rev(ptr: *mut f32, even: Self, odd: Self) {
            // low: [even3, odd3, even2, odd2]
            let low = i32x4_shuffle::<3, 7, 2, 6>(even.0, odd.0);
            // high: [even0, odd0, even0, odd0]
            let high = i32x4_shuffle::<1, 5, 0, 4>(even.0, odd.0);
            unsafe {
                v128_store(ptr as *mut v128, low);
                v128_store(ptr.add(4) as *mut v128, high);
            }
        }
    }
}

// Scalar fallback implementation (when SIMD is not available)
#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct ScalarFloat4([f32; 4]);

impl Float4 for ScalarFloat4 {
    #[inline(always)]
    fn mul(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] * b.0[i];
        }
        ScalarFloat4(result)
    }

    #[inline(always)]
    fn add(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] + b.0[i];
        }
        ScalarFloat4(result)
    }

    #[inline(always)]
    fn sub(a: Self, b: Self) -> Self {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = a.0[i] - b.0[i];
        }
        ScalarFloat4(result)
    }

    #[inline(always)]
    fn splat(f: f32) -> Self {
        ScalarFloat4([f; 4])
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        // Note: scalar load doesn't care about alignment
        let mut arr = [0.0; 4];
        unsafe { std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 4) }
        ScalarFloat4(arr)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f32, value: Self) {
        unsafe { std::ptr::copy_nonoverlapping(value.0.as_ptr(), ptr, 4) }
    }

    #[inline(always)]
    unsafe fn load_deinterleave2(ptr: *const f32) -> (Self, Self) {
        unsafe {
            let arr0 = [*ptr, *ptr.add(2), *ptr.add(4), *ptr.add(6)];
            let arr1 = [*ptr.add(1), *ptr.add(3), *ptr.add(5), *ptr.add(7)];
            (ScalarFloat4(arr0), ScalarFloat4(arr1))
        }
    }

    #[inline(always)]
    unsafe fn store_interleave2(ptr: *mut f32, even: Self, odd: Self) {
        unsafe {
            *ptr = even.0[0];
            *ptr.add(1) = odd.0[0];
            *ptr.add(2) = even.0[1];
            *ptr.add(3) = odd.0[1];
            *ptr.add(4) = even.0[2];
            *ptr.add(5) = odd.0[2];
            *ptr.add(6) = even.0[3];
            *ptr.add(7) = odd.0[3];
        }
    }

    #[inline(always)]
    unsafe fn load_deinterleave2_rev(ptr: *const f32) -> (Self, Self) {
        unsafe {
            let arr0 = [*ptr.add(6), *ptr.add(4), *ptr.add(2), *ptr];
            let arr1 = [*ptr.add(7), *ptr.add(5), *ptr.add(3), *ptr.add(1)];
            (ScalarFloat4(arr0), ScalarFloat4(arr1))
        }
    }

    #[inline(always)]
    unsafe fn store_interleave2_rev(ptr: *mut f32, even: Self, odd: Self) {
        unsafe {
            *ptr = even.0[3];
            *ptr.add(1) = odd.0[3];
            *ptr.add(2) = even.0[2];
            *ptr.add(3) = odd.0[2];
            *ptr.add(4) = even.0[1];
            *ptr.add(5) = odd.0[1];
            *ptr.add(6) = even.0[0];
            *ptr.add(7) = odd.0[0];
        }
    }
}

// Export the appropriate type based on target architecture
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(crate) use sse2::SSE2Float4 as SimdFloat4;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) use neon::NEONFloat4 as SimdFloat4;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm_simd::WASMFloat4 as SimdFloat4;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub(crate) use ScalarFloat4 as SimdFloat4;

#[cfg(test)]
mod tests {
    use super::{Float4, ScalarFloat4, SimdFloat4};

    fn to_array<T: Float4>(v: T) -> [f32; 4] {
        let mut arr = [0.0; 4];
        T::store_to_slice(v, &mut arr);
        arr
    }

    #[test]
    fn validate_simd_operations() {
        let f: [f32; 8] = [4., 5., 6., 7., 8., 9., 10., 11.];
        let a1 = SimdFloat4::load_from_slice(&f[0..]);
        let a2 = SimdFloat4::load_from_slice(&f[4..]);

        // Helper to compare results
        fn compare<T: Float4>(simd: T, v0: f32, v1: f32, v2: f32, v3: f32, operation: &str) {
            let simd_arr = to_array(simd);
            assert_eq!(simd_arr[0], v0, "{} mismatch at index 0", operation);
            assert_eq!(simd_arr[1], v1, "{} mismatch at index 1", operation);
            assert_eq!(simd_arr[2], v2, "{} mismatch at index 2", operation);
            assert_eq!(simd_arr[3], v3, "{} mismatch at index 3", operation);
        }

        compare(SimdFloat4::mul(a1, a2), 32., 45., 60., 77., "mul");
        compare(SimdFloat4::add(a1, a2), 12., 14., 16., 18., "add");
        compare(SimdFloat4::sub(a1, a2), -4., -4., -4., -4., "sub");
        compare(SimdFloat4::splat(15.), 15., 15., 15., 15., "splat");

        let (t, u) = unsafe { SimdFloat4::load_deinterleave2(f.as_ptr()) };
        compare(t, 4., 6., 8., 10., "load_deinterleave2 first");
        compare(u, 5., 7., 9., 11., "load_deinterleave2 second");

        let mut f_out = [0.0f32; 8];
        unsafe { SimdFloat4::store_interleave2(f_out.as_mut_ptr(), a2, a1) };
        assert_eq!(f_out, [8., 4., 9., 5., 10., 6., 11., 7.], "store_interleave2");

        let (t, u) = unsafe { SimdFloat4::load_deinterleave2_rev(f.as_ptr()) };
        compare(t, 10., 8., 6., 4., "load_deinterleave2_rev first");
        compare(u, 11., 9., 7., 5., "load_deinterleave2_rev second");

        let mut f_out = [0.0f32; 8];
        unsafe { SimdFloat4::store_interleave2_rev(f_out.as_mut_ptr(), a2, a1) };
        assert_eq!(f_out, [11., 7., 10., 6., 9., 5., 8., 4.], "store_interleave2_rev");
    }

    #[test]
    #[should_panic(expected = "slice must have at least 4 elements for Float4 load")]
    fn test_load_from_slice_too_short() {
        let data = [1.0, 2.0, 3.0];
        let _ = SimdFloat4::load_from_slice(&data);
    }

    #[test]
    #[should_panic(expected = "slice must have at least 4 elements for Float4 store")]
    fn test_store_to_slice_too_short() {
        let v = SimdFloat4::splat(1.0);
        let mut out = [0.0; 3];
        v.store_to_slice(&mut out);
    }

    #[test]
    fn test_simd_scalar_equivalence() {
        // Test that SimdFloat4 and ScalarFloat4 produce identical results for all operations
        use super::Float4;

        let f: [f32; 8] = [4., 5., 6., 7., 8., 9., 10., 11.];

        let simd_a1 = SimdFloat4::load_from_slice(&f[0..]);
        let simd_a2 = SimdFloat4::load_from_slice(&f[4..]);

        let scalar_a1 = ScalarFloat4::load_from_slice(&f[0..]);
        let scalar_a2 = ScalarFloat4::load_from_slice(&f[4..]);

        // Helper to compare results
        fn compare<T: Float4, U: Float4>(simd: T, scalar: U, operation: &str) {
            let simd_arr = to_array(simd);
            let scalar_arr = to_array(scalar);
            for i in 0..4 {
                assert_eq!(simd_arr[i], scalar_arr[i], "{} mismatch at index {}", operation, i);
            }
        }

        compare(SimdFloat4::add(simd_a1, simd_a2), ScalarFloat4::add(scalar_a1, scalar_a2), "add");
        compare(SimdFloat4::mul(simd_a1, simd_a2), ScalarFloat4::mul(scalar_a1, scalar_a2), "mul");
        compare(SimdFloat4::sub(simd_a1, simd_a2), ScalarFloat4::sub(scalar_a1, scalar_a2), "sub");
        compare(SimdFloat4::splat(15.0), ScalarFloat4::splat(15.0), "splat");
    }
}
