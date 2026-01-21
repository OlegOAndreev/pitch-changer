use wasm_bindgen::prelude::*;

// Copy-pasted from https://wasm-bindgen.github.io/wasm-bindgen/examples/console-log.html

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    pub fn error(s: &str);
}

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        {
            crate::web::log(&format_args!($($t)*).to_string());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!($($t)*);
        }
    }
}

#[allow(unused_macros)]
macro_rules! error_and_panic {
    ($($t:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        {
            crate::web::error(&format_args!($($t)*).to_string());
            panic!();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            panic!($($t)*);
        }
    }
}

#[allow(unused)]
pub(crate) use console_log;
#[allow(unused)]
pub(crate) use error_and_panic;

#[wasm_bindgen]
pub fn get_settings() -> String {
    let optimized = !cfg!(debug_assertions);

    let atomics = cfg!(target_feature = "atomics");
    let bulk_memory = cfg!(target_feature = "bulk-memory");
    let multivalue = cfg!(target_feature = "multivalue");
    let nontrapping_fptoint = cfg!(target_feature = "nontrapping-fptoint");
    let sign_ext = cfg!(target_feature = "sign-ext");
    let simd128 = cfg!(target_feature = "simd128");
    let relaxed_simd = cfg!(target_feature = "relaxed-simd");

    #[cfg(target_arch = "wasm32")]
    let memory_bytes = std::arch::wasm32::memory_size(0) * 65536;
    #[cfg(not(target_arch = "wasm32"))]
    let memory_bytes: usize = 0;

    format!(
        "optimized: {}, atomics: {}, bulk-memory: {}, multivalue: {}, nontrapping-fptoint: {}, sign-ext: {}, \
         simd128: {}, relaxed-simd: {}, memory: {}",
        optimized, atomics, bulk_memory, multivalue, nontrapping_fptoint, sign_ext, simd128, relaxed_simd, memory_bytes
    )
}
