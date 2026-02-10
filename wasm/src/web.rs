use std::error::Error;
use std::fmt::Display;
use std::panic;

use wasm_bindgen::prelude::*;

// Copy-pasted from https://wasm-bindgen.github.io/wasm-bindgen/examples/console-log.html

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    /// Proxy for browser console.log()
    pub fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    /// Proxy for browser console.error()
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
macro_rules! console_error {
    ($($t:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        {
            crate::web::error(&format_args!($($t)*).to_string());
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

/// Set the panic hook on start, otherwise we miss out on all our asserts.
#[wasm_bindgen(start)]
fn start() {
    panic::set_hook(Box::new(|panic_info| {
        let message = format!("Panic occurred: {}", panic_info);
        console_error!("{}", message);
    }));
}

/// WrapAnyhowError is a wrapper for anyhow Error which supports Into<JsError> (goddamn orphan rules).
#[derive(Debug)]
pub struct WrapAnyhowError(pub anyhow::Error);

impl Display for WrapAnyhowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for WrapAnyhowError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.0.source()
    }

    fn description(&self) -> &str {
        "description is deprecated"
    }

    fn cause(&self) -> Option<&dyn Error> {
        // Cause is deprecated
        None
    }
}

impl From<WrapAnyhowError> for JsValue {
    fn from(val: WrapAnyhowError) -> Self {
        JsValue::from_str(&format!("{:?}", val.0))
    }
}

#[wasm_bindgen]
/// Return a string describing compilation flags.
pub fn get_settings() -> String {
    let optimized = !cfg!(debug_assertions);

    let atomics = cfg!(target_feature = "atomics");
    let bulk_memory = cfg!(target_feature = "bulk-memory");
    let multivalue = cfg!(target_feature = "multivalue");
    let nontrapping_fptoint = cfg!(target_feature = "nontrapping-fptoint");
    let sign_ext = cfg!(target_feature = "sign-ext");
    let simd128 = cfg!(target_feature = "simd128");
    let relaxed_simd = cfg!(target_feature = "relaxed-simd");

    format!(
        "optimized: {}, atomics: {}, bulk-memory: {}, multivalue: {}, nontrapping-fptoint: {}, sign-ext: {}, \
         simd128: {}, relaxed-simd: {}",
        optimized, atomics, bulk_memory, multivalue, nontrapping_fptoint, sign_ext, simd128, relaxed_simd
    )
}

/// Float32Vec is a wrapper for Vec<f32> that is used to optimize passing data between wasm and js:
///  * Float32Arrays can be created from wasm memory using data_ptr() and len() to fill input and read output parameters
///  * clear() can be used to reuse Float32Owneds between calls
#[wasm_bindgen]
pub struct Float32Vec(#[wasm_bindgen(skip)] pub Vec<f32>);

#[wasm_bindgen]
impl Float32Vec {
    /// Create a new Float32Vec with given size, filled with 0.0.
    #[wasm_bindgen(constructor)]
    pub fn new(len: usize) -> Float32Vec {
        Float32Vec(vec![0.0; len])
    }

    /// Return Float32Vec data ptr (is a number on JS side). Accessing Float32Array a la Emscripten HEAPF32 on JS side
    /// should be marginally faster than creating Float32Array using array property.
    #[wasm_bindgen(getter)]
    pub fn data_ptr(&mut self) -> *mut f32 {
        self.0.as_mut_ptr()
    }

    /// Return Float32Vec length.
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Remove all data from Float32Vec.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Resize Float32Vec to given size, either truncating or appending 0.0 if required.
    #[wasm_bindgen]
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n, 0.0);
    }

    /// Resize Float32Vec with zeros.
    #[wasm_bindgen]
    pub fn fill(&mut self) {
        self.0.fill(0.0);
    }

    /// Set Float32Vec contents from Float32Array.
    #[wasm_bindgen]
    pub fn set(&mut self, arr: &js_sys::Float32Array) {
        self.0.resize(arr.length() as usize, 0.0);
        arr.copy_to(&mut self.0);
    }

    /// Create a Float32Array view for Float32Vec. Warning: the view is valid only until the Float32Vec is resized.
    #[wasm_bindgen(getter)]
    pub fn array(&self) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(&self.0) }
    }
}
