use wasm_bindgen::prelude::*;

// Float32Owned is used to optimize passing data between wasm and js:
//  * Float32Arrays can be created from wasm memory using data_ptr() and len() to fill input and read output parameters
//  * clear() can be used to reuse Float32Owneds between calls

#[wasm_bindgen]
pub struct Float32Owned {
	#[wasm_bindgen(skip)]
    pub data: Vec<f32>,
}

#[wasm_bindgen]
impl Float32Owned {
    #[wasm_bindgen(constructor)]
    pub fn new(len: usize) -> Float32Owned {
        let mut data = Vec::new();
        data.resize(len, 0.0);
        Float32Owned { data }
    }

    #[wasm_bindgen(getter)]
    pub fn data_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }

    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.data.clear();
    }
}
