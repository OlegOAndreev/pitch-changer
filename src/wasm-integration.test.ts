import { beforeAll, describe, expect, test } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';

// Skip all tests unless WASM_INTEGRATION environment variable is set
const runIntegrationTests = process.env.WASM_INTEGRATION === 'true';

if (runIntegrationTests) {
    describe('Float32Vec', () => {
        let Float32Vec: typeof import('../wasm/build/wasm_main_module').Float32Vec;

        beforeAll(async () => {
            // Dynamically import the wasm module only when tests are run
            const wasmModule = await import('../wasm/build/wasm_main_module');
            // Use initSync with the wasm binary for Node.js environment
            const wasmPath = join(__dirname, '../wasm/build/wasm_main_module_bg.wasm');
            const wasmBinary = readFileSync(wasmPath);
            wasmModule.initSync({ module: wasmBinary });
            Float32Vec = wasmModule.Float32Vec;
        });

        test('constructor', () => {
            const len = 10;
            const vec = new Float32Vec(len);
            try {
                expect(vec.len).toBe(len);
                const arr = vec.array;
                expect(arr).toHaveLength(len);
                for (let i = 0; i < arr.length; i++) {
                    expect(arr[i]).toBe(0);
                }
            } finally {
                vec.free();
            }
        });

        test('clear and set', () => {
            const vec = new Float32Vec(5);
            try {
                const data = new Float32Array([1, 2, 3, 4, 5]);
                vec.set(data);
                expect(vec.len).toBe(5);
                vec.clear();
                expect(vec.len).toBe(0);
                expect(vec.array).toHaveLength(0);
            } finally {
                vec.free();
            }
        });

        test('resize', () => {
            const vec = new Float32Vec(3);
            try {
                expect(vec.len).toBe(3);
                vec.resize(10);
                expect(vec.len).toBe(10);
                const arr = vec.array;
                for (let i = 3; i < arr.length; i++) {
                    expect(arr[i]).toBe(0);
                }
                vec.resize(2);
                expect(vec.len).toBe(2);
            } finally {
                vec.free();
            }
        });

        test('array view', () => {
            const vec = new Float32Vec(0);
            try {
                const data1 = new Float32Array([1, 2, 3]);
                vec.set(data1);
                const arr1 = vec.array;
                expect(arr1).toEqual(new Float32Array([1, 2, 3]));
                const arr2 = vec.array;
                arr2[0] = 4;
                arr2[1] = 5;
                arr2[2] = 6;
                expect(arr1).toEqual(new Float32Array([4, 5, 6]));
            } finally {
                vec.free();
            }
        });

        test('Symbol.dispose works', () => {
            const vec = new Float32Vec(10);
            // TypeScript's Symbol.dispose (if supported)
            if (typeof vec[Symbol.dispose] === 'function') {
                vec[Symbol.dispose]();
            } else {
                vec.free();
            }
        });
    });
} else {
    describe('Float32Vec', () => {
        test('skipped - WASM_INTEGRATION not set', () => {
            // This test suite is skipped when WASM_INTEGRATION is not 'true'
            expect(true).toBe(true);
        });
    });
}
