import { describe, expect, test, beforeAll } from 'vitest';
import { initSync, Float32Vec } from '../wasm/build/wasm_main_module';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmPath = join(__dirname, '../wasm/build/wasm_main_module_bg.wasm');

// Skip all tests unless WASM_INTEGRATION environment variable is set
const runIntegrationTests = process.env.WASM_INTEGRATION === 'true';

describe.skipIf(!runIntegrationTests)('Float32Vec', () => {
    beforeAll(() => {
        const wasmBuffer = readFileSync(wasmPath);
        initSync({ module: wasmBuffer });
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
