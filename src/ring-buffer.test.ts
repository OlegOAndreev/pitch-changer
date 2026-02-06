import { describe, expect, test } from 'vitest';

import { Float32RingBuffer } from './ring-buffer';

// Note: Testing actual blocking behavior requires multiple threads, which is beyond the scope of unit tests.
describe('Float32RingBuffer', () => {
    test('constructor', () => {
        const buffer = new Float32RingBuffer(1024);
        expect(buffer.available()).toBe(0);
    });

    test('push returns number of elements pushed', () => {
        const buffer = new Float32RingBuffer(8);
        const src = new Float32Array([1, 2, 3]);
        expect(buffer.push(src)).toBe(3);
        expect(buffer.available()).toBe(3);
    });

    test('push respects capacity', () => {
        const buffer = new Float32RingBuffer(4);
        const src1 = new Float32Array([1, 2, 3, 4]);
        expect(buffer.push(src1)).toBe(4);
        expect(buffer.available()).toBe(4);
        // buffer is full, next push should push 0
        const src2 = new Float32Array([5, 6]);
        expect(buffer.push(src2)).toBe(0);
        expect(buffer.available()).toBe(4);
    });

    test('pop returns number of elements popped', () => {
        const buffer = new Float32RingBuffer(8);
        const src = new Float32Array([1, 2, 3]);
        buffer.push(src);
        const dst = new Float32Array(2);
        const popped = buffer.pop(dst);
        expect(popped).toBe(2);
        expect(dst).toEqual(new Float32Array([1, 2]));
        expect(buffer.available()).toBe(1);
    });

    test('pop respects available data', () => {
        const buffer = new Float32RingBuffer(8);
        const src = new Float32Array([1, 2]);
        buffer.push(src);
        const dst = new Float32Array(5);
        const popped = buffer.pop(dst);
        expect(popped).toBe(2);
        expect(dst.slice(0, 2)).toEqual(new Float32Array([1, 2]));
        expect(buffer.available()).toBe(0);
    });

    test('push and pop wrap around', () => {
        const buffer = new Float32RingBuffer(4);
        buffer.push(new Float32Array([1, 2, 3, 4]));
        const dst1 = new Float32Array(2);
        buffer.pop(dst1);
        expect(dst1).toEqual(new Float32Array([1, 2]));
        expect(buffer.available()).toBe(2);
        buffer.push(new Float32Array([5, 6]));
        expect(buffer.available()).toBe(4);
        const dst2 = new Float32Array(4);
        buffer.pop(dst2);
        expect(dst2).toEqual(new Float32Array([3, 4, 5, 6]));
    });

    test('push and pop many times', () => {
        const buffer = new Float32RingBuffer(8);
        for (let i = 0; i < 100; i++) {
            const src = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
            buffer.push(src);
            expect(buffer.available()).toEqual(7);
            const dst = new Float32Array(7);
            buffer.pop(dst);
            expect(dst).toEqual(src);
            expect(buffer.available()).toEqual(0);
        }
    });

    test('handles index overflow correctly', () => {
        const buffer = new Float32RingBuffer(8);

        // Access private fields for testing
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const bufferRef = (buffer as any).buffer as SharedArrayBuffer;
        const readIndex = new Int32Array(bufferRef, 0, 1);
        const writeIndex = new Int32Array(bufferRef, 4, 1);

        // Simulate indices near 32-bit overflow. Set indices to MAX_INT - 3
        Atomics.store(readIndex, 0, 2147483645);
        Atomics.store(writeIndex, 0, 2147483645);
        expect(buffer.available()).toBe(0);

        // Push 7 elements - this should increment write index past MAX_INT
        const src = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
        expect(buffer.push(src)).toBe(7);
        expect(buffer.available()).toBe(7);
        expect(Atomics.load(writeIndex, 0)).toBeLessThan(0);

        const dst = new Float32Array(7);
        expect(buffer.pop(dst)).toBe(7);
        expect(dst).toEqual(new Float32Array([1, 2, 3, 4, 5, 6, 7]));
        expect(buffer.available()).toBe(0);

        expect(Atomics.load(readIndex, 0)).toBeLessThan(0);
    });

    test('waitAvailableAtLeastAsync resolves immediately', async () => {
        const buffer = new Float32RingBuffer(8);
        buffer.push(new Float32Array([1, 2, 3]));
        let resolved = false;
        await buffer.waitPushAsync(2).then(() => resolved = true);
        expect(resolved).toBe(true);
        
        resolved = false;
        await buffer.waitPushAsync(3).then(() => resolved = true);
        expect(resolved).toBe(true);
        
        resolved = false;
        await buffer.waitPushAsync(0).then(() => resolved = true);
        expect(resolved).toBe(true);
    });

    test('waitAvailableAtMostAsync resolves immediately', async () => {
        const buffer = new Float32RingBuffer(8);
        buffer.push(new Float32Array([1, 2, 3]));
        let resolved = false;
        await buffer.waitPopAsync(5).then(() => resolved = true);
        expect(resolved).toBe(true);
        
        resolved = false;
        await buffer.waitPopAsync(3).then(() => resolved = true);
        expect(resolved).toBe(true);
        
        resolved = false;
        await buffer.waitPopAsync(10).then(() => resolved = true);
        expect(resolved).toBe(true);
    });

    test('waitAvailableAtLeastAsync waits for push', async () => {
        const buffer = new Float32RingBuffer(8);
        buffer.push(new Float32Array([1, 2, 3]));
        let resolved = false;
        const promise = buffer.waitPushAsync(5).then(() => resolved = true);
        expect(resolved).toBe(false);
        
        buffer.push(new Float32Array([4]));
        expect(resolved).toBe(false);

        buffer.push(new Float32Array([5]));

        await promise;
        expect(resolved).toBe(true);
    });

    test('waitAvailableAtMostAsync waits for pop', async () => {
        const buffer = new Float32RingBuffer(8);
        buffer.push(new Float32Array([1, 2, 3, 4, 5]));
        let resolved = false;
        const promise = buffer.waitPopAsync(3).then(() => resolved = true);
        expect(resolved).toBe(false);
        
        buffer.pop(new Float32Array([4]));
        expect(resolved).toBe(false);

        buffer.pop(new Float32Array([5]));

        await promise;
        expect(resolved).toBe(true);
    });
});
