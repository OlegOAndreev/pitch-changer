import { describe, expect, test } from 'vitest';
import { CountDownLatch, Float32RingBuffer, drainRingBuffer, pushAllRingBuffer } from './sync';

describe('CountDownLatch', () => {
    test('withCount creates latch with correct count', () => {
        const latch = CountDownLatch.withCount(5);
        expect(latch.count).toBe(5);
    });

    test('countDown decrements count', () => {
        const latch = CountDownLatch.withCount(5);

        latch.countDown(2);
        expect(latch.count).toBe(3);

        latch.countDown();
        expect(latch.count).toBe(2);
    });

    test('countDown throws on non-positive count', () => {
        const latch = CountDownLatch.withCount(5);

        expect(() => latch.countDown(0)).toThrow();
        expect(() => latch.countDown(-1)).toThrow();
    });

    test('waitAsync resolves immediately when count is already zero', async () => {
        const latch = CountDownLatch.withCount(0);

        await expect(latch.waitAsync()).resolves.toBeUndefined();
    });

    test('multiple countDown calls trigger waitAsync', async () => {
        const latch = CountDownLatch.withCount(3);

        const waitPromise = latch.waitAsync();

        latch.countDown();
        latch.countDown();
        latch.countDown();

        await expect(waitPromise).resolves.toBeUndefined();
    });

    test('constructor validates buffer size', () => {
        const smallBuffer = new SharedArrayBuffer(2);
        expect(() => new CountDownLatch(smallBuffer)).toThrow();

        const validBuffer = new SharedArrayBuffer(4);
        expect(() => new CountDownLatch(validBuffer)).not.toThrow();
    });
});

// Note: Testing actual blocking behavior requires multiple threads, which is beyond the scope of unit tests.
describe('Float32RingBuffer', () => {
    test('constructor', () => {
        const buffer = Float32RingBuffer.withCapacity(1024);
        expect(buffer.available).toBe(0);
    });

    test('push returns number of elements pushed', () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        const src = new Float32Array([1, 2, 3]);
        expect(buffer.push(src)).toBe(3);
        expect(buffer.available).toBe(3);
    });

    test('push respects capacity', () => {
        const buffer = Float32RingBuffer.withCapacity(4);
        const src1 = new Float32Array([1, 2, 3, 4]);
        expect(buffer.push(src1)).toBe(4);
        expect(buffer.available).toBe(4);
        // buffer is full, next push should push 0
        const src2 = new Float32Array([5, 6]);
        expect(buffer.push(src2)).toBe(0);
        expect(buffer.available).toBe(4);
    });

    test('pop returns number of elements popped', () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        const src = new Float32Array([1, 2, 3]);
        buffer.push(src);
        const dst = new Float32Array(2);
        const popped = buffer.pop(dst);
        expect(popped).toBe(2);
        expect(dst).toEqual(new Float32Array([1, 2]));
        expect(buffer.available).toBe(1);
    });

    test('pop respects available data', () => {
        const buffer = Float32RingBuffer.withCapacity(10);
        const src = new Float32Array([1, 2]);
        buffer.push(src);
        const dst = new Float32Array(5);
        const popped = buffer.pop(dst);
        expect(popped).toBe(2);
        expect(dst.slice(0, 2)).toEqual(new Float32Array([1, 2]));
        expect(buffer.available).toBe(0);
    });

    test('push and pop wrap around', () => {
        const buffer = Float32RingBuffer.withCapacity(7);
        buffer.push(new Float32Array([1, 2, 3, 4]));
        const dst1 = new Float32Array(2);
        buffer.pop(dst1);
        expect(dst1).toEqual(new Float32Array([1, 2]));
        expect(buffer.available).toBe(2);
        buffer.push(new Float32Array([5, 6]));
        expect(buffer.available).toBe(4);
        const dst2 = new Float32Array(4);
        buffer.pop(dst2);
        expect(dst2).toEqual(new Float32Array([3, 4, 5, 6]));
    });

    test('push and pop many times', () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        for (let i = 0; i < 100; i++) {
            const src = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
            buffer.push(src);
            expect(buffer.available).toEqual(7);
            const dst = new Float32Array(7);
            buffer.pop(dst);
            expect(dst).toEqual(src);
            expect(buffer.available).toEqual(0);
        }
    });

    test('waitPushAsync resolves immediately when enough free space', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        buffer.push(new Float32Array([1, 2, 3]));
        // 5 free slots, requesting 2 — should resolve immediately
        let resolved = false;
        await buffer.waitPushAsync(2).then(() => resolved = true);
        expect(resolved).toBe(true);

        // 5 free slots, requesting 5 — should resolve immediately
        resolved = false;
        await buffer.waitPushAsync(5).then(() => resolved = true);
        expect(resolved).toBe(true);

        // requesting 0 — should resolve immediately
        resolved = false;
        await buffer.waitPushAsync(0).then(() => resolved = true);
        expect(resolved).toBe(true);
    });

    test('waitPopAsync resolves immediately when enough data available', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        buffer.push(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
        let resolved = false;
        await buffer.waitPopAsync(5).then(() => resolved = true);
        expect(resolved).toBe(true);

        resolved = false;
        await buffer.waitPopAsync(3).then(() => resolved = true);
        expect(resolved).toBe(true);

        resolved = false;
        await buffer.waitPopAsync(8).then(() => resolved = true);
        expect(resolved).toBe(true);
    });

    test('waitPush throws if n exceeds capacity', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        expect(() => buffer.waitPush(9)).toThrow('exceeds capacity');
        await expect(buffer.waitPushAsync(9)).rejects.toThrow('exceeds capacity');
    });

    test('waitPop throws if n exceeds capacity', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        expect(() => buffer.waitPop(9)).toThrow('exceeds capacity');
        await expect(buffer.waitPopAsync(9)).rejects.toThrow('exceeds capacity');
    });

    test('waitPushAsync waits until enough free space', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        buffer.push(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
        let resolved = false;
        const promise = buffer.waitPushAsync(3).then(() => resolved = true);
        await Promise.resolve();
        expect(resolved).toBe(false);

        buffer.pop(new Float32Array(2));
        await Promise.resolve();
        expect(resolved).toBe(false);

        buffer.pop(new Float32Array(1));

        await promise;
        expect(resolved).toBe(true);
    });

    test('waitPopAsync waits until enough data available', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        let resolved = false;
        const promise = buffer.waitPopAsync(3).then(() => resolved = true);
        await Promise.resolve();
        expect(resolved).toBe(false);

        buffer.push(new Float32Array([1, 2]));
        await Promise.resolve();
        expect(resolved).toBe(false);

        buffer.push(new Float32Array([3]));

        await promise;
        expect(resolved).toBe(true);
    });

    test('close method sets closed flag', () => {
        const buffer = Float32RingBuffer.withCapacity(9);
        expect(buffer.isClosed).toBe(false);
        buffer.close();
        expect(buffer.isClosed).toBe(true);
    });

    test('push returns 0 when closed', () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        const src = new Float32Array([1, 2, 3]);
        expect(buffer.push(src)).toBe(3);
        buffer.close();
        expect(buffer.push(src)).toBe(0);
    });

    test('waitPush returns immediately when closed', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        buffer.push(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
        buffer.close();
        // Should return immediately without throwing
        expect(() => buffer.waitPush(1)).not.toThrow();
        // Should resolve immediately without throwing
        await expect(buffer.waitPushAsync(1)).resolves.toBeUndefined();
    });

    test('waitPop returns immediately when closed', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        buffer.close();
        // Should return immediately without throwing
        expect(() => buffer.waitPop(1)).not.toThrow();
        // Should resolve immediately without throwing
        await expect(buffer.waitPopAsync(1)).resolves.toBeUndefined();
    });

    test('close wakes waitPush', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        buffer.push(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
        // buffer is full, waitPush will block
        const waitPromise = buffer.waitPushAsync(1);
        buffer.close();
        // Should resolve (not reject) because closed causes immediate return
        await expect(waitPromise).resolves.toBeUndefined();
    });

    test('close wakes waitPop', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        // buffer is empty, waitPop will block
        const waitPromise = buffer.waitPopAsync(1);
        buffer.close();
        // Should resolve (not reject) because closed causes immediate return
        await expect(waitPromise).resolves.toBeUndefined();
    });

    test('pop still works after close', () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        const src = new Float32Array([1, 2, 3]);
        buffer.push(src);
        buffer.close();
        expect(buffer.available).toBe(3);
        const dst = new Float32Array(3);
        expect(buffer.pop(dst)).toBe(3);
        expect(dst).toEqual(new Float32Array([1, 2, 3]));
        expect(buffer.available).toBe(0);
    });

    test('drainRingBuffer collects all data', async () => {
        const buffer = Float32RingBuffer.withCapacity(16);

        const data1 = new Float32Array([1, 2, 3, 4]);
        const data2 = new Float32Array([5, 6, 7, 8]);
        buffer.push(data1);
        buffer.push(data2);

        const drainPromise = drainRingBuffer(buffer);

        const data3 = new Float32Array([9, 10, 11, 12]);
        buffer.push(data3);
        buffer.close();

        const result = await drainPromise;

        expect(result.length).toBe(12);
        expect(Array.from(result)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        expect(buffer.available).toBe(0);
    });

    test('pushAllRingBuffer handles data larger than capacity', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);

        const pushPromise = pushAllRingBuffer(input, buffer);

        // While pushAllRingBuffer is running, we need to pop data to free up space
        const poppedData: number[] = [];
        async function consumeBuffer() {
            while (poppedData.length < input.length) {
                await buffer.waitPopAsync(1);
                const chunk = new Float32Array(4); // Pop in smaller chunks
                const popped = buffer.pop(chunk);
                for (let i = 0; i < popped; i++) {
                    poppedData.push(chunk[i]);
                }
            }
        }
        const consumePromise = consumeBuffer();

        await Promise.all([pushPromise, consumePromise]);

        expect(poppedData.length).toBe(17);
        expect(poppedData).toEqual(Array.from(input));
        expect(buffer.available).toBe(0);
    });

    test('pushAllRingBuffer stops when buffer is closed', async () => {
        const buffer = Float32RingBuffer.withCapacity(8);
        const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        const pushPromise = pushAllRingBuffer(input, buffer);

        const firstChunk = new Float32Array([1, 2, 3, 4]);
        buffer.push(firstChunk);
        
        buffer.close();
        await expect(pushPromise).resolves.toBeUndefined();

        expect(buffer.available).toBe(4);
        const output = new Float32Array(4);
        const popped = buffer.pop(output);
        expect(popped).toBe(4);
        expect(Array.from(output)).toEqual([1, 2, 3, 4]);
    });

    test('pushAllRingBuffer/drainRingBuffer concurrency', async () => {
        const buffer = Float32RingBuffer.withCapacity(15);

        // CHUNK_SIZE must divide ITERATIONS
        const ITERATIONS = 200000;
        const CHUNK_SIZE = 100; // CHUNK_SIZE is greater than buffer capacity
        async function producer(): Promise<void> {
            const chunk = new Float32Array(CHUNK_SIZE);
            for (let i = 0; i < ITERATIONS; i += CHUNK_SIZE) {
                for (let j = 0; j < CHUNK_SIZE; j++) {
                    chunk[j] = i + j;
                }
                await pushAllRingBuffer(chunk, buffer);
            }
            buffer.close();
        }

        const producerPromise = producer();
        const consumerPromise = drainRingBuffer(buffer);
        await producerPromise;
        const consumed = await consumerPromise;
        expect(consumed.length).toBe(ITERATIONS);
        for (let i = 0; i < ITERATIONS; i++) {
            expect(consumed[i]).toBe(i);
        }
    });
});
