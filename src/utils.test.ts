import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { debounce, secondsToString } from './utils';

describe('debounce', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    test('delays execution of sync function', () => {
        const mockFn = vi.fn();
        const debounced = debounce(100, mockFn);

        debounced();
        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(99);
        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(1);
        expect(mockFn).toHaveBeenCalledTimes(1);
    });

    test('delays execution of async function', async () => {
        const mockFn = vi.fn().mockResolvedValue(12345);
        const debounced = debounce(100, mockFn);

        const promise = debounced();
        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(100);
        await promise;
        expect(mockFn).toHaveBeenCalledTimes(1);
    });

    test('only executes the last call when called multiple times', () => {
        const mockFn = vi.fn();
        const debounced = debounce(100, mockFn);

        debounced('first');
        vi.advanceTimersByTime(50);
        debounced('second');
        vi.advanceTimersByTime(50);
        debounced('third');

        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(100);
        expect(mockFn).toHaveBeenCalledTimes(1);
        expect(mockFn).toHaveBeenCalledWith('third');
    });

    test('returns a promise that resolves after execution', async () => {
        let executed = false;
        const debounced = debounce(100, () => {
            executed = true;
        });

        const promise = debounced();
        expect(executed).toBe(false);

        vi.advanceTimersByTime(100);
        await promise;
        expect(executed).toBe(true);
    });

    test('awaits previous promise when new call happens', async () => {
        const mockFn = vi.fn();
        const debounced = debounce(100, mockFn);

        const promises = [];
        for (let i = 0; i < 1000; i++) {
            promises.push(debounced());
        }

        vi.advanceTimersByTime(100);
        for (const promise of promises) {
            await expect(promise).resolves.toBeUndefined();
        }
        expect(mockFn).toHaveBeenCalledTimes(1);
    });

    test('propagates errors from sync callback', async () => {
        const error = new Error('Test error');
        const debounced = debounce(100, () => {
            throw error;
        });

        const promise1 = debounced();
        const promise2 = debounced();
        vi.advanceTimersByTime(100);

        await expect(promise1).rejects.toBe(error);
        await expect(promise2).rejects.toBe(error);
    });

    test('propagates errors from async callback', async () => {
        const error = new Error('Async error');
        const debounced = debounce(100, async () => {
            throw error;
        });

        const promise1 = debounced();
        const promise2 = debounced();
        vi.advanceTimersByTime(100);

        await expect(promise1).rejects.toBe(error);
        await expect(promise2).rejects.toBe(error);
    });

    test('handles multiple arguments', () => {
        const mockFn = vi.fn();
        const debounced = debounce(100, mockFn);

        debounced(1, 'two', { three: 3 });
        vi.advanceTimersByTime(100);

        expect(mockFn).toHaveBeenCalledWith(1, 'two', { three: 3 });
    });

    test('new call during async execution does not hang', async () => {
        // Simulate an async callback that takes time to complete. A new debounced call arrives while the first callback
        // is still running. Both promises must settle (not hang forever).
        const { promise, resolve } = Promise.withResolvers<void>();
        let callCount = 0;
        const debounced = debounce(100, async () => {
            callCount++;
            if (callCount === 1) {
                await promise;
            }
        });

        // First call
        const p1 = debounced();
        vi.advanceTimersByTime(100);

        // While the first async callback is running, make a second call
        const p2 = debounced();
        vi.advanceTimersByTime(100);

        // Resolve the first async callback
        resolve();

        // Both promises should resolve without hanging
        await expect(p1).resolves.toBeUndefined();
        await expect(p2).resolves.toBeUndefined();
        expect(callCount).toBe(2);
    });
});

describe('secondsToString', () => {
    test('converts seconds to string with rounding', () => {
        expect(secondsToString(1.4)).toBe('1s');
        expect(secondsToString(65.6)).toBe('1m6s');
    });

    test('handles zero seconds', () => {
        expect(secondsToString(0)).toBe('0s');
        expect(secondsToString(0.4)).toBe('0s');
        expect(secondsToString(0.5)).toBe('1s');
    });

    test('handles minutes only', () => {
        expect(secondsToString(60)).toBe('1m0s');
        expect(secondsToString(120)).toBe('2m0s');
        expect(secondsToString(59.9)).toBe('1m0s');
    });

    test('handles large seconds', () => {
        expect(secondsToString(3661)).toBe('61m1s'); // 1 hour 1 minute 1 second
    });
});
