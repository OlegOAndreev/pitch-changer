import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { debounce, getBaseName } from './common-utils';

describe('getBaseName', () => {
    it('returns filename without extension and path', () => {
        expect(getBaseName('/path/to/file.txt')).toBe('file');
        expect(getBaseName('file.txt')).toBe('file');
        expect(getBaseName('/path/to/file')).toBe('file');
        expect(getBaseName('file')).toBe('file');
    });

    it('handles multiple dots in filename', () => {
        expect(getBaseName('file.min.js')).toBe('file.min');
        expect(getBaseName('/path/to/file.min.js')).toBe('file.min');
    });

    it('handles empty string', () => {
        expect(getBaseName('')).toBe('');
    });

    it('handles dot at start', () => {
        expect(getBaseName('.gitignore')).toBe('.gitignore');
        expect(getBaseName('/path/.gitignore')).toBe('.gitignore');
    });

    it('handles trailing slash', () => {
        expect(getBaseName('/path/to/')).toBe('to');
        expect(getBaseName('/')).toBe('');
    });
});

describe('debounce', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('delays execution of sync function', () => {
        const mockFn = vi.fn();
        const debounced = debounce(100, mockFn);

        debounced();
        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(99);
        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(1);
        expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('delays execution of async function', async () => {
        const mockFn = vi.fn().mockResolvedValue(12345);
        const debounced = debounce(100, mockFn);

        const promise = debounced();
        expect(mockFn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(100);
        await promise;
        expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('only executes the last call when called multiple times', () => {
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

    it('returns a promise that resolves after execution', async () => {
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

    it('awaits previous promise when new call happens', async () => {
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

    it('propagates errors from sync callback', async () => {
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

    it('propagates errors from async callback', async () => {
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

    it('handles multiple arguments', () => {
        const mockFn = vi.fn();
        const debounced = debounce(100, mockFn);

        debounced(1, 'two', { three: 3 });
        vi.advanceTimersByTime(100);

        expect(mockFn).toHaveBeenCalledWith(1, 'two', { three: 3 });
    });

    it('new call during async execution does not hang', async () => {
        // Simulate an async callback that takes time to complete. A new debounced call arrives while the first callback
        // is still running. Both promises must settle (not hang forever).
        let resolve;
        const promise = new Promise<void>((res) => {
            resolve = res;
        });
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
        resolve!();

        // Both promises should resolve without hanging
        await expect(p1).resolves.toBeUndefined();
        await expect(p2).resolves.toBeUndefined();
        expect(callCount).toBe(2);
    });
});
