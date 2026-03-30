import { describe, expect, test } from 'vitest';
import { concatArrays, secondsToString } from './utils';

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

describe('concatArrays', () => {
    test('concatenates empty array', () => {
        const result = concatArrays([]);
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(0);
    });

    test('concatenates single array', () => {
        const input = [new Float32Array([1, 2, 3])];
        const result = concatArrays(input);
        expect(result).toEqual(new Float32Array([1, 2, 3]));
    });

    test('concatenates multiple arrays', () => {
        const input = [new Float32Array([1, 2]), new Float32Array([3, 4, 5]), new Float32Array([6])];
        const result = concatArrays(input);
        expect(result).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });

    test('concatenates arrays with empty arrays', () => {
        const input = [new Float32Array([1, 2]), new Float32Array([]), new Float32Array([3, 4]), new Float32Array([])];
        const result = concatArrays(input);
        expect(result).toEqual(new Float32Array([1, 2, 3, 4]));
    });
});
