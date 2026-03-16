import { describe, expect, test } from 'vitest';
import { PlayerProcessorQueue } from './player-processor-queue';

describe('PlayerProcessorQueue', () => {
    describe('constructor', () => {
        test('initializes with correct number of channels', () => {
            const queue = new PlayerProcessorQueue(2);
            expect(queue.length).toBe(0);
        });

        test('initializes with mono channel', () => {
            const queue = new PlayerProcessorQueue(1);
            expect(queue.length).toBe(0);
        });

        test('initializes with multi-channel', () => {
            const queue = new PlayerProcessorQueue(5);
            expect(queue.length).toBe(0);
        });
    });

    describe('push', () => {
        test('adds mono chunk correctly', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2, 3, 4]);
            queue.pushInterleaved(chunk);
            expect(queue.length).toBe(4);
        });

        test('adds stereo chunk correctly', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk = new Float32Array([1, 2, 3, 4, 5, 6]);
            queue.pushInterleaved(chunk);
            expect(queue.length).toBe(3);
        });

        test('adds multiple chunks correctly', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk1 = new Float32Array([1, 2, 3, 4]);
            const chunk2 = new Float32Array([5, 6, 7, 8, 9, 10]);
            queue.pushInterleaved(chunk1);
            queue.pushInterleaved(chunk2);
            expect(queue.length).toBe(5);
        });

        test('adds empty chunk', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk = new Float32Array(0);
            queue.pushInterleaved(chunk);
            expect(queue.length).toBe(0);
        });
    });

    describe('pop', () => {
        test('pops exact chunk size mono', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2, 3, 4]);
            queue.pushInterleaved(chunk);

            const outputChannels = [new Float32Array(4)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(0);
            expect(outputChannels[0]).toEqual(new Float32Array([1, 2, 3, 4]));
            expect(queue.length).toBe(0);
        });

        test('pops exact chunk size stereo', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk = new Float32Array([1, 2, 3, 4, 5, 6]);
            queue.pushInterleaved(chunk);

            const outputChannels = [new Float32Array(3), new Float32Array(3)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(0);
            expect(outputChannels[0]).toEqual(new Float32Array([1, 3, 5]));
            expect(outputChannels[1]).toEqual(new Float32Array([2, 4, 6]));
            expect(queue.length).toBe(0);
        });

        test('pops partial data when queue has less than requested', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2]);
            queue.pushInterleaved(chunk);

            const outputChannels = [new Float32Array(4)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(2);
            expect(outputChannels[0]).toEqual(new Float32Array([1, 2, 0, 0]));
            expect(queue.length).toBe(0);
        });

        test('pops across chunk boundaries', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk1 = new Float32Array([1, 2]);
            const chunk2 = new Float32Array([3, 4]);
            queue.pushInterleaved(chunk1);
            queue.pushInterleaved(chunk2);

            const outputChannels = [new Float32Array(3)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(0);
            expect(outputChannels[0]).toEqual(new Float32Array([1, 2, 3]));
            expect(queue.length).toBe(1);
        });

        test('pops with offset within chunk', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2, 3, 4, 5]);
            queue.pushInterleaved(chunk);

            // First pop 2 samples
            const output1 = [new Float32Array(2)];
            const remaining1 = queue.popNonInterleaved(output1);
            expect(remaining1).toBe(0);
            expect(output1[0]).toEqual(new Float32Array([1, 2]));
            expect(queue.length).toBe(3);

            // Then pop 2 more samples (should continue from offset)
            const output2 = [new Float32Array(2)];
            const remaining2 = queue.popNonInterleaved(output2);
            expect(remaining2).toBe(0);
            expect(output2[0]).toEqual(new Float32Array([3, 4]));
            expect(queue.length).toBe(1);
        });

        test('handles small chunks with multiple pops', () => {
            const queue = new PlayerProcessorQueue(2);

            // Push many small chunks
            for (let i = 0; i < 10; i++) {
                // Each chunk has 1 sample (2 channels)
                const chunk = new Float32Array([i * 2, i * 2 + 1]);
                queue.pushInterleaved(chunk);
            }

            expect(queue.length).toBe(10);

            // Pop in various sizes
            const output1 = [new Float32Array(3), new Float32Array(3)];
            const remaining1 = queue.popNonInterleaved(output1);
            expect(remaining1).toBe(0);
            expect(output1[0]).toEqual(new Float32Array([0, 2, 4]));
            expect(output1[1]).toEqual(new Float32Array([1, 3, 5]));
            expect(queue.length).toBe(7);

            const output2 = [new Float32Array(4), new Float32Array(4)];
            const remaining2 = queue.popNonInterleaved(output2);
            expect(remaining2).toBe(0);
            expect(output2[0]).toEqual(new Float32Array([6, 8, 10, 12]));
            expect(output2[1]).toEqual(new Float32Array([7, 9, 11, 13]));
            expect(queue.length).toBe(3);
        });

        test('pops with chunk smaller than requested output', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2]);
            queue.pushInterleaved(chunk);

            const outputChannels = [new Float32Array(5)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(3);
            expect(outputChannels[0]).toEqual(new Float32Array([1, 2, 0, 0, 0]));
            expect(queue.length).toBe(0);
        });

        test('pops with multiple chunks and partial last chunk', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk1 = new Float32Array([1, 2, 3, 4]);
            const chunk2 = new Float32Array([5, 6]);
            queue.pushInterleaved(chunk1);
            queue.pushInterleaved(chunk2);

            const outputChannels = [new Float32Array(4), new Float32Array(4)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(1);
            expect(outputChannels[0]).toEqual(new Float32Array([1, 3, 5, 0]));
            expect(outputChannels[1]).toEqual(new Float32Array([2, 4, 6, 0]));
            expect(queue.length).toBe(0);
        });

        test('pops zero samples', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk = new Float32Array([1, 2, 3, 4]);
            queue.pushInterleaved(chunk);

            const outputChannels = [new Float32Array(0), new Float32Array(0)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(0);
            expect(queue.length).toBe(2);
        });

        test('pops from empty queue', () => {
            const queue = new PlayerProcessorQueue(1);
            const outputChannels = [new Float32Array(3)];
            const remaining = queue.popNonInterleaved(outputChannels);

            expect(remaining).toBe(3); // All samples missing
            expect(outputChannels[0]).toEqual(new Float32Array([0, 0, 0]));
            expect(queue.length).toBe(0);
        });

        test('maintains correct state after multiple push/pop operations', () => {
            const queue = new PlayerProcessorQueue(2);

            // Push 3 samples
            queue.pushInterleaved(new Float32Array([1, 2, 3, 4, 5, 6]));
            expect(queue.length).toBe(3);

            // Pop 2 samples
            const output1 = [new Float32Array(2), new Float32Array(2)];
            const remaining1 = queue.popNonInterleaved(output1);
            expect(remaining1).toBe(0);
            expect(output1[0]).toEqual(new Float32Array([1, 3]));
            expect(output1[1]).toEqual(new Float32Array([2, 4]));
            expect(queue.length).toBe(1);

            // Push 2 more samples
            queue.pushInterleaved(new Float32Array([7, 8, 9, 10]));
            expect(queue.length).toBe(3);

            // Pop 3 samples
            const output2 = [new Float32Array(3), new Float32Array(3)];
            const remaining2 = queue.popNonInterleaved(output2);
            expect(remaining2).toBe(0);
            expect(output2[0]).toEqual(new Float32Array([5, 7, 9]));
            expect(output2[1]).toEqual(new Float32Array([6, 8, 10]));
            expect(queue.length).toBe(0);
        });
    });

    describe('read', () => {
        test('reads without removing data mono', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2, 3, 4]);
            queue.pushInterleaved(chunk);

            const output = new Float32Array(4);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([1, 2, 3, 4]));
            expect(queue.length).toBe(4);
        });

        test('reads without removing data stereo', () => {
            const queue = new PlayerProcessorQueue(2);
            const chunk = new Float32Array([1, 2, 3, 4, 5, 6]);
            queue.pushInterleaved(chunk);

            const output = new Float32Array(6);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
            expect(queue.length).toBe(3);
        });

        test('reads partial data when queue has less than requested', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2]);
            queue.pushInterleaved(chunk);

            const output = new Float32Array(4);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([1, 2, 0, 0]));
            expect(queue.length).toBe(2);
        });

        test('reads across chunk boundaries', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk1 = new Float32Array([1, 2]);
            const chunk2 = new Float32Array([3, 4, 5]);
            queue.pushInterleaved(chunk1);
            queue.pushInterleaved(chunk2);

            const output = new Float32Array(3);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([1, 2, 3]));
            expect(queue.length).toBe(5);
        });

        test('reads with offset within chunk', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2, 3, 4, 5]);
            queue.pushInterleaved(chunk);

            const outputChannels = [new Float32Array(2)];
            queue.popNonInterleaved(outputChannels);

            // Now read without removing
            const output = new Float32Array(3);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([3, 4, 5]));
            expect(queue.length).toBe(3);
        });

        test('reads from empty queue', () => {
            const queue = new PlayerProcessorQueue(1);
            const output = new Float32Array(3);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([0, 0, 0]));
            expect(queue.length).toBe(0);
        });

        test('reads with multiple channels', () => {
            const queue = new PlayerProcessorQueue(3);
            const chunk = new Float32Array([1, 2, 3, 4, 5, 6]);
            queue.pushInterleaved(chunk);

            const output = new Float32Array(6);
            queue.readNonInterleaved(output);

            expect(output).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
            expect(queue.length).toBe(2);
        });

        test('read does not affect subsequent pop', () => {
            const queue = new PlayerProcessorQueue(1);
            const chunk = new Float32Array([1, 2, 3, 4]);
            queue.pushInterleaved(chunk);

            const readOutput = new Float32Array(4);
            queue.readNonInterleaved(readOutput);
            expect(readOutput).toEqual(new Float32Array([1, 2, 3, 4]));
            expect(queue.length).toBe(4);

            const popOutput = [new Float32Array(4)];
            const remaining = queue.popNonInterleaved(popOutput);
            expect(remaining).toBe(0);
            expect(popOutput[0]).toEqual(new Float32Array([1, 2, 3, 4]));
            expect(queue.length).toBe(0);
        });
    });

    describe('edge cases', () => {
        test('handles chunk boundaries with exact alignment', () => {
            const queue = new PlayerProcessorQueue(2);

            // Push chunks that align perfectly with pop boundaries
            queue.pushInterleaved(new Float32Array([1, 2, 3, 4])); // 2 samples
            queue.pushInterleaved(new Float32Array([5, 6, 7, 8])); // 2 samples

            // Pop exactly one chunk's worth
            const output1 = [new Float32Array(2), new Float32Array(2)];
            const remaining1 = queue.popNonInterleaved(output1);

            expect(remaining1).toBe(0);
            expect(output1[0]).toEqual(new Float32Array([1, 3]));
            expect(output1[1]).toEqual(new Float32Array([2, 4]));
            expect(queue.length).toBe(2);

            // Pop the next chunk
            const output2 = [new Float32Array(2), new Float32Array(2)];
            const remaining2 = queue.popNonInterleaved(output2);

            expect(remaining2).toBe(0);
            expect(output2[0]).toEqual(new Float32Array([5, 7]));
            expect(output2[1]).toEqual(new Float32Array([6, 8]));
            expect(queue.length).toBe(0);
        });

        test('handles very small chunks (single sample)', () => {
            const queue = new PlayerProcessorQueue(2);
            const numSamples = 10;

            // Push each sample as its own chunk
            for (let i = 0; i < numSamples; i++) {
                queue.pushInterleaved(new Float32Array([i * 2, i * 2 + 1]));
            }

            expect(queue.length).toBe(numSamples);

            // Pop in various sizes
            const output1 = [new Float32Array(3), new Float32Array(3)];
            const remaining1 = queue.popNonInterleaved(output1);

            expect(remaining1).toBe(0);
            expect(output1[0]).toEqual(new Float32Array([0, 2, 4]));
            expect(output1[1]).toEqual(new Float32Array([1, 3, 5]));
            expect(queue.length).toBe(7);

            const output2 = [new Float32Array(4), new Float32Array(4)];
            const remaining2 = queue.popNonInterleaved(output2);

            expect(remaining2).toBe(0);
            expect(output2[0]).toEqual(new Float32Array([6, 8, 10, 12]));
            expect(output2[1]).toEqual(new Float32Array([7, 9, 11, 13]));
            expect(queue.length).toBe(3);

            const output3 = [new Float32Array(3), new Float32Array(3)];
            const remaining3 = queue.popNonInterleaved(output3);

            expect(remaining3).toBe(0);
            expect(output3[0]).toEqual(new Float32Array([14, 16, 18]));
            expect(output3[1]).toEqual(new Float32Array([15, 17, 19]));
            expect(queue.length).toBe(0);
        });
    });
});
