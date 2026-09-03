// SamplesQueue stores incoming chunks of float32 data and allows popping and reading the data.
export class SamplesQueue {
    private numChannels: number;
    private chunks: Float32Array[] = [];
    private firstChunkOffset = 0;
    private totalSamples = 0;

    constructor(numChannels: number) {
        this.numChannels = numChannels;
    }

    // Pushes new chunk to the queue.
    pushInterleaved(chunk: Float32Array) {
        if (chunk.length % this.numChannels !== 0) {
            throw new Error(
                `Expected to chunk with full samples, ${chunk.length} not divisible by ${this.numChannels}`,
            );
        }
        this.chunks.push(chunk);
        this.totalSamples += chunk.length / this.numChannels;
    }

    // Pops data from the queue and deinterleaves it into multichannel arrays. Assumes that all outputChannels arrays
    // have the same length. The tail of the array is filled with zeros if there is not enough data, and the number of
    // written samples is returned.
    popNonInterleaved(outputChannels: Float32Array[]): number {
        if (outputChannels.length < this.numChannels) {
            throw new Error(`Expected to output at least to ${this.numChannels}, got ${outputChannels.length}`);
        }
        const numChannels = this.numChannels;

        let toPop = outputChannels[0].length;
        let chunkIdx = 0;
        let chunkOffset = this.firstChunkOffset;
        let outputOffset = 0;
        // console.log(
        //     `Need to pop ${toPop}, got queue length ${this.processedQueue.length} and ${this.processedQueueSamples}`,
        // );
        while (toPop > 0 && chunkIdx < this.chunks.length) {
            const chunk = this.chunks[chunkIdx];
            const chunkSamples = chunk.length / numChannels;
            const toPopChunk = Math.min(toPop, chunkSamples - chunkOffset);
            for (let i = 0; i < toPopChunk; i++) {
                for (let ch = 0; ch < numChannels; ch++) {
                    outputChannels[ch][i + outputOffset] = chunk[(i + chunkOffset) * numChannels + ch];
                }
            }
            toPop -= toPopChunk;
            chunkOffset += toPopChunk;
            outputOffset += toPopChunk;
            if (chunkOffset === chunkSamples) {
                chunkIdx++;
                chunkOffset = 0;
            }
        }
        this.chunks.splice(0, chunkIdx);
        this.firstChunkOffset = chunkOffset;
        const popped = outputChannels[0].length - toPop;
        this.totalSamples -= popped;

        for (let ch = 0; ch < this.numChannels; ch++) {
            outputChannels[ch].fill(0.0, outputOffset);
        }

        return popped;
    }

    // Skips up to n samples from the queue without reading the data. Returns the actual number of skipped samples,
    // which can be less than n if the queue does not have enough data.
    skip(n: number): number {
        if (n < 0) {
            return 0;
        }

        const numChannels = this.numChannels;
        let skipped = 0;
        let chunkIdx = 0;
        let chunkOffset = this.firstChunkOffset;
        while (skipped < n && chunkIdx < this.chunks.length) {
            const chunk = this.chunks[chunkIdx];
            const chunkSamples = chunk.length / numChannels;
            const toSkip = Math.min(n - skipped, chunkSamples - chunkOffset);
            skipped += toSkip;
            chunkOffset += toSkip;
            if (chunkOffset === chunkSamples) {
                chunkIdx++;
                chunkOffset = 0;
            }
        }
        this.chunks.splice(0, chunkIdx);
        this.firstChunkOffset = chunkOffset;
        this.totalSamples -= skipped;

        return skipped;
    }

    // Drops all data from the queue.
    clear() {
        this.chunks.length = 0;
        this.firstChunkOffset = 0;
        this.totalSamples = 0;
    }

    // Reads samples without removing them from the queue. It fills the output array with interleaved samples from the
    // queue (the tail of the array is filled with zeros if there is not enough data).
    readNonInterleaved(output: Float32Array) {
        if (output.length % this.numChannels !== 0) {
            throw new Error(`Expected to output full samples, ${output.length} not divisible by ${this.numChannels}`);
        }
        const numChannels = this.numChannels;

        let toRead = output.length / this.numChannels;
        let chunkIdx = 0;
        let chunkOffset = this.firstChunkOffset;
        let outputOffset = 0;
        while (toRead > 0 && chunkIdx < this.chunks.length) {
            const chunk = this.chunks[chunkIdx];
            const chunkSamples = chunk.length / numChannels;
            const toPopChunk = Math.min(toRead, chunkSamples - chunkOffset);
            output.set(
                chunk.subarray(chunkOffset * numChannels, (chunkOffset + toPopChunk) * numChannels),
                outputOffset * numChannels,
            );
            toRead -= toPopChunk;
            chunkOffset += toPopChunk;
            outputOffset += toPopChunk;
            if (chunkOffset === chunkSamples) {
                chunkIdx++;
                chunkOffset = 0;
            }
        }

        output.fill(0.0, outputOffset * numChannels);
    }

    get length(): number {
        return this.totalSamples;
    }
}
