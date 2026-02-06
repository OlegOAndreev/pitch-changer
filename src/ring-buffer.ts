// A simple implementation of SPSC ring buffer based on SharedArrayBuffer. Unlike the standard ringbuffer
// implementations, this implementation allows waking the writers when the readers read parts of buffer: this is
// important for implementing "refill on low watermark" pattern when a web worker generates the new data when buffer
// becomes near empty (but not completely empty to prevent underruns) and the audio worklet consumes the data.

export class Float32RingBuffer {
    private capacity: number;
    private buffer: SharedArrayBuffer;
    private data: Float32Array;
    private readIndex: Int32Array;
    private writeIndex: Int32Array;

    constructor(capacity: number) {
        // Enforce power-of-two capacity for efficient bitwise modulo operations
        if (capacity <= 0 || (capacity & (capacity - 1)) !== 0) {
            throw new Error(`Float32RingBuffer capacity must be a positive power of two, got ${capacity}`);
        }
        // Create SharedArrayBuffer with two int32 for read/write indices.
        this.capacity = capacity;
        this.buffer = new SharedArrayBuffer(8 + capacity * 4);
        this.readIndex = new Int32Array(this.buffer, 0, 1);
        this.writeIndex = new Int32Array(this.buffer, 4, 1);
        this.data = new Float32Array(this.buffer, 8, capacity);
    }

    // Append data from src and return the number of elements pushed.
    push(src: Float32Array): number {
        const read = Atomics.load(this.readIndex, 0);
        const write = Atomics.load(this.writeIndex, 0);
        const used = (write - read) >>> 0;
        const free = this.capacity - used;
        const toPush = Math.min(free, src.length);
        if (toPush === 0) {
            return 0;
        }
        const mask = this.capacity - 1;
        const writePos = write & mask;
        // The push should write one or two contiguous chunks, depending on whether the writePos is near the end of the
        // buffer.
        const firstChunkSize = Math.min(toPush, this.capacity - writePos);
        if (firstChunkSize > 0) {
            this.data.set(src.subarray(0, firstChunkSize), writePos);
        }
        const remaining = toPush - firstChunkSize;
        if (remaining > 0) {
            this.data.set(src.subarray(firstChunkSize, firstChunkSize + remaining), 0);
        }

        Atomics.store(this.writeIndex, 0, write + toPush);
        Atomics.notify(this.writeIndex, 0);
        return toPush;
    }

    // Pop data to dst and return the number of elements popped.
    pop(dst: Float32Array): number {
        const read = Atomics.load(this.readIndex, 0);
        const write = Atomics.load(this.writeIndex, 0);
        const used = (write - read) >>> 0;
        const toPop = Math.min(used, dst.length);
        if (toPop === 0) {
            return 0;
        }
        const mask = this.capacity - 1;
        const readPos = read & mask;
        const firstChunkSize = Math.min(toPop, this.capacity - readPos);
        dst.set(this.data.subarray(readPos, readPos + firstChunkSize), 0);
        const remaining = toPop - firstChunkSize;
        if (remaining > 0) {
            dst.set(this.data.subarray(0, remaining), firstChunkSize);
        }
        Atomics.store(this.readIndex, 0, read + toPop);
        Atomics.notify(this.readIndex, 0);
        return toPop;
    }

    // Return the number of elements available
    available(): number {
        const read = Atomics.load(this.readIndex, 0);
        const write = Atomics.load(this.writeIndex, 0);
        return (write - read) >>> 0;
    }

    // Wait until there is at least n elements available in the buffer.
    waitAvailableAtLeast(n: number) {
        while (true) {
            const read = Atomics.load(this.readIndex, 0);
            const write = Atomics.load(this.writeIndex, 0);
            const avail = (write - read) >>> 0;
            if (avail >= n) {
                return;
            }
            Atomics.wait(this.writeIndex, 0, write);
        }
    }

    // Async version of waitAvailableAtLeast, requires a modern browser with support for Atomics.waitAsync
    async waitAvailableAtLeastAsync(n: number): Promise<void> {
        while (true) {
            const read = Atomics.load(this.readIndex, 0);
            const write = Atomics.load(this.writeIndex, 0);
            const avail = (write - read) >>> 0;
            if (avail >= n) {
                return;
            }
            await Atomics.waitAsync(this.writeIndex, 0, write);
        }
    }

    // Wait until there are no more than n elements available in the buffer.
    waitAvailableAtMost(n: number) {
        while (true) {
            const read = Atomics.load(this.readIndex, 0);
            const write = Atomics.load(this.writeIndex, 0);
            const available = (write - read) >>> 0;
            if (available <= n) {
                return;
            }
            Atomics.wait(this.readIndex, 0, read);
        }
    }

    // Async version of waitAvailableAtMost, requires a modern browser with support for Atomics.waitAsync
    async waitAvailableAtMostAsync(n: number): Promise<void> {
        while (true) {
            const read = Atomics.load(this.readIndex, 0);
            const write = Atomics.load(this.writeIndex, 0);
            const available = (write - read) >>> 0;
            if (available <= n) {
                return;
            }
            await Atomics.waitAsync(this.readIndex, 0, read);
        }
    }
}
