// A simple implementation of Float32 SPSC ring buffer based on SharedArrayBuffer.
//
// Unlike the other ringbuffer implementations, this implementation:
//  * allows waking both the consumers when the new data is available and the producers when data is partially consumed
//  * contains async versions of waiting methods, which can be called from the main thread
//  * has close() method, which is a much cleaner way to pass closed state than out-of-band methods (Go channels served
//    as inspiration).
//
// These wait methods are important for implementing "refill on low watermark" pattern in audio worklets when a separate
// web worker generates the new data when buffer becomes e.g. half-empty (but not completely empty to prevent
// underruns).
export class Float32RingBuffer {
    // We encode the closed status in both readIndex and writeIndex so that we can notify both consumer and producers
    // when the buffer is closed.
    private static readonly CLOSED_BIT = 1;
    private static readonly INDEX_SHIFT = 1;

    private capacity: number;
    private buffer: SharedArrayBuffer;
    private data: Float32Array;
    private readIndex: Int32Array;
    private writeIndex: Int32Array;

    // Create a Float32RingBuffer based on existing SharedArrayBuffer. The buffer must have an appropriate capacity,
    // see bufferForCapacity() and byteLengthForCapacity().
    constructor(buffer: SharedArrayBuffer) {
        const capacity = (buffer.byteLength - 8) / 4;
        Float32RingBuffer.validateCapacity(capacity);
        this.capacity = capacity;
        this.buffer = buffer;
        this.readIndex = new Int32Array(this.buffer, 0, 1);
        this.writeIndex = new Int32Array(this.buffer, 4, 1);
        this.data = new Float32Array(this.buffer, 8, capacity);
    }

    // Create SharedArrayBuffer which is appropriate for passing to the constructor.
    static bufferForCapacity(capacity: number): SharedArrayBuffer {
        return new SharedArrayBuffer(Float32RingBuffer.byteLengthForCapacity(capacity));
    }

    // Compute byteLength of SharedArrayBuffer, which then can be passed to the constructor.
    static byteLengthForCapacity(capacity: number): number {
        Float32RingBuffer.validateCapacity(capacity);
        return 8 + capacity * 4;
    }

    private static validateCapacity(capacity: number) {
        if (capacity <= 0 || capacity !== capacity >>> 0 || (capacity & (capacity - 1)) !== 0) {
            throw new Error(`Invalid capacity for Float32RingBuffer: ${capacity}`);
        }
    }

    // Check if the buffer is closed.
    get isClosed(): boolean {
        // We read readIndex because it is updated before writeIndex: if we have woken a thread up on writeIndex,
        // it should already see the new readIndex. 
        return (Atomics.load(this.readIndex, 0) & Float32RingBuffer.CLOSED_BIT) !== 0;
    }

    // Close the buffer, waking any waiting threads.
    close(): void {
        // We need to update both indices so that wait do not hang due to race condition.
        Atomics.or(this.readIndex, 0, Float32RingBuffer.CLOSED_BIT);
        Atomics.or(this.writeIndex, 0, Float32RingBuffer.CLOSED_BIT);
        Atomics.notify(this.readIndex, 0);
        Atomics.notify(this.writeIndex, 0);
    }

    // Append data from src and return the number of elements pushed. For closed buffers returns 0 immediately.
    push(src: Float32Array): number {
        if (this.isClosed) {
            return 0;
        }
        const read = Atomics.load(this.readIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
        const write = Atomics.load(this.writeIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
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

        // We rely on wrapround here
        Atomics.add(this.writeIndex, 0, toPush << Float32RingBuffer.INDEX_SHIFT);
        Atomics.notify(this.writeIndex, 0);
        return toPush;
    }

    // Pop data to dst and return the number of elements popped.
    pop(dst: Float32Array): number {
        const read = Atomics.load(this.readIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
        const write = Atomics.load(this.writeIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
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

        // We rely on wrapround here
        Atomics.add(this.readIndex, 0, toPop << Float32RingBuffer.INDEX_SHIFT);
        Atomics.notify(this.readIndex, 0);
        return toPop;
    }

    // Return the number of elements available
    get available(): number {
        const readValue = Atomics.load(this.readIndex, 0);
        const writeValue = Atomics.load(this.writeIndex, 0);
        return (writeValue - readValue) >>> Float32RingBuffer.INDEX_SHIFT;
    }

    // Wait until there are at least n free elements in the buffer (blocks the producer).
    waitPush(n: number) {
        if (n > this.capacity) {
            throw new Error(`waitPush(${n}) exceeds capacity ${this.capacity}`);
        }
        while (true) {
            const readValue = Atomics.load(this.readIndex, 0);
            const read = readValue >>> Float32RingBuffer.INDEX_SHIFT;
            const write = Atomics.load(this.writeIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
            const used = (write - read) >>> 0;
            const free = this.capacity - used;
            if (free >= n) {
                return;
            }
            if (this.isClosed) {
                return;
            }
            Atomics.wait(this.readIndex, 0, readValue);
        }
    }

    // Async version of waitPush, requires a modern browser with support for Atomics.waitAsync
    async waitPushAsync(n: number): Promise<void> {
        if (n > this.capacity) {
            throw new Error(`waitPushAsync(${n}) exceeds capacity ${this.capacity}`);
        }
        while (true) {
            const readValue = Atomics.load(this.readIndex, 0);
            const read = readValue >>> Float32RingBuffer.INDEX_SHIFT;
            const write = Atomics.load(this.writeIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
            const used = (write - read) >>> 0;
            const free = this.capacity - used;
            if (free >= n) {
                return;
            }
            if (this.isClosed) {
                return;
            }
            const result = Atomics.waitAsync(this.readIndex, 0, readValue);
            if (result.async) {
                await result.value;
            }
        }
    }

    // Wait until there are at least n elements available in the buffer (blocks the consumer).
    waitPop(n: number) {
        if (n > this.capacity) {
            throw new Error(`waitPop(${n}) exceeds capacity ${this.capacity}`);
        }
        while (true) {
            const read = Atomics.load(this.readIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
            const writeValue = Atomics.load(this.writeIndex, 0);
            const write = writeValue >>> Float32RingBuffer.INDEX_SHIFT;
            const available = (write - read) >>> 0;
            if (available >= n) {
                return;
            }
            if (this.isClosed) {
                return;
            }
            Atomics.wait(this.writeIndex, 0, writeValue);
        }
    }

    // Async version of waitPop, requires a modern browser with support for Atomics.waitAsync
    async waitPopAsync(n: number): Promise<void> {
        if (n > this.capacity) {
            throw new Error(`waitPopAsync(${n}) exceeds capacity ${this.capacity}`);
        }
        while (true) {
            const read = Atomics.load(this.readIndex, 0) >>> Float32RingBuffer.INDEX_SHIFT;
            const writeValue = Atomics.load(this.writeIndex, 0);
            const write = writeValue >>> Float32RingBuffer.INDEX_SHIFT;
            const available = (write - read) >>> 0;
            if (available >= n) {
                return;
            }
            if (this.isClosed) {
                return;
            }
            const result = Atomics.waitAsync(this.writeIndex, 0, writeValue);
            if (result.async) {
                await result.value;
            }
        }
    }
}
