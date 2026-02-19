// A CountDownLatch implementation based on SharedArrayBuffer. It allows one or more threads to wait until a set of
// operations being performed in other threads completes.
export class CountDownLatch {
    private buffer_: SharedArrayBuffer;
    private count_: Int32Array;

    // Create a CountDownLatch based on existing SharedArrayBuffer. The buffer must have an appropriate size, see
    // byteLengthForCount().
    constructor(buffer: SharedArrayBuffer) {
        CountDownLatch.validateBuffer(buffer);
        this.buffer_ = buffer;
        this.count_ = new Int32Array(this.buffer_, 0, 1);
    }

    // Create a new CountDownLatch with the given initial count.
    static withCount(initialCount: number): CountDownLatch {
        const buffer = new SharedArrayBuffer(CountDownLatch.byteLengthForCount());
        const array = new Int32Array(buffer);
        array[0] = initialCount;
        return new CountDownLatch(buffer);
    }

    // Compute byteLength of SharedArrayBuffer, which then can be passed to the constructor.
    static byteLengthForCount(): number {
        return 4;
    }

    private static validateBuffer(buffer: SharedArrayBuffer) {
        if (buffer.byteLength != 4) {
            throw new Error(`Invalid buffer size for CountDownLatch: ${buffer.byteLength}`);
        }
    }

    // Get the underlying SharedArrayBuffer.
    get buffer(): SharedArrayBuffer {
        return this.buffer_;
    }

    // Get the current count value.
    get count(): number {
        return Atomics.load(this.count_, 0);
    }

    // Decrement the count by the specified amount. If the count reaches zero, all waiting threads are notified. We
    // allow the counter to become negative.
    countDown(count: number = 1): void {
        if (count <= 0) {
            throw new Error(`Count value must be positive, is ${count}`);
        }

        const oldCount = Atomics.sub(this.count_, 0, count);
        if (oldCount <= count) {
            // Notify all waiting threads that the count has reached zero
            Atomics.notify(this.count_, 0);
        }
    }

    // Wait asynchronously until the count reaches zero. Returns immediately if the count is already zero.
    async waitAsync(): Promise<void> {
        while (true) {
            const currentCount = Atomics.load(this.count_, 0);
            if (currentCount <= 0) {
                return;
            }

            const result = Atomics.waitAsync(this.count_, 0, currentCount);
            if (result.async) {
                await result.value;
            }
        }
    }
}

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

const CLOSED_BIT = 1;
const INDEX_SHIFT = 1;

export class Float32RingBuffer {
    private capacity_: number;
    private buffer_: SharedArrayBuffer;
    private data: Float32Array;
    // Read and write index can be in [0, 2 * capacity) range, which allows using full capacity. The indices are stored
    // in the upper bits, while the lower bit (see INDEX_SHIFT and CLOSED_BIT) stores the closed status. We encode the
    // closed status in both readIndex and writeIndex so that we can notify both consumer and producers when the buffer
    // is closed.
    private readIndex: Int32Array;
    private writeIndex: Int32Array;

    // Create a Float32RingBuffer based on existing SharedArrayBuffer. The buffer must have an appropriate capacity,
    // see withCapacity() and byteLengthForCapacity().
    constructor(buffer: SharedArrayBuffer) {
        const capacity = (buffer.byteLength - 8) / 4;
        Float32RingBuffer.validateCapacity(capacity);
        this.capacity_ = capacity >>> 0;
        this.buffer_ = buffer;
        this.readIndex = new Int32Array(this.buffer_, 0, 1);
        this.writeIndex = new Int32Array(this.buffer_, 4, 1);
        this.data = new Float32Array(this.buffer_, 8, capacity);
    }

    // Create a Float32RingBuffer for given capacity.
    static withCapacity(capacity: number): Float32RingBuffer {
        const buffer = new SharedArrayBuffer(Float32RingBuffer.byteLengthForCapacity(capacity));
        return new Float32RingBuffer(buffer);
    }

    // Compute byteLength of SharedArrayBuffer, which then can be passed to the constructor.
    static byteLengthForCapacity(capacity: number): number {
        Float32RingBuffer.validateCapacity(capacity);
        return 8 + capacity * 4;
    }

    private static validateCapacity(capacity: number) {
        if (capacity <= 0 || capacity !== capacity >>> 0) {
            throw new Error(`Invalid capacity for Float32RingBuffer: ${capacity}`);
        }
    }

    // Check if the buffer is closed.
    get isClosed(): boolean {
        // We read readIndex because it is updated before writeIndex: if we have woken a thread up on writeIndex,
        // it should already see the new readIndex.
        return (Atomics.load(this.readIndex, 0) & CLOSED_BIT) !== 0;
    }

    // Get the underlying SharedArrayBuffer.
    get buffer(): SharedArrayBuffer {
        return this.buffer_;
    }

    // Get the capacity.
    get capacity(): number {
        return this.capacity_;
    }

    // Close the buffer, waking any waiting threads.
    close(): void {
        // We need to update both indices so that wait do not hang due to race condition.
        Atomics.or(this.readIndex, 0, CLOSED_BIT);
        Atomics.or(this.writeIndex, 0, CLOSED_BIT);
        Atomics.notify(this.readIndex, 0);
        Atomics.notify(this.writeIndex, 0);
    }

    // Append data from src and return the number of elements pushed. For closed buffers returns 0 immediately.
    push(src: Float32Array): number {
        if (this.isClosed) {
            return 0;
        }
        const readIdx = Atomics.load(this.readIndex, 0) >>> INDEX_SHIFT;
        const writeIdx = Atomics.load(this.writeIndex, 0) >>> INDEX_SHIFT;
        const used = this.getUsed(readIdx, writeIdx);
        const free = this.capacity_ - used;
        const toPush = Math.min(free, src.length);
        if (toPush === 0) {
            return 0;
        }
        let writePos = writeIdx;
        if (writePos >= this.capacity_) {
            writePos -= this.capacity_;
        }
        // The push should write one or two contiguous chunks, depending on whether the writePos is near the end of the
        // buffer.
        const firstChunkSize = Math.min(toPush, this.capacity_ - writePos);
        if (firstChunkSize > 0) {
            this.data.set(src.subarray(0, firstChunkSize), writePos);
        }
        const remaining = toPush - firstChunkSize;
        if (remaining > 0) {
            this.data.set(src.subarray(firstChunkSize, firstChunkSize + remaining), 0);
        }

        // We do not directly store the new writeIndex because we want to preserve the closed bit which may be
        // concurrently set.
        let writeIdxDiff = toPush;
        if (writeIdx + toPush >= this.capacity_ * 2) {
            writeIdxDiff -= this.capacity_ * 2;
        }
        Atomics.add(this.writeIndex, 0, writeIdxDiff << INDEX_SHIFT);
        Atomics.notify(this.writeIndex, 0);
        return toPush;
    }

    // Pop data to dst and return the number of elements popped.
    pop(dst: Float32Array): number {
        const readIdx = Atomics.load(this.readIndex, 0) >>> INDEX_SHIFT;
        const toPop = this.peekImpl(dst, readIdx);
        // We do not directly store the new readIndex because we want to preserve the closed bit which may be
        // concurrently set.
        let reaIdxDiff = toPop;
        if (readIdx + toPop >= this.capacity_ * 2) {
            reaIdxDiff -= this.capacity_ * 2;
        }
        Atomics.add(this.readIndex, 0, reaIdxDiff << INDEX_SHIFT);
        Atomics.notify(this.readIndex, 0);
        return toPop;
    }

    // Read data to dst and return the number of elements read, but do not move the position.
    peek(dst: Float32Array): number {
        const readIdx = Atomics.load(this.readIndex, 0) >>> INDEX_SHIFT;
        return this.peekImpl(dst, readIdx);
    }

    private peekImpl(dst: Float32Array, readIdx: number): number {
        const writeIdx = Atomics.load(this.writeIndex, 0) >>> INDEX_SHIFT;
        const used = this.getUsed(readIdx, writeIdx);
        const toPeek = Math.min(used, dst.length);
        if (toPeek === 0) {
            return 0;
        }
        let readPos = readIdx;
        if (readPos >= this.capacity_) {
            readPos -= this.capacity_;
        }
        const firstChunkSize = Math.min(toPeek, this.capacity_ - readPos);
        dst.set(this.data.subarray(readPos, readPos + firstChunkSize), 0);
        const remaining = toPeek - firstChunkSize;
        if (remaining > 0) {
            dst.set(this.data.subarray(0, remaining), firstChunkSize);
        }
        return toPeek;
    }

    // Return the number of elements available
    get available(): number {
        const readIdx = Atomics.load(this.readIndex, 0) >>> INDEX_SHIFT;
        const writeIdx = Atomics.load(this.writeIndex, 0) >>> INDEX_SHIFT;
        return this.getUsed(readIdx, writeIdx);
    }

    // Wait until there are at least n free elements in the buffer (blocks the producer).
    waitPush(n: number) {
        if (n > this.capacity_) {
            throw new Error(`waitPush(${n}) exceeds capacity ${this.capacity_}`);
        }
        while (true) {
            const readValue = Atomics.load(this.readIndex, 0);
            const readIdx = readValue >>> INDEX_SHIFT;
            const writeIdx = Atomics.load(this.writeIndex, 0) >>> INDEX_SHIFT;
            const used = this.getUsed(readIdx, writeIdx);
            const free = this.capacity_ - used;
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
        if (n > this.capacity_) {
            throw new Error(`waitPushAsync(${n}) exceeds capacity ${this.capacity_}`);
        }
        while (true) {
            const readValue = Atomics.load(this.readIndex, 0);
            const readIdx = readValue >>> INDEX_SHIFT;
            const writeIdx = Atomics.load(this.writeIndex, 0) >>> INDEX_SHIFT;
            const used = this.getUsed(readIdx, writeIdx);
            const free = this.capacity_ - used;
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
        if (n > this.capacity_) {
            throw new Error(`waitPop(${n}) exceeds capacity ${this.capacity_}`);
        }
        while (true) {
            const readIdx = Atomics.load(this.readIndex, 0) >>> INDEX_SHIFT;
            const writeValue = Atomics.load(this.writeIndex, 0);
            const writeIdx = writeValue >>> INDEX_SHIFT;
            const used = this.getUsed(readIdx, writeIdx);
            if (used >= n) {
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
        if (n > this.capacity_) {
            throw new Error(`waitPopAsync(${n}) exceeds capacity ${this.capacity_}`);
        }
        while (true) {
            const readIdx = Atomics.load(this.readIndex, 0) >>> INDEX_SHIFT;
            const writeValue = Atomics.load(this.writeIndex, 0);
            const writeIdx = writeValue >>> INDEX_SHIFT;
            const used = this.getUsed(readIdx, writeIdx);
            if (used >= n) {
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

    getUsed(readIdx: number, writeIdx: number): number {
        if (writeIdx >= readIdx) {
            return writeIdx - readIdx;
        } else {
            return this.capacity_ * 2 + writeIdx - readIdx;
        }
    }
}

// Write the contents of ring buffer into a Float32Array, backed by SharedBufferArray, until the ring buffer is closed.
export async function drainRingBuffer(buffer: Float32RingBuffer): Promise<Float32Array> {
    const chunkSize = (buffer.capacity / 4) >>> 0;
    const recordedChunks: Float32Array[] = [];
    while (!buffer.isClosed) {
        await buffer.waitPopAsync(chunkSize);
        // Drain everything currently available
        const available = buffer.available;
        if (available > 0) {
            const chunk = new Float32Array(available);
            buffer.pop(chunk);
            recordedChunks.push(chunk);
        }
    }
    const remaining = buffer.available;
    if (remaining > 0) {
        const chunk = new Float32Array(remaining);
        buffer.pop(chunk);
        recordedChunks.push(chunk);
    }

    const totalLength = recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (totalLength === 0) {
        console.log('No audio data recorded');
    }
    const sab = new SharedArrayBuffer(totalLength * 4);
    const result = new Float32Array(sab);
    let offset = 0;
    for (const chunk of recordedChunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    return result;
}

// Write the contents of input into ring buffer until either all input is pushed or ring buffer is closed.
export async function pushAllRingBuffer(input: Float32Array, buffer: Float32RingBuffer): Promise<void> {
    let pushed = 0;
    // Do not try to push the whole buffer at once.
    const chunkSize = (buffer.capacity / 4) >>> 0;
    while (pushed < input.length) {
        const toPush = Math.min(chunkSize, input.length - pushed);
        await buffer.waitPushAsync(toPush);
        const n = buffer.push(input.subarray(pushed, pushed + toPush));
        if (buffer.isClosed) {
            return;
        }
        if (n !== toPush && !buffer.isClosed) {
            throw new Error(`Internal error: unexpected push(${toPush}) result: ${n}`);
        }
        pushed += n;
    }
}
