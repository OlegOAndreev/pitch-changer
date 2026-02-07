// Worker test for Float32RingBuffer using Node.js worker_threads. This test creates two workers (producer and consumer)
// that communicate via a Float32RingBuffer. This is a standalone test that doesn't use vitest or other test frameworks
// (run this test using npm run test:ring-buffer-worker). I do not currently know a better way to test the Atomic.wait
// (unlike Atomic.waitAsync).

/// <reference types="node" />

import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

import { Float32RingBuffer } from './ring-buffer.ts';

interface StartMessage {
    type: 'producer' | 'consumer',
    buffer: SharedArrayBuffer,
    totalElements: number,
}

interface DoneMessage {
    type: 'done';
    errors?: number;
    waitCount: number;
}

async function main(): Promise<void> {
    // Test with a small buffer to force synchronization
    const CAPACITY = 256;
    const TOTAL_ELEMENTS = 10000000;

    console.log(`Testing Float32RingBuffer with ${TOTAL_ELEMENTS} elements, capacity ${CAPACITY}`);
    const buffer = Float32RingBuffer.bufferForCapacity(CAPACITY);
    const producerWorker = new Worker(new URL(import.meta.url), {
        workerData: {
            type: 'producer',
            buffer: buffer,
            totalElements: TOTAL_ELEMENTS
        } as StartMessage
    });
    const consumerWorker = new Worker(new URL(import.meta.url), {
        workerData: {
            type: 'consumer',
            buffer: buffer,
            totalElements: TOTAL_ELEMENTS
        } as StartMessage
    });

    let producerDone = false;
    let consumerDone = false;
    let producerWaitCount = 0;
    let consumerWaitCount = 0;
    let consumerErrors = 0;
    function onFinish() {
        if (producerDone && consumerDone) {
            console.log(`Producer wait count: ${producerWaitCount}, consumer wait count ${consumerWaitCount}`);
            console.log(`Errors: ${consumerErrors}, result: ${consumerErrors === 0 ? 'PASS' : 'FAIL'}`);
            process.exit(consumerErrors === 0 ? 0 : 1);
        }
    }

    producerWorker.on('message', (msg: DoneMessage) => {
        producerWaitCount = msg.waitCount;
        producerDone = true;
        onFinish();
    });

    consumerWorker.on('message', (msg: DoneMessage) => {
        consumerWaitCount = msg.waitCount;
        consumerErrors = msg.errors!;
        consumerDone = true;
        onFinish();
    });

    producerWorker.on('error', (err: Error) => {
        console.error('Producer worker error:', err);
        process.exit(1);
    });

    consumerWorker.on('error', (err: Error) => {
        console.error('Consumer worker error:', err);
        process.exit(1);
    });
}

function worker() {
    const data: StartMessage = workerData;

    const ringBuffer = new Float32RingBuffer(data.buffer);
    const CHUNK_SIZE = 100;

    if (data.type === 'producer') {
        let waitCount = 0;

        const chunk = new Float32Array(CHUNK_SIZE);
        for (let i = 0; i < data.totalElements; i += CHUNK_SIZE) {
            const remaining = data.totalElements - i;
            const currentChunkSize = Math.min(CHUNK_SIZE, remaining);
            const currentChunk = chunk.subarray(0, currentChunkSize);
            for (let j = 0; j < currentChunkSize; j++) {
                currentChunk[j] = i + j;
            }

            let pushed = 0;
            while (pushed < currentChunkSize) {
                const slice = currentChunk.subarray(pushed);
                const n = ringBuffer.push(slice);
                if (n === 0) {
                    waitCount++;
                    ringBuffer.waitPush(slice.length / 2);
                } else {
                    pushed += n;
                }
            }
        }

        ringBuffer.close();

        parentPort!.postMessage({
            type: 'done',
            waitCount,
        } as DoneMessage);
    } else if (data.type === 'consumer') {
        let consumed = 0;
        let errors = 0;
        let waitCount = 0;

        const chunk = new Float32Array(CHUNK_SIZE);
        while (true) {
            let popped = 0;
            while (popped < CHUNK_SIZE) {
                const slice = chunk.subarray(popped);
                const n = ringBuffer.pop(slice);
                if (n === 0) {
                    if (ringBuffer.isClosed) {
                        break;
                    }
                    waitCount++;
                    ringBuffer.waitPop(slice.length / 2);
                } else {
                    popped += n;
                }
            }

            for (let j = 0; j < popped; j++) {
                const expected = consumed + j;
                if (chunk[j] !== expected) {
                    errors++;
                    if (errors < 10) {
                        console.error(`Mismatch at index ${expected}: expected ${expected}, got ${chunk[j]}`);
                    }
                }
            }
            consumed += popped;

            if (ringBuffer.isClosed) {
                break;
            }
        }

        if (consumed !== data.totalElements) {
            errors++;
            console.error(`Mismatched number of elements processed: expected ${data.totalElements}, got ${consumed}`);
        }

        parentPort!.postMessage({
            type: 'done',
            errors,
            waitCount,
        } as DoneMessage);
    }
}

if (isMainThread) {
    main().catch(err => {
        console.error('Test failed:', err);
        process.exit(1);
    });
} else {
    try {
        worker();
    } catch (err) {
        console.error('Worker error:', err);
    }
}
