import recorderProcessorUrl from './recorder-processor.ts?url';
import { Float32RingBuffer } from './ring-buffer';

// Ring buffer capacity. 2^18 = 262144 samples ~= 5.5s at 48kHz.
const RING_BUFFER_CAPACITY = 1 << 18;
// Incoming data is stored in chunks.
const MIN_CHUNK_SIZE = 4096;

// Exported only for recorder-processor.ts
export const recorderProcessorName = 'recorder-processor';
export interface RecorderProcessorOptions {
    ringBufferSab: SharedArrayBuffer
}

let moduleInitialized = false;

export class Recorder {
    readonly audioContext: AudioContext;
    private audioWorkletNode: AudioWorkletNode | null = null;
    private mediaStream: MediaStream | null = null;
    private mediaStreamSourceNode: MediaStreamAudioSourceNode | null = null;
    private ringBuffer: Float32RingBuffer | null = null;

    constructor(audioContext: AudioContext) {
        this.audioContext = audioContext;
    }

    // Workaround for lack of async constructors
    static async create(audioContext: AudioContext): Promise<Recorder> {
        if (!moduleInitialized) {
            console.log(`Initializing processor module ${recorderProcessorUrl}`)
            await audioContext.audioWorklet.addModule(recorderProcessorUrl);
            moduleInitialized = true;
        }
        return new Recorder(audioContext);
    }

    get isRecording(): boolean {
        return this.ringBuffer != null;
    }

    // Start recording, wait until stop() is called and return the full recording when stop() is called. This method can
    // be called from the main thread.
    async record(): Promise<Float32Array> {
        if (this.isRecording) {
            throw new Error('Recorder is already recording');
        }

        // At least Chrome requires this check
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        const ringBufferSab = Float32RingBuffer.bufferForCapacity(RING_BUFFER_CAPACITY);
        const ringBuffer = new Float32RingBuffer(ringBufferSab);
        this.ringBuffer = ringBuffer;

        const options: RecorderProcessorOptions = {
            ringBufferSab: ringBufferSab
        };
        this.audioWorkletNode = new AudioWorkletNode(this.audioContext, recorderProcessorName, {
            processorOptions: options
        });

        console.log('Requesting microphone access...');
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('Microphone access granted');

        this.mediaStreamSourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.mediaStreamSourceNode.connect(this.audioWorkletNode);

        console.log('Recording started');

        // Drain loop: collect data from the ring buffer until closed.
        const recordedChunks: Float32Array[] = [];
        while (!ringBuffer.isClosed) {
            await ringBuffer.waitPopAsync(MIN_CHUNK_SIZE);
            // Drain everything currently available
            const available = ringBuffer.available;
            if (available > 0) {
                const chunk = new Float32Array(available);
                ringBuffer.pop(chunk);
                recordedChunks.push(chunk);
            }
        }
        const remaining = ringBuffer.available;
        if (remaining > 0) {
            const chunk = new Float32Array(remaining);
            ringBuffer.pop(chunk);
            recordedChunks.push(chunk);
        }
        this.ringBuffer = null;

        const result = this.chunksToResult(recordedChunks);
        console.log(`Recorded ${result.length} audio samples`);
        return result;
    }

    stop(): void {
        if (!this.isRecording) {
            console.log('Recorder is already stopped');
            return;
        }
        this.ringBuffer!.close();

        // If we do not disconnect the MediaStreamAudioSourceNode, even if we stop the media stream tracks, it continues
        // sending data to connected nodes.
        this.mediaStreamSourceNode!.disconnect();
        this.mediaStreamSourceNode = null;
        this.mediaStream!.getTracks().forEach(track => track.stop());
        this.mediaStream = null;
        this.audioWorkletNode!.disconnect();
        this.audioWorkletNode = null;
    }

    // Accumulate chunks into a single Float32Array backed by SharedArrayBuffer.
    private chunksToResult(recordedChunks: Float32Array<ArrayBufferLike>[]): Float32Array {
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
}
