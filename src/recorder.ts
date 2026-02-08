import recorderProcessorUrl from './recorder-processor.ts?url';
import { drainRingBuffer, Float32RingBuffer } from './ring-buffer';

// Ring buffer capacity. 2^18 = 262144 samples ~= 5.5s at 48kHz.
const RING_BUFFER_CAPACITY = 1 << 18;

// Exported only for recorder-processor.ts
export const recorderProcessorName = 'recorder-processor';
export interface RecorderProcessorOptions {
    ringBufferSab: SharedArrayBuffer
}

let moduleInitialized = false;

// Recorder is a simple interface for recording mono PCM samples using web audio.
export class Recorder {
    readonly audioContext: AudioContext;
    private ringBuffer: Float32RingBuffer | null = null;

    private constructor(audioContext: AudioContext) {
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
        const audioWorkletNode = new AudioWorkletNode(this.audioContext, recorderProcessorName, {
            processorOptions: options
        });

        console.log('Requesting microphone access...');
        const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('Microphone access granted');

        const mediaStreamSourceNode = this.audioContext.createMediaStreamSource(mediaStream);
        mediaStreamSourceNode.connect(audioWorkletNode);

        console.log('Recording started');
        try {
            // Drain loop: collect data from the ring buffer until closed.
            const result = await drainRingBuffer(this.ringBuffer);
            this.ringBuffer = null;
            console.log(`Recorded ${result.length} audio samples`);
            return result;
        } finally {
            // If we do not disconnect the MediaStreamAudioSourceNode, even if we stop the media stream tracks, it continues
            // sending data to connected nodes.
            mediaStreamSourceNode.disconnect();
            mediaStream.getTracks().forEach(track => track.stop());
            // audioWorkletNode.disconnect();
        }
    }

    // Stops the current recording, forcing the current record() call to complete.
    stop(): void {
        if (!this.isRecording) {
            console.log('Recorder is already stopped');
            return;
        }
        this.ringBuffer!.close();
    }
}
