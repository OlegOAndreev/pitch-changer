import recorderProcessor from './recorder-processor.ts?worker&url';

import { drainRingBuffer, Float32RingBuffer } from './sync';
import type { InterleavedAudio } from './types';

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
            console.log(`Initializing recorder processor module`)
            await audioContext.audioWorklet.addModule(recorderProcessor);
            moduleInitialized = true;
        }
        return new Recorder(audioContext);
    }

    get isRecording(): boolean {
        return this.ringBuffer != null;
    }

    // Start recording, wait until stop() is called and return the full recording when stop() is called. This method can
    // be called from the main thread.
    async record(): Promise<InterleavedAudio> {
        if (this.isRecording) {
            throw new Error('Recorder is already recording');
        }

        // At least Chrome requires this check
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        const ringBuffer = Float32RingBuffer.withCapacity(RING_BUFFER_CAPACITY);
        this.ringBuffer = ringBuffer;

        const options: RecorderProcessorOptions = {
            ringBufferSab: ringBuffer.buffer
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
            const data = await drainRingBuffer(this.ringBuffer);
            this.ringBuffer = null;
            console.log(`Recorded ${data.length} audio samples`);
            return {
                data,
                sampleRate: this.audioContext.sampleRate,
                numChannels: 1
            };
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
