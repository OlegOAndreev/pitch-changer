import { logError } from './common-utils';
import recorderProcessor from './recorder-processor.ts?worker&url';
import { recorderProcessorName, type RecordedSamplesMessage } from './recorder-types';
import type { InterleavedAudio } from './types';
import { concatArrays } from './utils';

let moduleInitialized = false;

// Recorder is a simple interface for recording mono PCM samples using web audio.
export class Recorder {
    readonly audioContext: AudioContext;
    private audioWorkletNode: AudioWorkletNode | null = null;
    private mediaStreamSourceNode: MediaStreamAudioSourceNode | null = null;
    private mediaStream: MediaStream | null = null;
    private recordedChunks: Float32Array[] = [];
    private resolveRecording: ((audio: InterleavedAudio) => void) | null = null;
    private rejectRecording: ((error: Error) => void) | null = null;

    private constructor(audioContext: AudioContext) {
        this.audioContext = audioContext;
    }

    // Workaround for lack of async constructors
    static async create(audioContext: AudioContext): Promise<Recorder> {
        if (!moduleInitialized) {
            console.log(`Initializing recorder processor module`);
            await audioContext.audioWorklet.addModule(recorderProcessor);
            moduleInitialized = true;
        }
        return new Recorder(audioContext);
    }

    get isRecording(): boolean {
        return this.audioWorkletNode !== null;
    }

    // Start recording, wait until stop() is called and return the full recording when stop() is called. This method can
    // be called from the main thread. onStartRecord is called after all the preparation is done, this may be used to
    // signal the user that the audio will actually be recorded.
    async record(onStartRecord: () => void): Promise<InterleavedAudio> {
        if (this.isRecording) {
            throw new Error('Recorder is already recording');
        }

        // At least Chrome requires this check
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Reset state
        this.recordedChunks = [];

        const promise = new Promise<InterleavedAudio>((res, rej) => {
            this.resolveRecording = res;
            this.rejectRecording = rej;
        });

        try {
            this.audioWorkletNode = new AudioWorkletNode(this.audioContext, recorderProcessorName);
            this.audioWorkletNode.onprocessorerror = (event: ErrorEvent) => {
                logError('RecorderProcessor', event);
            };

            this.audioWorkletNode.port.onmessage = (event: MessageEvent<RecordedSamplesMessage>) => {
                this.recordedChunks.push(event.data.samples);
            };

            console.log('Requesting microphone access...');
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('Microphone access granted');

            this.mediaStreamSourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.mediaStreamSourceNode.connect(this.audioWorkletNode!);

            console.log('Recording started');
        } catch (error) {
            this.rejectRecording!(error instanceof Error ? error : new Error(String(error)));
            this.cleanup();
        }

        onStartRecord();

        return promise;
    }

    // Stops the current recording, forcing the current record() call to complete.
    stop(): void {
        if (!this.isRecording) {
            console.log('Recorder is already stopped');
            return;
        }
        this.completeRecording();
    }

    // Complete the recording and resolve the promise
    private completeRecording(): void {
        if (!this.resolveRecording) {
            return;
        }

        try {
            const combinedData = concatArrays(this.recordedChunks);
            const audio: InterleavedAudio = {
                data: combinedData,
                sampleRate: this.audioContext.sampleRate,
                numChannels: 1,
            };

            console.log(`Recorded ${combinedData.length} audio samples`);
            this.resolveRecording(audio);
        } catch (error) {
            if (this.rejectRecording) {
                this.rejectRecording(error instanceof Error ? error : new Error(String(error)));
            }
        } finally {
            this.cleanup();
        }
    }

    private cleanup(): void {
        if (this.mediaStreamSourceNode) {
            this.mediaStreamSourceNode.disconnect();
            this.mediaStreamSourceNode = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach((track) => track.stop());
            this.mediaStream = null;
        }

        if (this.audioWorkletNode) {
            this.audioWorkletNode.disconnect();
            this.audioWorkletNode = null;
        }

        this.recordedChunks = [];
        this.resolveRecording = null;
        this.rejectRecording = null;
    }
}
