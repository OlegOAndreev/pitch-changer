import recorderProcessorUrl from './recorder-processor.ts?url';

// Exported only for recorder-processor.ts
export const recorderProcessorName = 'recorder-processor';

let moduleInitialized = false;

export class Recorder {
    isRecording = false;
    recordingFinished: Promise<void> | null = null;
    audioContext: AudioContext;
    recordedChunks: Float32Array[] = [];
    audioWorkletNode: AudioWorkletNode | null = null;
    mediaStream: MediaStream | null = null;
    mediaStreamSourceNode: MediaStreamAudioSourceNode | null = null;

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

    async start() {
        if (this.isRecording) {
            console.log('Recorder is already started');
            return;
        }
        this.recordedChunks = [];

        // At least Chrome requires this
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        let resolveRecordingFinished: (() => void);
        this.recordingFinished = new Promise((resolve) => {
            resolveRecordingFinished = resolve;
        });
        this.audioWorkletNode = new AudioWorkletNode(this.audioContext, recorderProcessorName);
        this.audioWorkletNode.port.onmessage = (e: MessageEvent<Float32Array | null>) => {
            // Interpret null as 'flushed data', see RecorderProcessor.
            if (e.data === null) {
                resolveRecordingFinished();
            } else {
                this.recordedChunks.push(e.data);
            }
        }

        console.log('Requesting microphone access...');
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('Microphone access granted');

        this.mediaStreamSourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.mediaStreamSourceNode.connect(this.audioWorkletNode);
        this.isRecording = true;
        console.log('MediaStreamSource connected, recording started');
    }

    async stop(): Promise<Float32Array> {
        if (!this.isRecording) {
            console.log('Recorder is already stopped');
            return new Float32Array(0);
        }

        this.isRecording = false;

        this.audioWorkletNode!.port.postMessage('flush');
        await this.recordingFinished;

        // If we do not disconnect the MediaStreamAudioSourceNode, even if we stop the media stream tracks, it continues
        // sending data to connected nodes.
        this.mediaStreamSourceNode!.disconnect();
        this.mediaStreamSourceNode = null;
        this.mediaStream!.getTracks().forEach(track => track.stop());
        this.mediaStream = null;
        this.audioWorkletNode!.disconnect();
        this.audioWorkletNode = null;

        if (this.recordedChunks.length === 0) {
            console.log('No audio data recorded');
            return new Float32Array(0);
        }

        console.log(`Got recorded chunks ${this.recordedChunks.length}`);
        const totalLength = this.recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const result = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of this.recordedChunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        this.recordedChunks = [];

        console.log(`Recorded ${result.length} audio samples`);
        return result;
    }
}
