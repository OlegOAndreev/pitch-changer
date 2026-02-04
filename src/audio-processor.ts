import * as Comlink from 'comlink';

// Exported only for process-audio-worker
export type AudioProcessorWorker = {
    shiftPitch(audioData: Float32Array, sampleRate: number, pitchRatio: number): Float32Array;
    timeStretch(audioData: Float32Array, sampleRate: number, stretchRatio: number): Float32Array;
};

// Class for creating audio processor worker and managing async/await promises.
export class AudioProcessor {
    private worker: Worker;
    private proxy: Comlink.Remote<AudioProcessorWorker>;

    constructor() {
        this.worker = new Worker(new URL('./audio-processor-worker.ts', import.meta.url), {
            type: 'module'
        });
        this.proxy = Comlink.wrap<AudioProcessorWorker>(this.worker);
        console.log('Audio worker initialized with Comlink');
    }

    async shiftPitch(audioData: Float32Array, sampleRate: number, pitchRatio: number): Promise<Float32Array> {
        // TODO: Check what happens with buffer.
        return await this.proxy.shiftPitch(audioData, sampleRate, pitchRatio);
    }

    async timeStretch(audioData: Float32Array, sampleRate: number, stretchRatio: number): Promise<Float32Array> {
        // TODO: Check what happens with buffer.
        return await this.proxy.timeStretch(audioData, sampleRate, stretchRatio);
    }
}
