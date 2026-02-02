export class Player {
    readonly audioContext: AudioContext;
    isPlaying: boolean = false;
    playingBufferSource: AudioBufferSourceNode | null = null;
    onStop: (() => void) | null = null;

    constructor(audioContext: AudioContext) {
        this.audioContext = audioContext;
    }

    async play(data: Float32Array, sampleRate: number, onStop: () => void) {
        if (this.isPlaying) {
            stop();
        }

        // At least Chrome requires this
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        const audioBuffer = this.audioContext.createBuffer(1, data.length, sampleRate);
        audioBuffer.getChannelData(0).set(data);
        this.playingBufferSource = this.audioContext.createBufferSource();
        this.playingBufferSource.buffer = audioBuffer;
        this.playingBufferSource.connect(this.audioContext.destination);
        this.playingBufferSource.onended = (_e: Event) => {
            this.isPlaying = false;
            onStop();
        }
        this.playingBufferSource.start();
        this.onStop = onStop;
        this.isPlaying = true;

        console.log(`Playing ${audioBuffer.length} samples at ${audioBuffer.sampleRate}Hz`);
    }

    stop() {
        if (!this.isPlaying) {
            return;
        }
        this.isPlaying = false;
        // No need to disconnect the source: "The nodes will automatically get disconnected from the graph and will be
        // deleted when they have no more references" from https://webaudio.github.io/web-audio-api/#dynamic-lifetime-background
        this.playingBufferSource!.stop();
        this.playingBufferSource = null;

        this.onStop!();
    }
}
