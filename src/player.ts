import playerProcessorUrl from './player-processor.ts?url';
import { CountDownLatch, Float32RingBuffer } from './sync';

// Exported only for player-processor.ts
export const playerProcessorName = 'player-processor';
export interface PlayerProcessorOptions {
    ringBufferSab: SharedArrayBuffer;
    latchSab: SharedArrayBuffer;
}

let moduleInitialized = false;

export class Player {
    readonly audioContext: AudioContext;
    private ringBuffer: Float32RingBuffer | null = null;
    private latch: CountDownLatch | null = null;

    private constructor(audioContext: AudioContext) {
        this.audioContext = audioContext;
    }

    // Workaround for lack of async constructors
    static async create(audioContext: AudioContext): Promise<Player> {
        if (!moduleInitialized) {
            console.log(`Initializing processor module ${playerProcessorUrl}`)
            await audioContext.audioWorklet.addModule(playerProcessorUrl);
            moduleInitialized = true;
        }
        return new Player(audioContext);
    }

    get isPlaying(): boolean {
        return this.ringBuffer != null;
    }

    async play(ringBuffer: Float32RingBuffer): Promise<void> {
        if (this.isPlaying) {
            return;
        }

        this.ringBuffer = ringBuffer;
        // At least Chrome requires this check
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        const latch = CountDownLatch.withCount(1);
        this.latch = latch;
        const options: PlayerProcessorOptions = {
            ringBufferSab: ringBuffer.buffer,
            latchSab: latch.buffer
        };
        const workletNode = new AudioWorkletNode(this.audioContext, playerProcessorName, {
            processorOptions: options
        });
        workletNode.connect(this.audioContext.destination);

        try {
            await latch.waitAsync();
        } finally {
            workletNode.disconnect();
            this.ringBuffer = null;
            this.latch = null;
        }
    }

    // Stops the playback, forcing the current record() call to complete.
    stop() {
        if (!this.isPlaying) {
            console.log('Player is already stopped');
            return;
        }
        this.ringBuffer!.close();
    }
}
