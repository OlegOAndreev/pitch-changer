import playerProcessorUrl from './player-processor.ts?url';
import { CountDownLatch, Float32RingBuffer } from './sync';

// Exported only for player-processor.ts
export const playerProcessorName = 'player-processor';
export interface PlayerProcessorOptions {
    ringBufferSab: SharedArrayBuffer;
    latchSab: SharedArrayBuffer;
    numChannels: number,
}

let moduleInitialized = false;

export class Player {
    readonly audioContext: AudioContext;
    private ringBuffer: Float32RingBuffer | null = null;
    private stoppedLatch: CountDownLatch | null = null;

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

    // Return true if playing is in progress.
    get isPlaying(): boolean {
        return this.ringBuffer != null;
    }

    // Play interleaved data form ring buffer. When the promise completes, playing has completed. The method completes
    // when either the buffer becomes closed or stoppedLatch is count down.
    //
    // Note: Player expectes ringBuffer to contain data in sample rate of AudioContext passed to constructor.
    async play(ringBuffer: Float32RingBuffer, numChannels: number): Promise<void> {
        if (this.isPlaying) {
            return;
        }

        this.ringBuffer = ringBuffer;
        // At least Chrome requires this check
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        this.stoppedLatch = CountDownLatch.withCount(1);
        const options: PlayerProcessorOptions = {
            ringBufferSab: ringBuffer.buffer,
            latchSab: this.stoppedLatch.buffer,
            numChannels: numChannels,
        };
        const workletNode = new AudioWorkletNode(this.audioContext, playerProcessorName, {
            processorOptions: options,
            outputChannelCount: [numChannels],
        });
        workletNode.connect(this.audioContext.destination);

        try {
            // The latch is triggered either by PlayerProcessor or stop() method.
            await this.stoppedLatch.waitAsync();
        } finally {
            workletNode.disconnect();
            this.ringBuffer = null;
            this.stoppedLatch = null
        }
    }

    // Stop the playback, forcing the current record() call to complete.
    stop() {
        if (!this.isPlaying) {
            console.log('Player is already stopped');
            return;
        }
        this.ringBuffer!.close();
        this.stoppedLatch!.countDown();
    }
}
