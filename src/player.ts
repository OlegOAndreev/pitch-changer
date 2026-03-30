// Player starts a Worker for processing audio samples and an AudioWorklet for playing them.
//   1. Player starts an AudioProcessorWorker in the constructor
//   2. On play() the Player copies the data into PlayerProcessor
//   3. The PlayerProcessor keeps a buffer of processed audio and requests new data from the worker via clientPort
//
// Why we use this design:
//   * we can't share data because we want to support non-cross-origin-isolated environments
//   * we can't transfer the source buffer away permanently because the user might click Save after Play
//   * we want low-latency processing and immediate parameter updates
//   * audio thread must not depend on UI thread scheduling

import { AudioProcessorManager } from './audio-processor';
import playerProcessorModule from './player-processor.ts?worker&url';
import {
    playerProcessorName,
    type PlayerGetStatsMessage,
    type PlayerInitMessage,
    type PlayerProcessorOptions,
    type PlayerResponse,
    type PlayerStats,
    type PlayerStopMessage,
} from './player-types';

import { logError } from './common-utils';
import type { InterleavedAudio, ProcessingMode } from './types';

export { type PlayerStats };

export class Player {
    private readonly audioContext: AudioContext;
    private readonly processorManager: AudioProcessorManager;

    private processingMode: ProcessingMode = 'pitch';
    private pitchValue = 1.0;
    private fftSize: number;

    private isPlaying = false;

    private inputNumChannels = 1;

    private workletNode: AudioWorkletNode | null = null;

    private playResolve: ((value?: void) => void) | null = null;
    private latestSamplesResolves: Array<(stats: PlayerStats) => void> = [];

    private constructor(audioContext: AudioContext, processorManager: AudioProcessorManager, fftSize: number) {
        this.audioContext = audioContext;
        this.processorManager = processorManager;
        this.fftSize = fftSize;
    }

    // Workaround for lack of async constructors.
    static async create(audioContext: AudioContext, fftSize: number): Promise<Player> {
        console.log('Initializing player processor module');
        await audioContext.audioWorklet.addModule(playerProcessorModule);
        const processorManager = await AudioProcessorManager.create();
        return new Player(audioContext, processorManager, fftSize);
    }

    get playing(): boolean {
        return this.isPlaying;
    }

    setParams(processingMode: ProcessingMode, pitchValue: number): void {
        this.processingMode = processingMode;
        this.pitchValue = pitchValue;

        if (!this.isPlaying) {
            return;
        }

        this.processorManager.setParams(
            this.processingMode,
            this.pitchValue,
            this.audioContext.sampleRate,
            this.inputNumChannels,
            this.fftSize,
        );
    }

    async play(input: InterleavedAudio): Promise<void> {
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        if (this.isPlaying) {
            console.error('play() called while playing');
            return;
        }

        this.isPlaying = true;
        this.inputNumChannels = input.numChannels;

        this.processorManager.setParams(
            this.processingMode,
            this.pitchValue,
            this.audioContext.sampleRate,
            this.inputNumChannels,
            this.fftSize,
        );

        // We create a new port for new worklet
        const clientPort = this.processorManager.createClientPort();

        const options: PlayerProcessorOptions = {
            numChannels: this.inputNumChannels,
            sampleRate: this.audioContext.sampleRate,
            bufferSamples: this.fftSize,
        };

        this.workletNode = new AudioWorkletNode(this.audioContext, playerProcessorName, {
            processorOptions: options,
            outputChannelCount: [this.inputNumChannels],
        });
        this.workletNode.onprocessorerror = (event: ErrorEvent) => {
            logError('PlayerProcessor', event);
        };

        this.workletNode.port.onmessage = (event: MessageEvent<PlayerResponse>) => {
            this.onWorkletMessage(event);
        };

        const initMessage: PlayerInitMessage = {
            type: 'init',
            input: input.data,
            clientPort,
        };
        this.workletNode.port.postMessage(initMessage, [clientPort]);

        this.workletNode.connect(this.audioContext.destination);

        const promise = new Promise<void>((res) => {
            this.playResolve = res;
        });
        try {
            await promise;
        } finally {
            this.cleanup();
        }
    }

    stop(): void {
        if (!this.isPlaying || !this.workletNode) {
            return;
        }

        this.finishPlay();
    }

    async getPlayerStats(numSamples: number): Promise<PlayerStats> {
        if (!this.isPlaying || !this.workletNode) {
            return {
                latestSamples: new Float32Array(0),
                numUnderruns: 0,
            };
        }

        let resolve: (value: PlayerStats) => void;
        const promise = new Promise<PlayerStats>((res) => {
            resolve = res;
        });
        this.latestSamplesResolves.push(resolve!);

        const message: PlayerGetStatsMessage = {
            type: 'getLatestSamples',
            numSamples: numSamples,
        };
        this.workletNode.port.postMessage(message);

        return promise;
    }

    private onWorkletMessage(event: MessageEvent<PlayerResponse>): void {
        const message = event.data;

        switch (message.type) {
            case 'latestSamples':
                while (this.latestSamplesResolves.length > 0) {
                    const resolve = this.latestSamplesResolves.pop();
                    resolve!(message.stats);
                }
                break;
            case 'playbackFinished':
                this.finishPlay();
                break;
        }
    }

    private finishPlay(): void {
        if (this.playResolve) {
            this.playResolve();
        } else {
            throw new Error('Impossible: playback finished without set futures');
        }
        this.playResolve = null;
    }

    private cleanup(): void {
        if (this.workletNode) {
            const stopMessage: PlayerStopMessage = {
                type: 'stop',
            };
            this.workletNode.port.postMessage(stopMessage);
            this.workletNode.port.close();
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        this.isPlaying = false;
    }
}
