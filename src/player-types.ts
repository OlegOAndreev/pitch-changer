export const playerProcessorName = 'player-processor';

export interface PlayerProcessorOptions {
    numChannels: number;
    sampleRate: number;
    bufferSamples: number;
}

export interface PlayerInitMessage {
    type: 'init';
    input: Float32Array;
    clientPort: MessagePort;
}

export interface GetLatestSamplesMessage {
    type: 'getLatestSamples';
    numSamples: number;
}

export interface PlayerStopMessage {
    type: 'stop';
}

export type PlayerRequest = PlayerInitMessage | GetLatestSamplesMessage | PlayerStopMessage;

export interface PlaybackFinishedMessage {
    type: 'playbackFinished';
}

export interface LatestSamplesMessage {
    type: 'latestSamples';
    samples: Float32Array;
}

export type PlayerResponse = PlaybackFinishedMessage | LatestSamplesMessage;
