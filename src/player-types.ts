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

export interface PlayerGetStatsMessage {
    type: 'getLatestSamples';
    numSamples: number;
}

export interface PlayerStopMessage {
    type: 'stop';
}

export type PlayerRequest = PlayerInitMessage | PlayerGetStatsMessage | PlayerStopMessage;

export interface PlaybackFinishedMessage {
    type: 'playbackFinished';
}

// This is a copy of PlayerStats from player.ts
export interface PlayerStats {
    latestSamples: Float32Array;
    numUnderruns: number;
}

export interface PlayerStatsMessage {
    type: 'latestSamples';
    stats: PlayerStats;
}

export type PlayerResponse = PlaybackFinishedMessage | PlayerStatsMessage;
