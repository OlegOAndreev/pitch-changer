import type { ProcessingMode } from './types';

// Message types for communication with the AudioProcessor
export interface WorkerParams {
    processingMode: ProcessingMode;
    pitchValue: number;
    sampleRate: number;
    numChannels: number;
    fftSize: number;
}

export interface WorkerInitMessage {
    type: 'init';
}

export interface WorkerInitResponse {
    type: 'initDone';
}

export interface SetClientPortMessage {
    type: 'setClientPort';
    clientPort: MessagePort;
}

export interface SetParamsMessage {
    type: 'setParams';
    params: WorkerParams;
}

export interface ResetRequest {
    type: 'resetRequest';
}

export interface ProcessSamplesRequest {
    type: 'processSamplesRequest';
    samples: Float32Array;
}

export interface FinishProcessRequest {
    type: 'finishRequest';
}

// This message is sent in response both to ProcessSamplesRequest and FinishProcessRequest
export interface ProcessSamplesResponse {
    type: 'processSamplesResponse';
    samples: Float32Array;
    finished: boolean;
}

export type AudioProcessorRequest = WorkerInitMessage | SetClientPortMessage | SetParamsMessage;
export type AudioProcessorResponse = WorkerInitResponse;
export type AudioProcessorClientRequest = ResetRequest | ProcessSamplesRequest | FinishProcessRequest;
