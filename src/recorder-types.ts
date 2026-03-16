// Exported only for recorder-processor.ts
export const recorderProcessorName = 'recorder-processor';

// Message types for communication between Recorder and RecorderProcessor
export interface RecordedSamplesMessage {
    samples: Float32Array;
}
