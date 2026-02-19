// Processing mode for audio transformation
export type ProcessingMode = 'pitch' | 'formant-preserving-pitch' | 'time';

// Multi-channel interleaved audio samples
export interface InterleavedAudio {
    data: Float32Array;
    sampleRate: number;
    numChannels: number;
}

// Calculate total duration in seconds of an interleaved audio buffer
export function getAudioSeconds(audio: InterleavedAudio): number {
    return audio.data.length / (audio.sampleRate * audio.numChannels);
}

// Calculate number of samples per channel in an interleaved audio buffer
export function getAudioLength(audio: InterleavedAudio): number {
    return audio.data.length / audio.numChannels;
}
