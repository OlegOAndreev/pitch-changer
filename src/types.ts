// Multi-channel interleaved audio samples
export interface InterleavedAudio {
    data: Float32Array,
    sampleRate: number,
    numChannels: number,
}

export function getAudioSeconds(audio: InterleavedAudio): number {
    return audio.data.length / (audio.sampleRate * audio.numChannels);
}

export function getAudioLength(audio: InterleavedAudio): number {
    return audio.data.length / audio.numChannels;
}
