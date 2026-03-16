// Decode an audio file from blob to interleaved PCM array with sample rate. We must use a real AudioContext, not
// OfflineAudioContext, because we want to get data automatically resampled into playable sample rate.
import type { InterleavedAudio } from './types';

// Decode audio blob to interleaved PCM using AudioContext
export async function decodeAudioFromBlob(blob: Blob, audioContext: AudioContext): Promise<InterleavedAudio> {
    const audioData = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(audioData);

    const numChannels = audioBuffer.numberOfChannels;
    const samplesPerChannel = audioBuffer.length;
    const data = new Float32Array(numChannels * samplesPerChannel);

    for (let ch = 0; ch < numChannels; ch++) {
        const channelData = audioBuffer.getChannelData(ch);
        for (let i = 0; i < samplesPerChannel; i++) {
            data[i * numChannels + ch] = channelData[i];
        }
    }

    return {
        data: data,
        sampleRate: audioBuffer.sampleRate,
        numChannels: numChannels,
    };
}
