// Decode an audio file from blob to mono PCM array with sample rate, the result is backed by SharedArrayBuffer. We must
// use a real AudioContext, not OfflineAudioContext, because we want to get data automatically resampled into playable
// sample rate.
export async function decodeAudioFromBlob(blob: Blob, audioContext: AudioContext): Promise<{ data: Float32Array; sampleRate: number }> {
    const audioData = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(audioData);

    const resultSab = new SharedArrayBuffer(audioBuffer.length * 4);
    const result = new Float32Array(resultSab);
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
        const channelData = audioBuffer.getChannelData(ch);
        for (let i = 0; i < channelData.length; i++) {
            result[i] += channelData[i];
        }
    }
    for (let i = 0; i < result.length; i++) {
        result[i] /= audioBuffer.numberOfChannels;
    }

    return { data: result, sampleRate };
}
