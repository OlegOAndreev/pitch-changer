import { ALL_FORMATS, BufferSource, Input } from 'mediabunny';

// Decode an audio file from blob to mono PCM array with sample rate, the result is backed by SharedArrayBuffer.
export async function decodeAudioFromBlob(blob: Blob): Promise<{ data: Float32Array; sampleRate: number }> {
    // We use Mediabunny only for determining the sample rate (to prevent double resampling). Unfortunately Mediabunny
    // support fewer filetypes than AudioContext (at least on Chrome).
    const buffer = await blob.arrayBuffer();
    // Do not use BlobSource: it appears that Mediabunny starts trusting the mimetype, which can be misleading.
    const source = new BufferSource(buffer);
    const input = new Input({
        source,
        formats: ALL_FORMATS,
    });
    const mimeType = await input.getMimeType();
    console.log(`Detected mime type ${mimeType}`);

    try {
        const tracks = await input.getTracks();
        const audioTrack = tracks.find(track => track.isAudioTrack());
        if (!audioTrack) {
            throw new Error('No audio track found in file');
        }

        const sampleRate = audioTrack.sampleRate;
        const offlineCtx = new OfflineAudioContext({
            length: sampleRate,
            sampleRate: sampleRate,
        });
        const audioBuffer = await offlineCtx.decodeAudioData(buffer);

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
    } finally {
        input.dispose();
    }
}
