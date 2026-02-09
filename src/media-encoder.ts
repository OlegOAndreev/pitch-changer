import { registerMp3Encoder } from '@mediabunny/mp3-encoder';
import {
    AudioSample,
    AudioSampleSource,
    BufferTarget,
    canEncodeAudio,
    Mp3OutputFormat,
    OggOutputFormat,
    Output,
    OutputFormat,
    QUALITY_HIGH,
    WavOutputFormat,
    type AudioCodec
} from 'mediabunny';
import type { InterleavedAudio } from './types';

if (!(await canEncodeAudio('mp3'))) {
    registerMp3Encoder();
}

// Encode interleaved PCM data into a blob.
export async function encodeAudioToBlob(fileType: string, audio: InterleavedAudio): Promise<Blob> {
    let encodedData: ArrayBuffer;
    let mimeType: string;

    const startTime = performance.now();
    if (fileType === 'mp3') {
        encodedData = await encodeAudioToBuffer(audio, new Mp3OutputFormat(), 'mp3');
        mimeType = 'audio/mpeg';
    } else if (fileType === 'ogg') {
        encodedData = await encodeAudioToBuffer(audio, new OggOutputFormat(), 'vorbis');
        mimeType = 'audio/ogg';
    } else if (fileType === 'wav') {
        encodedData = await encodeAudioToBuffer(audio, new WavOutputFormat(), 'pcm-f32');
        mimeType = 'audio/wav';
    } else {
        throw new Error('Unsupported file format');
    }
    console.log(`Encoded ${audio.data.length} samples into ${fileType} in ${performance.now() - startTime}ms`);

    return new Blob([encodedData], { type: mimeType });
}

async function encodeAudioToBuffer(
    audio: InterleavedAudio,
    outputFormat: OutputFormat,
    codec: AudioCodec
): Promise<ArrayBuffer> {
    const output = new Output({
        format: outputFormat,
        target: new BufferTarget()
    });

    const numberOfFrames = audio.data.length / audio.numChannels;
    const audioSample = new AudioSample({
        format: 'f32', // Interleaved float32 format
        sampleRate: audio.sampleRate,
        numberOfChannels: audio.numChannels,
        numberOfFrames: numberOfFrames,
        timestamp: 0,
        data: audio.data
    });

    try {
        const audioSource = new AudioSampleSource({
            codec: codec,
            bitrate: QUALITY_HIGH
        });

        output.addAudioTrack(audioSource);
        await output.start();
        await audioSource.add(audioSample);
        await output.finalize();

        return output.target.buffer!;
    } finally {
        audioSample.close();
    }
}
