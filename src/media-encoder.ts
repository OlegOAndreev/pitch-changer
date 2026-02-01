import { registerMp3Encoder } from '@mediabunny/mp3-encoder';
import {
    AudioBufferSource,
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

if (!(await canEncodeAudio('mp3'))) {
    registerMp3Encoder();
}

export async function encodeToBlob(fileType: string, data: Float32Array, sampleRate: number): Promise<Blob> {
    let encodedData: ArrayBuffer;
    let mimeType: string;

    const startTime = performance.now();
    if (fileType === 'mp3') {
        encodedData = await encodeToData(data, new Mp3OutputFormat(), 'mp3', sampleRate);
        mimeType = 'audio/mpeg';
    } else if (fileType === 'ogg') {
        encodedData = await encodeToData(data, new OggOutputFormat(), 'vorbis', sampleRate);
        mimeType = 'audio/ogg';
    } else if (fileType === 'wav') {
        encodedData = await encodeToData(data, new WavOutputFormat(), 'pcm-f32', sampleRate);
        mimeType = 'audio/wav';
    } else {
        throw new Error('Unsupported file format');
    }
    console.log(`Encoded ${data.length} samples into ${fileType} in ${performance.now() - startTime}ms`);

    return new Blob([encodedData], { type: mimeType });
}

async function encodeToData(
    audioData: Float32Array,
    outputFormat: OutputFormat,
    codec: AudioCodec,
    sampleRate: number
): Promise<ArrayBuffer> {
    const output = new Output({
        format: outputFormat,
        target: new BufferTarget()
    });

    const audioBuffer = new AudioBuffer({
        length: audioData.length,
        numberOfChannels: 1,
        sampleRate: sampleRate
    });
    audioBuffer.getChannelData(0).set(audioData);

    const audioSource = new AudioBufferSource({
        codec: codec,
        bitrate: QUALITY_HIGH
    });

    output.addAudioTrack(audioSource);
    await output.start();
    await audioSource.add(audioBuffer);
    await output.finalize();

    return output.target.buffer!;
}
