import { PROCESSOR_NAME } from './common.js';

class PitchProcessor extends AudioWorkletProcessor {
    static get parameterDescriptors() {
        return [
            {
                name: 'pitchValue',
                defaultValue: 1.0,
                minValue: 0.0,
                maxValue: 10.0,
                automationRate: 'k-rate',
            },
        ];
    }

    process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean {
        const input = inputs[0];
        const output = outputs[0];
        const origGain = parameters.pitchValue[0];
        const gain = 1.0 + (origGain - 1.0) * 2.0;
        // console.debug(`Got ${origGain} -> ${gain}, ${parameters.pitchValue.length}, ${input.length}, ${output.length}`);

        if (!input || !output) {
            return true;
        }

        const minChannels = Math.min(input.length, output.length);
        const blockSize = output[0].length;
        for (let ch = 0; ch < minChannels; ch++) {
            const inputChannel = input[ch];
            const outputChannel = output[ch];

            for (let i = 0; i < blockSize; i++) {
                outputChannel[i] = inputChannel[i] * gain;
            }
        }

        // Fill any extra output channels with silence
        for (let ch = minChannels; ch < output.length; ch++) {
            output[ch].fill(0);
        }

        // Add any extra input channels to all output channels
        for (let ch = minChannels; ch < input.length; ch++) {
            const inputChannel = input[ch];
            for (let outCh = 0; outCh < minChannels; outCh++) {
                const outputChannel = output[outCh];
                for (let i = 0; i < blockSize; i++) {
                    outputChannel[i] += (inputChannel[i] / output.length) * gain;
                }
            }
        }

        return true;
    }
}

registerProcessor(PROCESSOR_NAME, PitchProcessor);
