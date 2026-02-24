import type { OverrideScriptExports, OverrideStatsResult } from './common.js';

console.log('Initializing injected content script');

// We override global AudioContext constructor to insert our worklet before destination
//
// We insert a GainNode as a fake destination node before real destination node: this simplifies
// connecting/disconnecting our worklet In theory we could have made our worklet passthrough, but this simplifies code.
// Adding a single GainNode should not be too resource consuming.

// GainNode does not satisfy AudioDestinationNode, because it does not have one property. We simply add this property
// to it.
function gainWithMaxChannelCount(gainNode: GainNode, maxChannelCount: number): AudioDestinationNode {
    //@ts-expect-error We are monkey-patching the live object, all the other solutions did not get accept by other
    // AudioNode.connect() for some reason
    gainNode.maxChannelCount = maxChannelCount;
    return gainNode as unknown as AudioDestinationNode;
}

let numAudioContextDestinations = 0;

class OverridingAudioContext extends AudioContext {
    private realDestination: AudioDestinationNode | null = null;
    private gainNode: AudioDestinationNode | null = null;

    constructor(contextOptions?: AudioContextOptions | undefined) {
        super(contextOptions);
        console.log('OverridingAudioContext constructed with:', contextOptions);
    }

    get destination(): AudioDestinationNode {
        if (!this.realDestination) {
            console.log('OverridingAudioContext getting destination');
            this.realDestination = super.destination;
            const gainNode = this.createGain();
            gainNode.connect(this.realDestination);
            this.gainNode = gainWithMaxChannelCount(gainNode, this.realDestination.maxChannelCount);
            numAudioContextDestinations++;
        }

        return this.gainNode as unknown as AudioDestinationNode;
    }

    async close(): Promise<void> {
        await super.close();
        console.log('OverridingAudioContext close');
        if (this.realDestination) {
            numAudioContextDestinations--;
        }
    }
}

function getStats(): OverrideStatsResult {
    return {
        numAudioContextDestinations: numAudioContextDestinations,
    };
}

function setupExports() {
    (globalThis as unknown as OverrideScriptExports).exportPitchShifterOverrideGetStats = getStats;
}

function init() {
    setupExports();

    globalThis.AudioContext = OverridingAudioContext;
}

init();
