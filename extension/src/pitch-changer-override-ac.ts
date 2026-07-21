import type { ExtensionSettings, OverrideScriptExports, OverrideStatsResult, PitchChangerOverrideInit } from './common.js';

// All functions and variables in this script are inserted into MAIN world of client scripts and must have unique prefix
// to not clash with user functions and variables.

let pitchChangerOverrideSettings: ExtensionSettings;

function pitchChangerOverrideDebugLog(...args: unknown[]): void {
    if (pitchChangerOverrideSettings?.debugLogging) {
        console.debug(...args);
    }
}

let pitchChangerOverrideNumAudioContextDestinations = 0;
let pitchChangerOverrideProcessorUrl: string;
let pitchChangerOverrideWasmUrl: string;

// We override global AudioContext constructor to insert our worklet node before the destination
//
// We insert a GainNode as a fake destination node before real destination node: this simplifies
// connecting/disconnecting our worklet In theory we could have made our worklet passthrough, but this simplifies code.
// Adding a single GainNode should not be too resource consuming.
class PitchChangerOverrideAudioContext extends AudioContext {
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
            this.gainNode = this.gainWithMaxChannelCount(gainNode);
            pitchChangerOverrideNumAudioContextDestinations++;
        }

        return this.gainNode as unknown as AudioDestinationNode;
    }

    // GainNode does not satisfy AudioDestinationNode, because it does not have one property. We simply add this
    // property to it.
    gainWithMaxChannelCount(gainNode: GainNode): AudioDestinationNode {
        //@ts-expect-error We are monkey-patching the live object, all the other solutions did not get accept by other AudioNode.connect() for some reason
        gainNode.maxChannelCount = this.realDestination.maxChannelCount;
        return gainNode as unknown as AudioDestinationNode;
    }

    async close(): Promise<void> {
        await super.close();
        console.log('OverridingAudioContext close');
        if (this.realDestination) {
            pitchChangerOverrideNumAudioContextDestinations--;
        }
    }
}

function pitchChangerOverrideApplySettings(newSettings: ExtensionSettings) {
    pitchChangerOverrideDebugLog(`Applying new settings in MAIN: ${JSON.stringify(newSettings)}, current settings ${JSON.stringify(pitchChangerOverrideSettings)}`);
}

function pitchChangerOverrideGetStats(): OverrideStatsResult {
    return {
        numAudioContextDestinations: pitchChangerOverrideNumAudioContextDestinations,
    };
}

function pitchChangerOverrideSetupExports() {
    (globalThis as unknown as OverrideScriptExports).exportPitchChangerOverrideApplySettings = pitchChangerOverrideApplySettings;
    (globalThis as unknown as OverrideScriptExports).exportPitchChangerOverrideGetStats = pitchChangerOverrideGetStats;
}

async function pitchChangerOverrideAsyncInit(init: PitchChangerOverrideInit) {
    pitchChangerOverrideSettings = init.settings;
    pitchChangerOverrideDebugLog('Audio Pitch Changer: Initializing MAIN content script, settings', init.settings);

    pitchChangerOverrideProcessorUrl = init.processorUrl;
    pitchChangerOverrideWasmUrl = init.wasmUrl;
}

// This must be run at document_start for two reasons:
//  1. We need to override AudioContext constructor before any other scripts are run.
//  2. We need to setup message handler before ISOLATED content script sends messages.
function pitchChangerOverrideInit() {
    pitchChangerOverrideSetupExports();

    globalThis.AudioContext = PitchChangerOverrideAudioContext;

    window.addEventListener('message', (event: MessageEvent) => {
        if (event.data?.type === 'pitch-changer-extension-main-init') {
            pitchChangerOverrideAsyncInit(event.data as PitchChangerOverrideInit);
        }
    })
}

pitchChangerOverrideInit();
