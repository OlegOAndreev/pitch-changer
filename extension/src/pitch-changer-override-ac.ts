import { PROCESSOR_NAME, type ExtensionSettings, type OverrideScriptExports, type OverrideStatsResult, type PitchChangerOverrideInit } from './common.js';

// All functions and variables in this script are inserted into MAIN world of client scripts and must have unique prefix
// to not clash with user functions and variables.

let pitchChangerOverrideSettings: ExtensionSettings;
let pitchChangerOverrideProcessorUrl: string;
let pitchChangerOverrideWasmUrl: string;
const pitchChangerOverridenAudioContexts: PitchChangerOverrideAudioContext[] = [];

function pitchChangerOverrideDebugLog(...args: unknown[]): void {
    if (pitchChangerOverrideSettings?.debugLogging) {
        console.debug(...args);
    }
}

// We override global AudioContext constructor to insert our worklet node before the destination
//
// We insert a GainNode as a fake destination node before real destination node: this simplifies
// connecting/disconnecting our worklet In theory we could have made our worklet passthrough, but this simplifies code.
// Adding a single GainNode should not be too resource consuming.
class PitchChangerOverrideAudioContext extends AudioContext {
    private pitchChangerOverrideRealDestination: AudioDestinationNode | null = null;
    private pitchChangerOverrideGainNode: AudioDestinationNode | null = null;
    private pitchChangerOverrideWorkletNode: AudioWorkletNode | null = null;
    private pitchChangerOverrideModuleAdded = false;

    constructor(contextOptions?: AudioContextOptions | undefined) {
        super(contextOptions);
        pitchChangerOverrideDebugLog('OverridingAudioContext constructed with:', contextOptions);
    }

    get destination(): AudioDestinationNode {
        if (!this.pitchChangerOverrideRealDestination) {
            pitchChangerOverrideDebugLog('OverridingAudioContext getting destination');
            this.pitchChangerOverrideRealDestination = super.destination;
            const gainNode = this.createGain();
            gainNode.connect(this.pitchChangerOverrideRealDestination);
            // GainNode does not satisfy AudioDestinationNode, because it does not have one property. We simply add this
            // property to it.
            //@ts-expect-error We are monkey-patching the live object, all the other solutions did not get accept by other AudioNode.connect() for some reason
            gainNode.maxChannelCount = this.pitchChangerOverrideRealDestination.maxChannelCount;
            this.pitchChangerOverrideGainNode = gainNode as unknown as AudioDestinationNode;
            pitchChangerOverridenAudioContexts.push(this);

            // We intentionally do not await this.
            this.pitchChangerOverrideApplySettings();
        }

        return this.pitchChangerOverrideGainNode as unknown as AudioDestinationNode;
    }

    async close(): Promise<void> {
        await super.close();
        pitchChangerOverrideDebugLog('OverridingAudioContext close');
        if (this.pitchChangerOverrideRealDestination) {
            const idx = pitchChangerOverridenAudioContexts.indexOf(this);
            if (idx === -1) {
                console.error('Could not find this context in pitchChangerOverridenAudioContexts');
            } else {
                pitchChangerOverridenAudioContexts.splice(idx, 1);
            }
        }
    }

    async pitchChangerOverrideApplySettings() {
        if (!this.pitchChangerOverrideGainNode || !this.pitchChangerOverrideRealDestination) {
            console.error('Apply setting to the PitchChangerOverrideAudioContext without destination');
            return;
        }
        // AudioContext could've been created before the script has been fully initialized. This method will be later
        // called in asyncInit()
        if (!pitchChangerOverrideSettings || !pitchChangerOverrideProcessorUrl || !pitchChangerOverrideWasmUrl) {
            return;
        }
        if (pitchChangerOverrideSettings.enabled) {
            if (this.pitchChangerOverrideWorkletNode) {
                //@ts-expect-error AudioParamMap does not currently have full interface described in TypeScript
                (this.pitchChangerOverrideWorkletNode.parameters.get('pitchValue') as AudioParam).setValueAtTime(
                    pitchChangerOverrideSettings.pitchValue,
                    this.currentTime,
                );
            } else {
                if (!this.pitchChangerOverrideModuleAdded) {
                    await this.audioWorklet.addModule(pitchChangerOverrideProcessorUrl);
                    this.pitchChangerOverrideModuleAdded = true;
                    pitchChangerOverrideDebugLog(`Loaded processor from ${pitchChangerOverrideProcessorUrl} in MAIN`);
                }
                const destChannelCount = this.pitchChangerOverrideRealDestination.channelCount;
                this.pitchChangerOverrideWorkletNode = new AudioWorkletNode(this, PROCESSOR_NAME, {
                    // Force the WebAudio do up/downmixing for us.
                    channelCount: destChannelCount,
                    channelCountMode: 'explicit',
                    outputChannelCount: [destChannelCount],
                    parameterData: {
                        pitchValue: pitchChangerOverrideSettings.pitchValue,
                    },
                });
                this.pitchChangerOverrideGainNode.disconnect();
                this.pitchChangerOverrideGainNode.connect(this.pitchChangerOverrideWorkletNode);
                this.pitchChangerOverrideWorkletNode.connect(this.pitchChangerOverrideRealDestination);
            }
        } else {
            if (this.pitchChangerOverrideWorkletNode) {
                this.pitchChangerOverrideWorkletNode.disconnect();
                this.pitchChangerOverrideGainNode.disconnect();
                this.pitchChangerOverrideGainNode.connect(this.pitchChangerOverrideRealDestination);
                this.pitchChangerOverrideWorkletNode = null;
            }
        }
    }
}

function pitchChangerOverrideApplySettings(newSettings: ExtensionSettings) {
    pitchChangerOverrideDebugLog(`Applying new settings in MAIN: ${JSON.stringify(newSettings)}, current settings ${JSON.stringify(pitchChangerOverrideSettings)}`);
    pitchChangerOverrideSettings = newSettings;

    for (const context of pitchChangerOverridenAudioContexts) {
        // We intentionally do not await this.
        context.pitchChangerOverrideApplySettings();
    }
}

function pitchChangerOverrideGetStats(): OverrideStatsResult {
    return {
        numAudioContexts: pitchChangerOverridenAudioContexts.length,
    };
}

function pitchChangerOverrideSetupExports() {
    (globalThis as unknown as OverrideScriptExports).exportPitchChangerOverrideApplySettings = pitchChangerOverrideApplySettings;
    (globalThis as unknown as OverrideScriptExports).exportPitchChangerOverrideGetStats = pitchChangerOverrideGetStats;
}

async function pitchChangerOverrideAsyncInit(init: PitchChangerOverrideInit) {
    pitchChangerOverrideDebugLog('Audio Pitch Changer: Initializing MAIN content script, settings', init.settings);
    pitchChangerOverrideSettings = init.settings;
    pitchChangerOverrideProcessorUrl = init.processorUrl;
    pitchChangerOverrideWasmUrl = init.wasmUrl;

    for (const context of pitchChangerOverridenAudioContexts) {
        // We intentionally do not await this.
        context.pitchChangerOverrideApplySettings();
    }
}

// This must be run at document_start for two reasons:
//  1. We need to override AudioContext constructor before any other scripts are run.
//  2. We need to setup message handler before ISOLATED content script sends messages.
function pitchChangerOverrideInit() {
    globalThis.AudioContext = PitchChangerOverrideAudioContext;

    pitchChangerOverrideSetupExports();

    window.addEventListener('message', (event: MessageEvent) => {
        if (event.data?.type === 'pitch-changer-override-init') {
            pitchChangerOverrideAsyncInit(event.data as PitchChangerOverrideInit);
        }
    })
}

pitchChangerOverrideInit();
