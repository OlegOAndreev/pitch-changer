import {
    PROCESSOR_NAME,
    type ExtensionSettings,
    type OverrideScriptExports,
    type OverrideStatsResult,
    type PitchChangerOverrideInit,
} from './common.js';

// Do IIFE to prevent symbols from leaking into main user script.
(function () {
    let settings: ExtensionSettings;
    let processorUrl: string;
    let wasmUrl: string;
    // We push the AudioContext only when the destination() is called first time.
    const overridenAudioContexts: PitchChangerOverrideAudioContext[] = [];

    function debugLog(...args: unknown[]): void {
        if (settings?.debugLogging) {
            console.debug(...args);
        }
    }

    // We override global AudioContext constructor to insert our worklet node before the destination
    //
    // We insert a GainNode as a fake destination node before real destination node: this simplifies
    // connecting/disconnecting our worklet In theory we could have made our worklet passthrough, but this simplifies
    // code. Adding a single GainNode should not be too resource consuming.
    class PitchChangerOverrideAudioContext extends AudioContext {
        // All fields are prefixed because they are visible to user scripts.
        private pitchChangerOverrideRealDestination: AudioDestinationNode | null = null;
        private pitchChangerOverrideGainNode: AudioDestinationNode | null = null;
        // We start creating pitchChangerOverrideWorkletNode only after destination() has been called at least once and
        // the pitch changer has been enabled.
        private pitchChangerOverrideWorkletNode: Promise<AudioWorkletNode> | null = null;
        private pitchChangerOverrideWasEnabled = false;
        private pitchChangerOverrideClosed = false;

        constructor(contextOptions?: AudioContextOptions | undefined) {
            super(contextOptions);
            debugLog('OverridingAudioContext constructed with:', contextOptions);
        }

        get destination(): AudioDestinationNode {
            if (!this.pitchChangerOverrideRealDestination) {
                debugLog('OverridingAudioContext getting destination');

                this.pitchChangerOverrideRealDestination = super.destination;
                const gainNode = this.createGain();
                gainNode.connect(this.pitchChangerOverrideRealDestination);
                // GainNode does not satisfy AudioDestinationNode, because it does not have one property. We simply add
                // this property to it. All the other solutions did not get accept by other AudioNode.connect() for some
                // reason.
                //@ts-expect-error We are monkey-patching the live object,
                gainNode.maxChannelCount = this.pitchChangerOverrideRealDestination.maxChannelCount;
                this.pitchChangerOverrideGainNode = gainNode as unknown as AudioDestinationNode;

                overridenAudioContexts.push(this);

                this.pitchChangerOverrideApplySettings();
            }

            return this.pitchChangerOverrideGainNode as unknown as AudioDestinationNode;
        }

        async close(): Promise<void> {
            debugLog('OverridingAudioContext close');
            if (this.pitchChangerOverrideRealDestination) {
                const idx = overridenAudioContexts.indexOf(this);
                if (idx === -1) {
                    console.error('Could not find this context in pitchChangerOverridenAudioContexts');
                } else {
                    overridenAudioContexts.splice(idx, 1);
                }
                this.pitchChangerOverrideClosed = true;
            }

            await super.close();
        }

        async pitchChangerOverrideApplySettings(): Promise<void> {
            // AudioContext could've been created before the script has been fully initialized from ISOLATED content
            // script. This method will be later called in onMessage handler from init.
            if (!settings || !processorUrl || !wasmUrl) {
                return;
            }

            if (settings.enabled) {
                const pitchChangerOverrideWorkletNode = await this.getPitchChangerOverrideWorkletNode();
                // Re-check the condition after await.
                if (settings.enabled && !this.pitchChangerOverrideClosed) {
                    if (!this.pitchChangerOverrideWasEnabled) {
                        // We cannot reach this point if the destination() has not been called: this means that
                        // PitchChangerOverrideAudioContext has not been published in overridenAudioContexts.
                        this.pitchChangerOverrideGainNode!.disconnect();
                        this.pitchChangerOverrideGainNode!.connect(pitchChangerOverrideWorkletNode);
                        pitchChangerOverrideWorkletNode.connect(this.pitchChangerOverrideRealDestination!);
                        this.pitchChangerOverrideWasEnabled = true;
                    }
                    //@ts-expect-error AudioParamMap does not currently have full interface described in TypeScript
                    (pitchChangerOverrideWorkletNode.parameters.get('pitchValue') as AudioParam).setValueAtTime(
                        settings.pitchValue,
                        0.0,
                    );

                }
            } else {
                // Do nothing if we haven't started initializing worklet node: this can happen only when we get
                // initialized with disabled extension.
                if (this.pitchChangerOverrideWorkletNode) {
                    const pitchChangerOverrideWorkletNode = await this.pitchChangerOverrideWorkletNode;
                    // Re-check the condition after await.
                    if (!settings.enabled && !this.pitchChangerOverrideClosed) {
                        if (this.pitchChangerOverrideWasEnabled) {
                            pitchChangerOverrideWorkletNode.disconnect();
                            // We cannot reach this point if the destination() has not been called: this means that
                            // PitchChangerOverrideAudioContext has not been published in overridenAudioContexts.
                            this.pitchChangerOverrideGainNode!.disconnect();
                            this.pitchChangerOverrideGainNode!.connect(this.pitchChangerOverrideRealDestination!);
                            this.pitchChangerOverrideWasEnabled = false;
                        }
                    }
                }
            }
        }

        async getPitchChangerOverrideWorkletNode(): Promise<AudioWorkletNode> {
            if (!this.pitchChangerOverrideWorkletNode) {
                this.pitchChangerOverrideWorkletNode = this.initPitchChangerOverrideWorkletNode();
            }
            return this.pitchChangerOverrideWorkletNode;
        }

        async initPitchChangerOverrideWorkletNode(): Promise<AudioWorkletNode> {
            await this.audioWorklet.addModule(processorUrl);
            debugLog(`Loaded processor from ${processorUrl} in MAIN`);

            // We cannot reach this point if the destination() has not been called: this means that
            // PitchChangerOverrideAudioContext has not been published in overridenAudioContexts.
            const destChannelCount = this.pitchChangerOverrideRealDestination!.channelCount;
            const result = new AudioWorkletNode(this, PROCESSOR_NAME, {
                // Force the WebAudio do up/downmixing for us.
                channelCount: destChannelCount,
                channelCountMode: 'explicit',
                outputChannelCount: [destChannelCount],
                parameterData: {
                    pitchValue: settings.pitchValue,
                },
            });
            return result;
        }
    }

    function applySettings(newSettings: ExtensionSettings) {
        debugLog(
            `Applying new settings in MAIN: ${JSON.stringify(newSettings)}, current settings ${JSON.stringify(settings)}`,
        );
        settings = newSettings;

        for (const context of overridenAudioContexts) {
            // We intentionally do not await this.
            context.pitchChangerOverrideApplySettings();
        }
    }

    function getStats(): OverrideStatsResult {
        return {
            numAudioContexts: overridenAudioContexts.length,
        };
    }

    function setupExports() {
        (globalThis as unknown as OverrideScriptExports).exportPitchChangerExtensionOverrideGetStats = getStats;
        (globalThis as unknown as OverrideScriptExports).exportPitchChangerExtensionOverrideApplySettings =
            applySettings;
    }

    // This must be run at document_start for two reasons:
    //  1. We need to override AudioContext constructor before any other scripts are run.
    //  2. We need to setup message handler before ISOLATED content script sends messages.
    function init() {
        globalThis.AudioContext = PitchChangerOverrideAudioContext;

        setupExports();

        const onMessage = (event: MessageEvent) => {
            if (event.source !== window) {
                return;
            }
            if (event.data?.type === 'pitch-changer-extension-override-init') {
                const init = event.data as PitchChangerOverrideInit;
                debugLog('Audio Pitch Changer: Initializing MAIN content script, settings', init.settings);
                settings = init.settings;
                processorUrl = init.processorUrl;
                wasmUrl = init.wasmUrl;

                for (const context of overridenAudioContexts) {
                    // We intentionally do not await this.
                    context.pitchChangerOverrideApplySettings();
                }

                window.removeEventListener('message', onMessage);
            }
        };
        window.addEventListener('message', onMessage);
    }

    init();
})();
