import {
    PROCESSOR_NAME,
    type ExtensionSettings,
    type OverrideScriptExports,
    type OverrideStatsResult,
    type PitchChangerOverrideInit,
    type ProcessorInit,
    type ProcessorOptions,
    type ProcessorSetParams,
    type ProcessorStats,
    type WorkerIframeInit,
} from './common.js';

// Do IIFE to prevent symbols from leaking into main user script.
(function () {
    let settings: ExtensionSettings;
    let processorUrl: string;
    let workerIframeUrl: string;
    let audioProcessorWorkerUrl: string;
    // We push the AudioContext only when the destination() is called for the first time.
    const overriddenAudioContexts: PitchChangerOverrideAudioContext[] = [];

    function debugLog(...args: unknown[]): void {
        if (settings?.debugLogging) {
            console.debug(...args);
        }
    }

    // This is a copy of createWorkerInIframe from pitch-changer-content.ts, adapted to MAIN world.
    async function createWorkerInIframe(): Promise<MessagePort> {
        const workerIframe = document.createElement('iframe');
        const workerIframeUrlParsed = new URL(workerIframeUrl);
        // The parameters are parsed by worker-iframe.ts
        workerIframeUrlParsed.searchParams.append('worker_url', audioProcessorWorkerUrl);

        let resolve: (value: MessagePort) => void;
        const promise = new Promise<MessagePort>((res) => {
            resolve = res;
        });
        const onMessage = (event: MessageEvent) => {
            if (event.source !== workerIframe.contentWindow) {
                return;
            }
            if (event.data?.type !== 'pitch-changer-extension-worker-iframe-init') {
                throw new Error(`MAIN: Unknown message type from worker iframe: ${JSON.stringify(event.data)}`);
            }
            const data = event.data as WorkerIframeInit;
            resolve(data.audioProcessorClientPort);
            window.removeEventListener('message', onMessage);
        };
        window.addEventListener('message', onMessage);

        workerIframe.src = workerIframeUrlParsed.href;
        workerIframe.hidden = true;
        document.body.appendChild(workerIframe);

        return promise;
    }

    // We override global AudioContext constructor to insert our worklet node before the destination
    //
    // We insert a GainNode as a fake destination node before real destination node: this simplifies
    // connecting/disconnecting our worklet. Adding a single GainNode should not be too resource consuming.
    class PitchChangerOverrideAudioContext extends AudioContext {
        // All fields are prefixed because they are visible to user scripts.
        private pitchChangerOverrideRealDestination: AudioDestinationNode | null = null;
        private pitchChangerOverrideGainNode: AudioDestinationNode | null = null;
        // We start creating pitchChangerOverrideWorkletNode only after destination() has been called at least once and
        // the pitch changer has been enabled.
        private pitchChangerOverrideWorkletNode: Promise<AudioWorkletNode> | null = null;
        private pitchChangerOverrideWasEnabled = false;
        private pitchChangerOverrideClosed = false;
        // Statistics reported by the processor of this context, read by getStats().
        pitchChangerOverrideFastPathActive = true;
        pitchChangerOverrideCurrentLatencyMs = 0;
        pitchChangerOverrideNumUnderruns = 0;
        pitchChangerOverrideQueueLength = 0;

        constructor(contextOptions?: AudioContextOptions | undefined) {
            super(contextOptions);
            debugLog('PitchChangerOverrideAudioContext constructed with:', contextOptions);
        }

        get destination(): AudioDestinationNode {
            if (!this.pitchChangerOverrideRealDestination) {
                debugLog('PitchChangerOverrideAudioContext getting destination');

                this.pitchChangerOverrideRealDestination = super.destination;
                const gainNode = this.createGain();
                gainNode.connect(this.pitchChangerOverrideRealDestination);
                // GainNode does not satisfy AudioDestinationNode, because it does not have one property. We simply add
                // this property to it. All the other solutions did not get accepted by other AudioNode.connect() for
                // some reason.
                //@ts-expect-error We are monkey-patching the live object,
                gainNode.maxChannelCount = this.pitchChangerOverrideRealDestination.maxChannelCount;
                this.pitchChangerOverrideGainNode = gainNode as unknown as AudioDestinationNode;

                overriddenAudioContexts.push(this);

                this.pitchChangerOverrideApplySettings();
            }

            return this.pitchChangerOverrideGainNode as unknown as AudioDestinationNode;
        }

        async close(): Promise<void> {
            debugLog('PitchChangerOverrideAudioContext close');
            if (this.pitchChangerOverrideRealDestination) {
                const idx = overriddenAudioContexts.indexOf(this);
                if (idx === -1) {
                    console.error('Could not find this context in overriddenAudioContexts');
                } else {
                    overriddenAudioContexts.splice(idx, 1);
                }
                this.pitchChangerOverrideClosed = true;
            }

            await super.close();
        }

        async pitchChangerOverrideApplySettings(): Promise<void> {
            // AudioContext could've been created before the script has been fully initialized from ISOLATED content
            // script. This method will be later called in onMessage handler from init.
            if (!settings || !processorUrl) {
                return;
            }

            if (settings.enabled) {
                const pitchChangerOverrideWorkletNode = await this.getPitchChangerOverrideWorkletNode();
                // Re-check the condition after await.
                if (settings.enabled && !this.pitchChangerOverrideClosed) {
                    if (!this.pitchChangerOverrideWasEnabled) {
                        // We cannot reach this point if the destination() has not been called: this means that
                        // PitchChangerOverrideAudioContext has not been published in overriddenAudioContexts.
                        this.pitchChangerOverrideGainNode!.disconnect();
                        this.pitchChangerOverrideGainNode!.connect(pitchChangerOverrideWorkletNode);
                        pitchChangerOverrideWorkletNode.connect(this.pitchChangerOverrideRealDestination!);
                        this.pitchChangerOverrideWasEnabled = true;
                    }
                    pitchChangerOverrideWorkletNode.port.postMessage({
                        type: 'pitch-changer-extension-processor-set-params',
                        processingMode: settings.processingMode,
                        pitchValue: settings.pitchValue,
                        targetLatency: settings.targetLatency,
                        enablePassthroughOptimization: settings.enablePassthroughOptimization,
                    } as ProcessorSetParams);
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
                            // PitchChangerOverrideAudioContext has not been published in overriddenAudioContexts.
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

            // We cannot reach this point if the destination() has not been called: this means that
            // PitchChangerOverrideAudioContext has not been published in overriddenAudioContexts.
            const destChannelCount = this.pitchChangerOverrideRealDestination!.channelCount;
            const result = new AudioWorkletNode(this, PROCESSOR_NAME, {
                // Force the WebAudio do up/downmixing for us.
                channelCount: destChannelCount,
                channelCountMode: 'explicit',
                outputChannelCount: [destChannelCount],
                processorOptions: {
                    numChannels: destChannelCount,
                } as ProcessorOptions,
            });
            result.onprocessorerror = (event: ErrorEvent) => {
                console.error(
                    `Error from PitchChangerProcessor: ${event.message}, ${event.filename}:${event.lineno}, ${event.error}`,
                );
            };
            result.port.onmessage = (event: MessageEvent<ProcessorStats>) => {
                const message = event.data;
                if (message.type !== 'pitch-changer-extension-processor-stats') {
                    throw new Error(`MAIN: Unknown message type from processor: ${JSON.stringify(message)}`);
                }
                this.pitchChangerOverrideFastPathActive = message.fastPathActive;
                this.pitchChangerOverrideCurrentLatencyMs = message.currentLatencyMs;
                this.pitchChangerOverrideNumUnderruns = message.numUnderruns;
                this.pitchChangerOverrideQueueLength = message.queueLength;
            };

            const audioProcessorClientPort = await createWorkerInIframe();
            result.port.postMessage(
                {
                    type: 'pitch-changer-extension-processor-init',
                    audioProcessorClientPort: audioProcessorClientPort,
                } as ProcessorInit,
                [audioProcessorClientPort],
            );
            result.port.postMessage({
                type: 'pitch-changer-extension-processor-set-params',
                processingMode: settings.processingMode,
                pitchValue: settings.pitchValue,
                targetLatency: settings.targetLatency,
                enablePassthroughOptimization: settings.enablePassthroughOptimization,
            } as ProcessorSetParams);
            debugLog(`Loaded processor from ${processorUrl} and worker from ${audioProcessorWorkerUrl} in MAIN`);

            return result;
        }
    }

    function applySettings(newSettings: ExtensionSettings) {
        debugLog(
            `Applying new settings in MAIN: ${JSON.stringify(newSettings)}, current settings ${JSON.stringify(settings)}`,
        );
        settings = newSettings;

        for (const context of overriddenAudioContexts) {
            // We intentionally do not await this.
            context.pitchChangerOverrideApplySettings();
        }
    }

    function getStats(): OverrideStatsResult {
        let fastPathActive = true;
        let currentLatencyMs = 0;
        let numUnderruns = 0;
        let queueLength = 0;
        for (const context of overriddenAudioContexts) {
            fastPathActive = fastPathActive && context.pitchChangerOverrideFastPathActive;
            currentLatencyMs = Math.max(currentLatencyMs, context.pitchChangerOverrideCurrentLatencyMs);
            numUnderruns += context.pitchChangerOverrideNumUnderruns;
            queueLength = Math.max(queueLength, context.pitchChangerOverrideQueueLength);
        }
        return {
            numAudioContexts: overriddenAudioContexts.length,
            fastPathActive: fastPathActive,
            currentLatencyMs: currentLatencyMs,
            numUnderruns: numUnderruns,
            queueLength: queueLength,
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
                workerIframeUrl = init.workerIframeUrl;
                audioProcessorWorkerUrl = init.audioProcessorWorkerUrl;

                for (const context of overriddenAudioContexts) {
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
