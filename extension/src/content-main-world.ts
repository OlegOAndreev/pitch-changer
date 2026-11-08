// This file MUST be compiled with --format=iife, otherwise we would leak all our names etc. into global namespace.

import {
    PROCESSOR_NAME,
    type ExtensionSettings,
    type MainWorldInit,
    type ProcessorInit,
    type ProcessorSetParams,
    type ProcessorStats,
    type WorkerIframeInit,
    isMessageWithType,
    type ApplySettings,
    type StatsResult,
} from './common.js';
import { anyToString } from '../../src/common-utils.js';

let settings: ExtensionSettings;
let processorUrl: string;
let workerIframeUrl: string;
let audioProcessorWorkerUrl: string;
// We add the AudioContext only when the destination() is called for the first time. The contexts are held weakly:
// pages often drop contexts without calling close(), and we must not keep them alive.
const overriddenAudioContexts = new Set<WeakRef<PitchChangerOverrideAudioContext>>();

// Iterates over contexts which are still alive, dropping the collected ones.
function liveAudioContexts(): PitchChangerOverrideAudioContext[] {
    const result = [];
    for (const ref of overriddenAudioContexts) {
        const context = ref.deref();
        if (context) {
            result.push(context);
        } else {
            overriddenAudioContexts.delete(ref);
        }
    }
    return result;
}

function debugLog(...args: unknown[]): void {
    if (settings?.debugLogging) {
        console.debug(...args);
    }
}

// This is a copy of createWorkerInIframe in content.ts, adapted to MAIN world.
// Removing the returned iframe terminates the worker.
async function createWorkerInIframe(): Promise<{ iframe: HTMLIFrameElement; port: MessagePort }> {
    const workerIframe = document.createElement('iframe');
    const workerIframeUrlParsed = new URL(workerIframeUrl);
    // The parameters are parsed by worker-iframe.ts
    workerIframeUrlParsed.searchParams.append('worker_url', audioProcessorWorkerUrl);

    type WorkerIframeResult = { iframe: HTMLIFrameElement; port: MessagePort };
    let resolve: (value: WorkerIframeResult) => void;
    let reject: (reason?: unknown) => void;
    const promise = new Promise<WorkerIframeResult>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    const onMessage = (event: MessageEvent) => {
        if (event.source !== workerIframe.contentWindow || !isMessageWithType(event.data)) {
            return;
        }
        if (event.data.type !== 'pitchChangerExtensionWorkerIframeInit') {
            reject(new Error(`MAIN: Unknown message type from worker iframe: ${anyToString(event.data)}`));
            return;
        }
        window.removeEventListener('message', onMessage);
        const data = event.data as WorkerIframeInit;
        resolve({ iframe: workerIframe, port: data.audioProcessorClientPort });
    };
    window.addEventListener('message', onMessage);

    try {
        workerIframe.src = workerIframeUrlParsed.href;
        workerIframe.style.display = 'none';
        document.body.appendChild(workerIframe);
    } catch (error) {
        window.removeEventListener('message', onMessage);
        throw new Error(`Failed to insert the worker iframe into the page: ${anyToString(error)}`, {
            cause: error,
        });
    }

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
    // Hosts the worker for this context, removed on close().
    private pitchChangerOverrideWorkerIframe: HTMLIFrameElement | null = null;
    private readonly pitchChangerOverrideRef = new WeakRef(this);
    // Statistics reported by the processor of this context, read by getStats().
    pitchChangerOverrideFastPathActive = true;
    pitchChangerOverrideCurrentLatencyMs = 0;
    pitchChangerOverrideNumUnderruns = 0;
    pitchChangerOverrideQueueLength = 0;

    constructor(contextOptions?: AudioContextOptions) {
        super(contextOptions);
        debugLog('PitchChangerOverrideAudioContext constructed with:', contextOptions);
    }

    get destination(): AudioDestinationNode {
        try {
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

                overriddenAudioContexts.add(this.pitchChangerOverrideRef);

                this.pitchChangerOverrideApplySettings().catch((error: unknown) => {
                    console.error('MAIN: failed to apply settings to a new AudioContext', error);
                });
            }

            return this.pitchChangerOverrideGainNode as unknown as AudioDestinationNode;
        } catch (error) {
            throw new Error(`Failed to set up the overridden AudioContext destination: ${anyToString(error)}`, {
                cause: error,
            });
        }
    }

    async close(): Promise<void> {
        debugLog('PitchChangerOverrideAudioContext close');
        overriddenAudioContexts.delete(this.pitchChangerOverrideRef);
        this.pitchChangerOverrideClosed = true;
        try {
            if (this.pitchChangerOverrideWorkerIframe) {
                this.pitchChangerOverrideWorkerIframe.remove();
                this.pitchChangerOverrideWorkerIframe = null;
            }
        } catch (error) {
            // Failing to remove the iframe must not prevent closing the context.
            console.error('MAIN: failed to remove the audio processor worker iframe', error);
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
                try {
                    if (!this.pitchChangerOverrideWasEnabled) {
                        // We cannot reach this point if the destination() has not been called: this means that
                        // PitchChangerOverrideAudioContext has not been published in overriddenAudioContexts.
                        this.pitchChangerOverrideGainNode!.disconnect();
                        this.pitchChangerOverrideGainNode!.connect(pitchChangerOverrideWorkletNode);
                        pitchChangerOverrideWorkletNode.connect(this.pitchChangerOverrideRealDestination!);
                        this.pitchChangerOverrideWasEnabled = true;
                    }
                    pitchChangerOverrideWorkletNode.port.postMessage({
                        type: 'pitchChangerExtensionProcessorSetParams',
                        processingMode: settings.processingMode,
                        pitchValue: settings.pitchValue,
                        targetLatency: settings.targetLatency,
                        quality: settings.quality,
                        enablePassthroughOptimization: settings.enablePassthroughOptimization,
                    } satisfies ProcessorSetParams);
                } catch (error) {
                    throw new Error(
                        `Failed to enable pitch processing for the overridden AudioContext: ${anyToString(error)}`,
                        { cause: error },
                    );
                }
            }
        } else {
            // Do nothing if we haven't started initializing worklet node: this can happen only when we get
            // initialized with disabled extension.
            if (this.pitchChangerOverrideWorkletNode) {
                const pitchChangerOverrideWorkletNode = await this.pitchChangerOverrideWorkletNode;
                // Re-check the condition after await.
                if (!settings.enabled && !this.pitchChangerOverrideClosed) {
                    try {
                        if (this.pitchChangerOverrideWasEnabled) {
                            pitchChangerOverrideWorkletNode.disconnect();
                            // We cannot reach this point if the destination() has not been called: this means that
                            // PitchChangerOverrideAudioContext has not been published in overriddenAudioContexts.
                            this.pitchChangerOverrideGainNode!.disconnect();
                            this.pitchChangerOverrideGainNode!.connect(this.pitchChangerOverrideRealDestination!);
                            this.pitchChangerOverrideWasEnabled = false;
                        }
                    } catch (error) {
                        throw new Error(
                            `Failed to disable pitch processing for the overridden AudioContext: ${anyToString(error)}`,
                            { cause: error },
                        );
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
        try {
            await this.audioWorklet.addModule(processorUrl);
        } catch (error) {
            throw new Error(`Failed to load the processor module from ${processorUrl}`, { cause: error });
        }

        // We cannot reach this point if the destination() has not been called: this means that
        // PitchChangerOverrideAudioContext has not been published in overriddenAudioContexts.
        const destChannelCount = this.pitchChangerOverrideRealDestination!.channelCount;
        let result: AudioWorkletNode;
        try {
            result = new AudioWorkletNode(this, PROCESSOR_NAME, {
                // Force the WebAudio do up/downmixing for us.
                channelCount: destChannelCount,
                channelCountMode: 'explicit',
                outputChannelCount: [destChannelCount],
            });
        } catch (error) {
            throw new Error(`Failed to create the ${PROCESSOR_NAME} AudioWorkletNode: ${anyToString(error)}`, {
                cause: error,
            });
        }
        result.onprocessorerror = (event: ErrorEvent) => {
            console.error(`Error from PitchChangerProcessor: ${event.message}, ${event.filename}:${event.lineno}`);
        };
        result.port.onmessage = (event: MessageEvent<ProcessorStats>) => {
            const message = event.data;
            if (message.type !== 'pitchChangerExtensionProcessorStats') {
                console.error(`MAIN: Unknown message type from processor: ${anyToString(message)}`);
                return;
            }
            this.pitchChangerOverrideFastPathActive = message.fastPathActive;
            this.pitchChangerOverrideCurrentLatencyMs = message.currentLatencyMs;
            this.pitchChangerOverrideNumUnderruns = message.numUnderruns;
            this.pitchChangerOverrideQueueLength = message.queueLength;
        };

        const { iframe, port: audioProcessorClientPort } = await createWorkerInIframe();
        if (this.pitchChangerOverrideClosed) {
            // close() was called while the worker iframe was loading.
            iframe.remove();
        } else {
            this.pitchChangerOverrideWorkerIframe = iframe;
        }
        try {
            result.port.postMessage(
                {
                    type: 'pitchChangerExtensionProcessorInit',
                    numChannels: destChannelCount,
                    audioProcessorClientPort: audioProcessorClientPort,
                } satisfies ProcessorInit,
                [audioProcessorClientPort],
            );
            result.port.postMessage({
                type: 'pitchChangerExtensionProcessorSetParams',
                processingMode: settings.processingMode,
                pitchValue: settings.pitchValue,
                targetLatency: settings.targetLatency,
                quality: settings.quality,
                enablePassthroughOptimization: settings.enablePassthroughOptimization,
            } satisfies ProcessorSetParams);
        } catch (error) {
            throw new Error(`failed to send initialization messages to the processor: ${anyToString(error)}`, {
                cause: error,
            });
        }
        debugLog(`Loaded processor from ${processorUrl} and worker from ${audioProcessorWorkerUrl} in MAIN`);

        return result;
    }
}

function applySettings(newSettings: ExtensionSettings) {
    debugLog(`Applying new settings in MAIN: ${anyToString(newSettings)}, current settings ${anyToString(settings)}`);
    settings = newSettings;

    for (const context of liveAudioContexts()) {
        // We intentionally do not await this.
        context.pitchChangerOverrideApplySettings().catch((error: unknown) => {
            console.error('MAIN: failed to apply settings to an overridden AudioContext', error);
        });
    }
}

function getStats(): StatsResult {
    let fastPathActive = true;
    let currentLatencyMs = 0;
    let numUnderruns = 0;
    let queueLength = 0;
    let numAudioContexts = 0;
    for (const context of liveAudioContexts()) {
        numAudioContexts++;
        fastPathActive = fastPathActive && context.pitchChangerOverrideFastPathActive;
        currentLatencyMs = Math.max(currentLatencyMs, context.pitchChangerOverrideCurrentLatencyMs);
        numUnderruns += context.pitchChangerOverrideNumUnderruns;
        queueLength = Math.max(queueLength, context.pitchChangerOverrideQueueLength);
    }
    return {
        type: 'pitchChangerExtensionStatsResult',
        numPlayingElements: 0,
        numAudioContexts: numAudioContexts,
        fastPathActive: fastPathActive,
        currentLatencyMs: currentLatencyMs,
        numUnderruns: numUnderruns,
        queueLength: queueLength,
    };
}

function completeInit(init: MainWorldInit) {
    debugLog('Audio Pitch Changer: Initializing MAIN content script, settings', init.settings);
    settings = init.settings;
    processorUrl = init.processorUrl;
    workerIframeUrl = init.workerIframeUrl;
    audioProcessorWorkerUrl = init.audioProcessorWorkerUrl;

    for (const context of liveAudioContexts()) {
        // We intentionally do not await this.
        context.pitchChangerOverrideApplySettings().catch((error: unknown) => {
            console.error('MAIN: failed to apply settings to an overridden AudioContext', error);
        });
    }
}

// This must be run at document_start for two reasons:
//  1. We need to override AudioContext constructor before any other scripts are run.
//  2. We need to setup message handler before ISOLATED content script sends messages.
function init() {
    globalThis.AudioContext = PitchChangerOverrideAudioContext;

    window.addEventListener('message', (event: MessageEvent) => {
        if (event.source !== window || !isMessageWithType(event.data)) {
            return;
        }
        switch (event.data.type) {
            case 'pitchChangerExtensionMainWorldInit':
                completeInit(event.data as MainWorldInit);
                break;
            case 'pitchChangerExtensionApplySettings':
                applySettings((event.data as ApplySettings).settings);
                break;
            case 'pitchChangerExtensionGetStats':
                window.postMessage(getStats(), '*');
                break;
        }
    });
}

init();
