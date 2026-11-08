import {
    disabledSettings,
    isMessageWithType,
    loadSettings,
    PROCESSOR_NAME,
    type ApplySettings,
    type ExtensionSettings,
    type GetStats,
    type MainWorldInit,
    type ProcessorInit,
    type ProcessorSetParams,
    type ProcessorStats,
    type StatsResult,
    type WorkerIframeInit,
} from './common.js';
import { anyToString } from '../../src/common-utils.js';

function debugLog(...args: unknown[]): void {
    if (settings.debugLogging) {
        console.debug(...args);
    }
}

// Notes on scripting.
//
// The extension does two things:
// 1. it re-routes all audio and video elements through AudioContext and applies AudioWorklet with pitch processing
//    if enabled (this can be done in the ISOLATED environment from this script)
// 2. it overrides AudioWorklet constructor (this must be done in the MAIN environment in content-main-world: we need to
//    replace the global variable)
//
// We want to process all frames (it's extremely frequent that audio/video elements are inside iframes) in the tab.
//
// The AudioWorklet overriding script must be run as soon as possible, before the page script: the easiest way to do it
// is by adding to manifest.json. The alternative of calling executeScript() is harder to pull off: getting current tab
// is an async operation, getting current frame is non-trivial.
//
// We propagate changed settings from popup to tabs and also get stats via messages. In case of MAIN world we proxy the
// messages from ISOLATED to MAIN using window.postMessage().
//
// Another way to propagate changes would be listening to storage.onChanged, but we debounce writes and want to have
// immediate feedback when changing the parameters.
//
// We run processing of all audio and video elements through a single AudioWorkletNode, that the elements connect to. It
// is lazily initialized.
//
// Each overridden AudioContext has its own AudioWorkletNode.
//
// MAIN world content script is run first before all the other scripts are loaded to override the AudioContext
// constructor, the ISOLATED content script is run later, because it needs to complete initialization of MAIN world
// content script (see order of scripts in manifest.json and
// https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/manifest.json/content_scripts#load_order).
// Unlike the ISOLATED world content script, MAIN world content script cannot access chrome.runtime APIs and cannot get
// the URLs of extension scripts/wasm files or current settings.

// Initializing global values is much easier than leaving them undefined.
let settings: ExtensionSettings = disabledSettings();

interface WorkletState {
    context: AudioContext;
    node: AudioWorkletNode;
}
let globalState: Promise<WorkletState> | null = null;

// Statistics reported by the worklet processor via messages.
let workletProcessorNumUnderruns = 0;
let workletProcessorFastPathActive = true;
let workletProcessorQueueLength = 0;
let workletProcessorCurrentLatencyMs = 0;

// All HTMLMediaElements in the document get registered in the mediaElements as soon as they get played for the first
// time. NOTE: we miss the shadow DOM or detached elements, those may be caught only by overriding constructors in MAIN
// world script.
//
// While we register the elements on play, we do not immediately reroute them to our worklet, if the extension is
// disabled - this is what 'registered' type means. If the extension is enabled, we reroute its output via
// createMediaElementSource(). We cannot destroy the created MediaElementAudioSourceNode (see
// https://github.com/WebAudio/web-audio-api/issues/1202), once the element becomes 'rerouted' it stays that way until
// it gets destroyed.
//
// We skip creating MediaElementAudioSourceNode for the elements that we assume will be muted when they are played --
// see isMediaElementSafeForWebAudio. Unfortunately, it still may break because the audio may change to the unsafe src
// after playing from safe src.
type ElementState =
    | { type: 'registered'; element: WeakRef<HTMLMediaElement> }
    | { type: 'rerouted'; element: WeakRef<HTMLMediaElement>; sourceNode: WeakRef<AudioNode> };
const mediaElements = new Set<ElementState>();
// We keep the list of playing elements to suspend the audio context when no elements are playing. We track this set
// even if the extension is disabled. Unlike mediaElements we use regular references here, because we remove elements
// from this set when stop playing.
const playingElements = new Set<HTMLMediaElement>();

async function initWorkletState(): Promise<WorkletState> {
    const context = new AudioContext();
    const processorUrl = chrome.runtime.getURL('processor.js');
    try {
        await context.audioWorklet.addModule(processorUrl);
    } catch (error) {
        throw new Error(`Failed to load the processor module from ${processorUrl}: ${anyToString(error)}`, {
            cause: error,
        });
    }
    debugLog(`Created shared AudioContext and loaded processor from ${processorUrl} in ISOLATED`);

    const destChannelCount = context.destination.channelCount;
    let node: AudioWorkletNode;
    try {
        // We originally passed channelCount via processorOptions, but this breaks in Firefox even for simplest objects.
        // The node.port on the other hand works perfectly.
        node = new AudioWorkletNode(context, PROCESSOR_NAME, {
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

    try {
        const audioProcessorClientPort = await createWorkerInIframe();
        node.port.postMessage(
            {
                type: 'pitchChangerExtensionProcessorInit',
                numChannels: destChannelCount,
                audioProcessorClientPort: audioProcessorClientPort,
            } satisfies ProcessorInit,
            [audioProcessorClientPort],
        );
        node.port.postMessage({
            type: 'pitchChangerExtensionProcessorSetParams',
            processingMode: settings.processingMode,
            pitchValue: settings.pitchValue,
            targetLatency: settings.targetLatency,
            quality: settings.quality,
            enablePassthroughOptimization: settings.enablePassthroughOptimization,
        } satisfies ProcessorSetParams);
    } catch (error) {
        throw new Error(`Failed to set the processor parameters: ${anyToString(error)}`, { cause: error });
    }

    node.port.onmessage = (event: MessageEvent<ProcessorStats>) => {
        const message = event.data;
        if (message.type !== 'pitchChangerExtensionProcessorStats') {
            console.error(`ISOLATED: Unknown message type from processor: ${anyToString(message)}`);
        }
        workletProcessorFastPathActive = message.fastPathActive;
        workletProcessorCurrentLatencyMs = message.currentLatencyMs;
        workletProcessorNumUnderruns = message.numUnderruns;
        workletProcessorQueueLength = message.queueLength;
    };

    node.connect(context.destination);
    if (context.state !== 'running') {
        context.resume().catch((error: unknown) => {
            console.error('ISOLATED: failed to resume the shared AudioContext', error);
        });
    }

    debugLog(`Created shared worklet node, channelCount ${destChannelCount}`);
    return {
        context,
        node,
    };
}

// Unfortunately due to https://issues.chromium.org/issues/41098022 we cannot directly create a worker using the script
// from the extension. One workaround is starting the worker from blob URL, but it does not work consistently: the page
// CSP may prevent loading the blob URL. Our workaround is rather horrible: we create an iframe inside the page, create
// a worker inside it and communicate with it.
//
// Inspired by https://github.com/Rob--W/chrome-api/tree/master/worker_proxy
async function createWorkerInIframe(): Promise<MessagePort> {
    const workerIframe = document.createElement('iframe');
    const workerIframeUrl = new URL(chrome.runtime.getURL('worker-iframe.html'));
    // The parameters are parsed by worker-iframe.ts
    workerIframeUrl.searchParams.append('worker_url', chrome.runtime.getURL('audio-processor-worker.js'));

    let resolve: (value: MessagePort) => void;
    let reject: (reason?: unknown) => void;
    const promise = new Promise<MessagePort>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    const onMessage = (event: MessageEvent) => {
        if (event.source !== workerIframe.contentWindow || !isMessageWithType(event.data)) {
            return;
        }
        if (event.data.type !== 'pitchChangerExtensionWorkerIframeInit') {
            reject(new Error(`ISOLATED: Unknown message type from worker iframe: ${anyToString(event.data)}`));
            return;
        }
        window.removeEventListener('message', onMessage);
        const data = event.data as WorkerIframeInit;
        resolve(data.audioProcessorClientPort);
    };
    window.addEventListener('message', onMessage);

    workerIframe.src = workerIframeUrl.href;
    workerIframe.style.display = 'none';
    document.body.appendChild(workerIframe);

    return promise;
}

async function getWorkletState(): Promise<WorkletState> {
    if (!globalState) {
        globalState = initWorkletState();
    }
    return globalState;
}

function isWorkletStateInitialized(): boolean {
    return globalState !== null;
}

// Rerouting an element through WebAudio (via createMediaElementSource()) silences it if the resource is cross-origin
// and was not fetched with CORS. The routing cannot be undone, so we need to be sure that the element is safe to be
// rerouted: the best way is to do this check inside 'play' handler, so that currentSrc is available.
function isMediaElementSafeForWebAudio(element: HTMLMediaElement): boolean {
    if (element.currentSrc === '') {
        return true;
    }
    const url = new URL(element.currentSrc);
    if (url.protocol === 'blob:' || url.protocol === 'data:') {
        return true;
    }
    if (url.origin === location.origin) {
        return true;
    }
    // For cross-origin URLs the media is fetched in CORS mode only if the crossorigin attribute is present. If it
    // is present and the resource is playing back, the CORS check has passed (a CORS failure results in a load
    // error, not in tainted playback). Corner case: an attribute which was added after the resource was already
    // loaded in no-cors mode does not un-taint the already loaded data.
    return element.crossOrigin !== null;
}

function findElementState(e: HTMLMediaElement): ElementState | null {
    for (const state of mediaElements) {
        const element = state.element.deref();
        if (element === e) {
            // We stop iterating here and do not clean up all the stale elements.
            return state;
        }
        // Do cooperative cleanup of mediaElements.
        if (!element) {
            mediaElements.delete(state);
        }
    }
    return null;
}

function isElementRegistered(e: HTMLMediaElement): boolean {
    // This loop is responsible for two things: checking if the element is present and removing the stale element, do
    // not short circuit it.
    let result = false;
    for (const state of mediaElements) {
        const element = state.element.deref();
        if (element === e) {
            // We already registered the element.
            result = true;
            continue;
        }
        // Do cooperative cleanup of mediaElements.
        if (!element) {
            mediaElements.delete(state);
        }
    }
    return result;
}

// Perform registration, all element re-routing is now done in play/pause handlers of the element.
function registerElement(element: HTMLMediaElement) {
    const state: ElementState = {
        type: 'registered',
        element: new WeakRef(element),
    };
    mediaElements.add(state);

    // If registerElement() is called from onWindowPlay(), onElementPlay() will be called later, see
    // https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener
    //
    // > However, the new listener may be triggered during a later stage of event flow, such as during the bubbling
    // phase.
    element.addEventListener('play', () => onElementPlay(element));
    element.addEventListener('pause', () => onElementPause(element));
}

async function onElementPlay(element: HTMLMediaElement) {
    try {
        await onElementPlayImpl(element);
    } catch (error) {
        console.error('ISOLATED: onElementPlay() failed', error);
    }
}

async function onElementPlayImpl(element: HTMLMediaElement) {
    if (element.paused) {
        console.log(`Got paused element in onElementPlay: ${element.nodeName}, ${element.id}`);
        return;
    }

    playingElements.add(element);
    // Re-enable the context if it has been created.
    if (playingElements.size === 1 && isWorkletStateInitialized()) {
        const worklet = await getWorkletState();
        // The context could be suspended either by us, or by the browser itself.
        if (worklet.context.state !== 'running') {
            debugLog(`Context state ${worklet.context.state}, resuming`);
            worklet.context.resume().catch((error: unknown) => {
                console.error('ISOLATED: failed to resume the shared AudioContext', error);
            });
        }
    }

    if (!settings.enabled) {
        return;
    }
    const worklet = await getWorkletState();
    // Re-check after await.
    if (!settings.enabled) {
        return;
    }

    const state = findElementState(element);
    if (!state) {
        console.error(`ISOLATED: playing element is missing: ${element.nodeName} ${element.id}, did it get deleted?`);
        return;
    }

    if (state.type === 'registered') {
        rerouteElement(state, worklet);
    }
}

function rerouteElement(state: ElementState, worklet: WorkletState) {
    try {
        rerouteElementImpl(state, worklet);
    } catch (error) {
        console.error('ISOLATED: rerouteElement() failed', error);
    }
}

function rerouteElementImpl(state: ElementState, worklet: WorkletState) {
    const element = state.element.deref();
    if (!element) {
        console.error('ISOLATED: cannot reroute a media element which has been deleted');
        return;
    }

    if (!isMediaElementSafeForWebAudio(element)) {
        // Leave the element playing unprocessed instead of silencing it.
        debugLog(
            `Not rerouting worklet to CORS-unsafe element ${element.nodeName} ${element.id} (${element.currentSrc})`,
        );
        return;
    }

    let sourceNode: MediaElementAudioSourceNode;
    try {
        sourceNode = worklet.context.createMediaElementSource(element);
    } catch (error) {
        throw new Error(
            `Failed to create a MediaElementAudioSourceNode for ${element.nodeName} ${element.id}: ${anyToString(error)}`,
            { cause: error },
        );
    }
    try {
        sourceNode.connect(worklet.node);
    } catch (error) {
        throw new Error(
            `Failed to connect ${element.nodeName} ${element.id} to the worklet node: ${anyToString(error)}`,
            { cause: error },
        );
    }
    mediaElements.delete(state);
    mediaElements.add({
        type: 'rerouted',
        element: new WeakRef(element),
        sourceNode: new WeakRef(sourceNode),
    });
    debugLog(`Rerouted element ${element.nodeName} ${element.id}, channelCount ${sourceNode.channelCount}`);
}

async function onElementPause(element: HTMLMediaElement) {
    try {
        await onElementPauseImpl(element);
    } catch (error) {
        console.error('ISOLATED: onElementPause() failed', error);
    }
}

// Disconnecting the worklet is not enough for process() callback to stop being called, instead we suspend the whole
// context. See https://github.com/WebAudio/web-audio-api/issues/2658
async function onElementPauseImpl(element: HTMLMediaElement) {
    if (!element.paused) {
        console.log(`Got not paused element in onElementPause: ${element.nodeName}, ${element.id}`);
        return;
    }

    playingElements.delete(element);
    if (playingElements.size === 0 && isWorkletStateInitialized()) {
        const worklet = await getWorkletState();
        if (worklet.context.state === 'running') {
            debugLog(`No playing elements, suspending the context`);
            void worklet.context.suspend();
        }
    }
}

// This function must be used only for initial registration of media elements. After that we react to 'play' and 'pause'
// on the elements themselves.
function onWindowPlay(ev: Event) {
    try {
        onWindowPlayImpl(ev);
    } catch (error) {
        console.error('ISOLATED: onWindowPlay() failed', error);
    }
}

function onWindowPlayImpl(ev: Event) {
    if (!ev.target || !(ev.target instanceof HTMLMediaElement)) {
        console.error('ISOLATED: expected "play" to fire on HTMLMediaElement, got', ev.target);
        return;
    }

    if (isElementRegistered(ev.target)) {
        return;
    }

    registerElement(ev.target);
}

async function applySettings(newSettings: ExtensionSettings) {
    debugLog(
        `Applying new settings in ISOLATED: ${anyToString(newSettings)}, current settings ${anyToString(settings)}`,
    );
    const gotEnabled = newSettings.enabled && !settings.enabled;
    const gotDisabled = !newSettings.enabled && settings.enabled;
    settings = newSettings;

    if (gotEnabled) {
        debugLog('Re-enabling processing for page');
        // Re-route all elements via sourceNode
        if (mediaElements.size > 0) {
            const worklet = await getWorkletState();
            for (const state of mediaElements) {
                const element = state.element.deref();
                if (!element) {
                    continue;
                }
                try {
                    switch (state.type) {
                        case 'registered':
                            if (!element.paused) {
                                rerouteElement(state, worklet);
                            }
                            break;
                        case 'rerouted': {
                            const sourceNode = state.sourceNode.deref();
                            if (!sourceNode) {
                                continue;
                            }
                            sourceNode.disconnect();
                            sourceNode.connect(worklet.node);
                            break;
                        }
                    }
                } catch (error) {
                    console.error(`Failed to enable processing for ${element.nodeName} ${element.id}`, error);
                }
            }
        }
    }

    if (gotDisabled) {
        debugLog('Disabling processing for page');
        if (mediaElements.size > 0) {
            const worklet = await getWorkletState();
            // Re-route all elements via context destination.
            for (const state of mediaElements) {
                if (state.type !== 'rerouted') {
                    continue;
                }
                const sourceNode = state.sourceNode.deref();
                if (!sourceNode) {
                    continue;
                }
                const element = state.element.deref();
                try {
                    sourceNode.disconnect();
                    sourceNode.connect(worklet.context.destination);
                } catch (error) {
                    console.error(`Failed to disable processing for ${element?.nodeName} ${element?.id}`, error);
                }
            }
        }
    }

    // Update the parameters of the worklet if it exists.
    if (globalState) {
        const worklet = await globalState;
        worklet.node.port.postMessage({
            type: 'pitchChangerExtensionProcessorSetParams',
            processingMode: settings.processingMode,
            pitchValue: settings.pitchValue,
            targetLatency: settings.targetLatency,
            quality: settings.quality,
            enablePassthroughOptimization: settings.enablePassthroughOptimization,
        } satisfies ProcessorSetParams);
    }
}

function getStats(): StatsResult {
    return {
        type: 'pitchChangerExtensionStatsResult',
        numAudioContexts: 0,
        numPlayingElements: playingElements.size,
        fastPathActive: workletProcessorFastPathActive,
        currentLatencyMs: workletProcessorCurrentLatencyMs,
        numUnderruns: workletProcessorNumUnderruns,
        queueLength: workletProcessorQueueLength,
    };
}

async function init(): Promise<void> {
    window.addEventListener('play', onWindowPlay, true);

    chrome.runtime.onMessage.addListener((message: ApplySettings | GetStats) => {
        // Pass the settings to the MAIN world content script.
        window.postMessage(message, '*');

        switch (message.type) {
            case 'pitchChangerExtensionApplySettings':
                applySettings(message.settings).catch((error: unknown) => {
                    console.error('ISOLATED: Failed to apply settings', error);
                });
                break;
            case 'pitchChangerExtensionGetStats':
                chrome.runtime.sendMessage(getStats()).catch((error: unknown) => {
                    console.error('ISOLATED: Send stats', error);
                });
                break;
        }
    });
    window.addEventListener('message', (event: MessageEvent) => {
        if (event.source !== window || !isMessageWithType(event.data)) {
            return;
        }
        if (event.data.type === 'pitchChangerExtensionStatsResult') {
            chrome.runtime.sendMessage(event.data).catch((error: unknown) => {
                console.error('ISOLATED: Forward stats', error);
            });
        }
    });

    await applySettings(await loadSettings());
    debugLog('Audio Pitch Changer: Initializing ISOLATED content script');

    // Complete initialization of the MAIN content script. Ideally we would simply use executeScript from ISOLATED
    // content script to run MAIN init, but browser does not allow getting current tab id :-(
    window.postMessage(
        {
            type: 'pitchChangerExtensionMainWorldInit',
            processorUrl: chrome.runtime.getURL('processor.js'),
            workerIframeUrl: chrome.runtime.getURL('worker-iframe.html'),
            audioProcessorWorkerUrl: chrome.runtime.getURL('audio-processor-worker.js'),
            settings: settings,
        } satisfies MainWorldInit,
        '*',
    );

    // Our content script is running relatively. We do this to e.g. catch document.write() which updates the blank
    // iframe. This means that e.g. the document could've started autoplaying and we need to catch this element.
    for (const element of document.querySelectorAll('audio, video')) {
        if (!(element instanceof HTMLMediaElement)) {
            continue;
        }
        if (element.paused) {
            continue;
        }
        try {
            debugLog(`Caught autoplaying element ${element.nodeName} ${element.id}`);
            if (!isElementRegistered(element)) {
                registerElement(element);
                void onElementPlay(element);
            }
        } catch (error) {
            throw new Error(
                `Failed to register autoplaying element ${element.nodeName} ${element.id}: ${anyToString(error)}`,
                { cause: error },
            );
        }
    }
}

init().catch((error: unknown) => {
    console.error('Audio Pitch Changer: ISOLATED content script initialization failed', error);
});
