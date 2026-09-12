import {
    loadSettings,
    PROCESSOR_NAME,
    type ContentScriptExports,
    type ExtensionSettings,
    type PitchChangerOverrideInit,
    type ProcessorInit,
    type ProcessorOptions,
    type ProcessorSetParams,
    type ProcessorStats,
    type StatsResult,
    type WorkerIframeInit,
} from './common.js';

function debugLog(...args: unknown[]): void {
    if (settings?.debugLogging) {
        console.debug(...args);
    }
}

// Notes on scripting.
//
// The extension does two things:
// 1. it re-routes all audio and video elements through AudioContext and applies AudioWorklet with pitch processing
//    if enabled (this can be done in the ISOLATED environment from this script)
// 2. it overrides AudioWorklet constructor (this must be done in the MAIN environment from pitch-changer-override-ac:
//    we need to replace the global variable)
//
// We want to process all frames (it's extremely frequent that audio/video elements are inside iframes) in the tab.
//
// The AudioWorklet overriding script must be run as soon as possible, before the page script: the easiest way to do it
// is by adding to manifest.json. The alternative of calling executeScript() is harder to pull off: getting current tab
// is an async operation, getting current frame is non-trivial.
//
// We propagate changes from popup to tabs via calling executeScript. The standard way of communicating between scripts
// (e.g. popup <-> content scripts) is sendMessage, but a) this is a very verbose way of doing it, b) you can't
// sendMessage into a MAIN environment on Firefox and c) you have to enumerate frames if we need to get results from all
// frames. Instead we call functions using executeScript(): in order to do that we store the references to those
// functions in globalThis, which looks like a hack but works. See ContentScriptExports and OverrideScriptExports.
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
// constructor, the ISOLATED content script is ran much later, because it needs to complete initialization of MAIN world
// content script. Unlike the ISOLATED world content script, MAIN world content script cannot access chrome.runtime APIs
// and cannot get the URLs of extension scripts/wasm files or current settings.

let settings: ExtensionSettings;

let globalAudioContext: Promise<AudioContext> | null = null;
let globalWorkletNode: Promise<AudioWorkletNode> | null = null;

// Statistics reported by the worklet processor via messages.
let workletProcessorNumUnderruns = 0;
let workletProcessorFastPathActive = true;
let workletProcessorQueueLength = 0;
let workletProcessorCurrentLatencyMs = 0;

// WeakMap would have been nice here, but it is not iterable and we have a MutationObserver anyway.
//
// Because of the unsolved https://github.com/WebAudio/web-audio-api/issues/1202 this map may contain somewhat stale
// data: we add the elements to this map when the processing is enabled, but not when it is disabled. If the extension
// is enabled after the page was loaded, we re-scan all elements and add the new ones to the map.
//
// Basically, we try not to add worklets to elements if extension is disabled, but we never remove them from active
// elements.
const nodesMap = new Map<HTMLMediaElement, ElementState>();

// The WebAudio source node is created (or not created, depending on CORS) lazily when the element first plays. Before
// that the element in registered in nodesMap, but not touched in any way.
type ElementState = { type: 'added' } | { type: 'initialized'; sourceNode: AudioNode };

// We do not immediately disconnect and remove elements in MutationObserver: they may get reparented immediately or
// after short amount of time. Instead we put the element into the "queue" and disconnect them only after delay.
const nodesPendingRemove = new Map<HTMLMediaElement, number>();
// A single timeout shared by all pending removals.
let pendingRemoveTimer: number | null = null;
const PENDING_REMOVE_DELAY_MS = 10_000;

async function initAudioContext(): Promise<AudioContext> {
    const newAudioContext = new AudioContext();
    const processorUrl = chrome.runtime.getURL('pitch-changer-processor.js');
    await newAudioContext.audioWorklet.addModule(processorUrl);
    debugLog(`Created shared AudioContext and loaded processor from ${processorUrl} in ISOLATED`);
    return newAudioContext;
}

async function getWorkletAudioContext(): Promise<AudioContext> {
    if (!globalAudioContext) {
        globalAudioContext = initAudioContext();
    }
    return globalAudioContext;
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
    const promise = new Promise<MessagePort>((res) => {
        resolve = res;
    });
    const onMessage = (event: MessageEvent) => {
        if (event.source !== workerIframe.contentWindow) {
            return;
        }
        if (event.data?.type !== 'pitch-changer-extension-worker-iframe-init') {
            throw new Error(`ISOLATED: Unknown message type from worker iframe: ${JSON.stringify(event.data)}`);
        }
        const data = event.data as WorkerIframeInit;
        resolve(data.audioProcessorClientPort);
        window.removeEventListener('message', onMessage);
    };
    window.addEventListener('message', onMessage);

    workerIframe.src = workerIframeUrl.href;
    workerIframe.hidden = true;
    document.body.appendChild(workerIframe);

    return promise;
}

async function getWorkletNode(context: AudioContext): Promise<AudioWorkletNode> {
    if (!globalWorkletNode) {
        globalWorkletNode = initWorkletNode(context);
    }
    return globalWorkletNode;
}

async function initWorkletNode(context: AudioContext): Promise<AudioWorkletNode> {
    const destChannelCount = context.destination.channelCount;
    const result = new AudioWorkletNode(context, PROCESSOR_NAME, {
        // Force the WebAudio do up/downmixing for us.
        channelCount: destChannelCount,
        channelCountMode: 'explicit',
        outputChannelCount: [destChannelCount],
        processorOptions: {
            numChannels: destChannelCount,
        } as ProcessorOptions,
    });

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

    result.port.onmessage = (event: MessageEvent<ProcessorStats>) => {
        const message = event.data;
        if (message.type !== 'pitch-changer-extension-processor-stats') {
            throw new Error(`ISOLATED: Unknown message type from processor: ${JSON.stringify(message)}`);
        }
        workletProcessorFastPathActive = message.fastPathActive;
        workletProcessorCurrentLatencyMs = message.currentLatencyMs;
        workletProcessorNumUnderruns = message.numUnderruns;
        workletProcessorQueueLength = message.queueLength;
    };

    result.connect(context.destination);

    debugLog(`Created shared worklet node, channelCount ${destChannelCount}`);
    return result;
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

// Create the MediaElementAudioSourceNode for a registered element, if it is safe to do so. Assumes that the extension
// is enabled and connects the newly created node to the worklet.
function initializeSourceNode(element: HTMLMediaElement, context: AudioContext, workletNode: AudioWorkletNode) {
    const state = nodesMap.get(element);
    if (!state || state.type === 'initialized') {
        return;
    }

    try {
        if (!isMediaElementSafeForWebAudio(element)) {
            // Leave the element playing unprocessed instead of silencing it.
            debugLog(`Not attaching worklet to CORS-unsafe element ${element.id} (${element.currentSrc})`);
            return;
        }

        const sourceNode = context.createMediaElementSource(element);
        sourceNode.connect(workletNode);
        nodesMap.set(element, { type: 'initialized', sourceNode });
        debugLog(`Added source to shared worklet for element ${element.id}, channelCount ${sourceNode.channelCount}`);
    } catch (error) {
        console.error('Error adding worklet to node', error);
    }
}

async function applyWorklet(element: HTMLMediaElement) {
    if (nodesMap.get(element)?.type === 'initialized') {
        return;
    }
    if (!settings.enabled) {
        console.error('Trying to add element when the extension is disabled', element);
        return;
    }

    try {
        const context = await getWorkletAudioContext();
        const workletNode = await getWorkletNode(context);
        if (nodesMap.get(element)?.type === 'initialized') {
            // Re-check if a concurrent applyWorklet already attached the element.
            return;
        }
        if (!element.isConnected) {
            // The element was removed from the document while we were awaiting audio context or workletNode; if it is
            // re-inserted, the observer will call applyWorklet again.
            return;
        }

        if (!nodesMap.get(element)) {
            // This event registers as user interaction in Chrome apparently.
            element.addEventListener('play', () => {
                if (!nodesMap.has(element)) {
                    // The element was detached from the document and dropped after PENDING_REMOVE_DELAY_MS, but still
                    // played: print a warning.
                    debugLog(`WARNING: Element ${element.id} (${element.currentSrc}) plays after being removed`);
                    return;
                }
                if (!settings.enabled) {
                    return;
                }
                if (context.state === 'suspended') {
                    // Required for Chrome
                    debugLog('Context got suspended, resuming');
                    context.resume();
                }
                initializeSourceNode(element, context, workletNode);
            });

            nodesMap.set(element, { type: 'added' });
            debugLog(`Registered element ${element.id}`);
        }

        // The element may already be playing (e.g. it was playing before the extension was enabled): the play event has
        // already fired, so initialize the element right away.
        if (!element.paused) {
            initializeSourceNode(element, context, workletNode);
        }
    } catch (error) {
        console.error('Error adding worklet to node', error);
    }
}

function removeWorklet(element: HTMLMediaElement) {
    const state = nodesMap.get(element);
    if (!state) {
        console.error('Removing non-added element', element);
        return;
    }

    try {
        if (state.type === 'initialized') {
            state.sourceNode.disconnect();
        }
        nodesMap.delete(element);
        debugLog(`Removed element ${element.id}`);
    } catch (error) {
        console.error('Error removing worklet from element:', error);
    }
}

function setPendingRemoveTimer() {
    if (pendingRemoveTimer !== null) {
        // The timer is already armed, skip it.
        return;
    }
    pendingRemoveTimer = setTimeout(doPendingRemove, PENDING_REMOVE_DELAY_MS);
}

function doPendingRemove() {
    pendingRemoveTimer = null;
    const now = performance.now();
    nodesPendingRemove.forEach((removedAt, element) => {
        if (removedAt + PENDING_REMOVE_DELAY_MS > now) {
            return;
        }
        nodesPendingRemove.delete(element);
        if (element.isConnected) {
            debugLog(`Element ${element.id} is connected again, keeping its worklet`);
            return;
        }
        if (!nodesMap.has(element)) {
            return;
        }
        debugLog(`Grace period for element ${element.id} expired, removing worklet`);
        removeWorklet(element);
    })

    // Re-arm the timer if the map is still non-empty.
    if (nodesPendingRemove.size > 0) {
        setPendingRemoveTimer();
    }
}

function schedulePendingRemove(element: HTMLMediaElement) {
    if (!nodesMap.has(element)) {
        return;
    }
    nodesPendingRemove.set(element, performance.now());
    setPendingRemoveTimer();
    debugLog(`Scheduled worklet removal for element ${element.id} in ${PENDING_REMOVE_DELAY_MS / 1000}s`);
}

function cancelPendingRemove(element: HTMLMediaElement) {
    if (nodesPendingRemove.delete(element)) {
        debugLog(`Canceled pending worklet removal for element ${element.id}`);
    }
}

function applyWorkletToChildren(node: Element | Document) {
    if (node instanceof HTMLMediaElement) {
        applyWorklet(node);
        return;
    }
    const elements = node.querySelectorAll('audio, video');
    if (elements.length === 0) {
        debugLog(`Found no audio/video elements in ${node}`);
        return;
    }
    debugLog(`Found ${elements.length} audio/video elements in ${node}, adding worklet`);

    elements.forEach((e) => {
        if (e instanceof HTMLMediaElement) {
            applyWorklet(e);
        } else {
            console.error('Got element which is not audio/video', e);
        }
    });
}

function schedulePendingRemoveOfChildren(node: Element) {
    if (node instanceof HTMLMediaElement) {
        schedulePendingRemove(node);
        return;
    }
    const elements = node.querySelectorAll('audio, video');
    if (elements.length === 0) {
        debugLog(`Found no audio/video elements in ${node}`);
        return;
    }
    debugLog(`Found ${elements.length} audio/video elements in ${node}, scheduling worklet removal`);

    elements.forEach((e) => {
        if (e instanceof HTMLMediaElement) {
            schedulePendingRemove(e);
        } else {
            console.error('Got element which is not audio/video', e);
        }
    });
}

function cancelPendingRemoveOfChildren(node: Element) {
    if (nodesPendingRemove.size === 0) {
        return;
    }
    if (node instanceof HTMLMediaElement) {
        cancelPendingRemove(node);
        return;
    }
    node.querySelectorAll('audio, video').forEach((e) => {
        if (e instanceof HTMLMediaElement) {
            cancelPendingRemove(e);
        }
    });
}

function applyStoredSettings() {
    if (!settings.enabled) {
        debugLog('Extension is disabled');
        return;
    }
    debugLog('Applying stored settings', settings);
    applyWorkletToChildren(document);
}

function applySettings(newSettings: ExtensionSettings) {
    debugLog(
        `Applying new settings in ISOLATED: ${JSON.stringify(newSettings)}, current settings ${JSON.stringify(settings)}`,
    );
    const gotEnabled = settings && !settings.enabled && newSettings.enabled;
    const gotDisabled = settings && settings.enabled && !newSettings.enabled;
    settings = newSettings;

    // Do not await the future, so that applySettings exits asap.
    applySettingsImpl(gotEnabled, gotDisabled);
}

async function applySettingsImpl(gotEnabled: boolean, gotDisabled: boolean) {
    if (gotEnabled) {
        debugLog('Re-enabling pitch shift for page');
        // Re-scan for any media elements that appeared while extension was disabled.
        applyWorkletToChildren(document);

        if (nodesMap.size > 0) {
            // Route all nodes through our worklet (newly added nodes in applyWorkletToChildren are already routed
            // through worklet, for those it will do a no-op).
            const context = await getWorkletAudioContext();
            const worklet = await getWorkletNode(context);
            for (const state of nodesMap.values()) {
                if (state.type !== 'initialized') {
                    continue;
                }
                state.sourceNode.disconnect();
                state.sourceNode.connect(worklet);
            }
        }
    }
    if (gotDisabled) {
        if (nodesMap.size > 0) {
            // Route all nodes through default destination.
            const context = await getWorkletAudioContext();
            for (const state of nodesMap.values()) {
                if (state.type !== 'initialized') {
                    continue;
                }
                state.sourceNode.disconnect();
                state.sourceNode.connect(context.destination);
            }
        }
    }

    // Update the parameters of the worklet if it exists.
    if (globalWorkletNode) {
        (await globalWorkletNode).port.postMessage({
            type: 'pitch-changer-extension-processor-set-params',
            processingMode: settings.processingMode,
            pitchValue: settings.pitchValue,
            targetLatency: settings.targetLatency,
            enablePassthroughOptimization: settings.enablePassthroughOptimization,
        } as ProcessorSetParams);
    }
}

function getStats(): StatsResult {
    const response: StatsResult = {
        numAudioElements: 0,
        numVideoElements: 0,
        fastPathActive: workletProcessorFastPathActive,
        currentLatencyMs: workletProcessorCurrentLatencyMs,
        numUnderruns: workletProcessorNumUnderruns,
        queueLength: workletProcessorQueueLength,
    };
    for (const node of nodesMap.keys()) {
        if (node instanceof HTMLAudioElement) {
            response.numAudioElements++;
        } else if (node instanceof HTMLVideoElement) {
            response.numVideoElements++;
        } else {
            console.error('Strange node in nodesMap', node);
        }
    }
    return response;
}

function setupExports() {
    (globalThis as unknown as ContentScriptExports).exportGetStats = getStats;
    (globalThis as unknown as ContentScriptExports).exportApplySettings = applySettings;
}

async function init(): Promise<void> {
    setupExports();

    settings = await loadSettings();
    debugLog('Audio Pitch Changer: Initializing ISOLATED content script');
    applyStoredSettings();

    // Complete initialization of the MAIN content script. Ideally we would simply use executeScript from ISOLATED
    // content script to run MAIN init, but browser does not allow getting current tab id :-(
    window.postMessage(
        {
            type: 'pitch-changer-extension-override-init',
            processorUrl: chrome.runtime.getURL('pitch-changer-processor.js'),
            workerIframeUrl: chrome.runtime.getURL('worker-iframe.html'),
            audioProcessorWorkerUrl: chrome.runtime.getURL('audio-processor-worker.js'),
            settings: settings,
        } as PitchChangerOverrideInit,
        '*',
    );

    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            // Canceling pending removals must not depend on settings.enabled: removals are scheduled regardless
            // of it, and a re-parented element must not end up disconnected after the grace period.
            if (mutation.addedNodes) {
                mutation.addedNodes.forEach((node) => {
                    if (node instanceof Element) {
                        cancelPendingRemoveOfChildren(node);
                        if (settings.enabled) {
                            applyWorkletToChildren(node);
                        }
                    }
                });
            }
            if (mutation.removedNodes) {
                mutation.removedNodes.forEach((node) => {
                    if (node instanceof Element) {
                        schedulePendingRemoveOfChildren(node);
                    }
                });
            }
        }
    });
    observer.observe(document, { subtree: true, childList: true });
}

init();
