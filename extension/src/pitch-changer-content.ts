import {
    loadSettings,
    PROCESSOR_NAME,
    type ContentScriptExports,
    type ExtensionSettings,
    type StatsResult,
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
// * We want to process all frames (it's extremely frequent that audio/video elements are inside iframes) in the tab.
//
// * The AudioWorklet overriding script must be ran as soon as possible, before all the page script: the easiest way
//   to do it is by adding to manifest.json. The alternative of calling executeScript() is harder to pull off: getting
//   current tab is an async operation, getting current frame is non-trivial.
//
// * The standard advice for communicating between scripts (e.g. popup <-> content scripts) is sendMessage, but a) this
//   is a very verbose way of doing it, b) you can't sendMessage into a MAIN environment on Firefox and c) you have to
//   enumerate frames if we need to get results from all frames. Instead we call functions using executeScript(). In
//   order to do that we store the references to those functions into globalThis, which looks like a hack but works.

let settings: ExtensionSettings;

let workletAudioContext: Promise<AudioContext> | null = null;

interface NodeWithWorklet {
    sourceNode: MediaElementAudioSourceNode;
    workletNode: AudioWorkletNode | null;
}

// WeakMap would have been nice here, but it is not iterable and we have a MutationObserver anyway.
//
// Because of the unsolved https://github.com/WebAudio/web-audio-api/issues/1202 this map may contain somewhat stale
// data: we add the elements to this map when the processing is enabled, but not when it is disabled. If the extension
// is enabled after the page was loaded, we re-scan all elements and add the new ones to this map.
//
// Basically, we try not to add worklets to elements if extension is disabled, but we never remove them from active
// elements.
const nodesMap = new Map<HTMLMediaElement, NodeWithWorklet>();

async function initAudioContext(): Promise<AudioContext> {
    const newAudioContext = new AudioContext();
    const processorUrl = chrome.runtime.getURL('pitch-changer-processor.js');
    await newAudioContext.audioWorklet.addModule(processorUrl);
    debugLog(`Created shared AudioContext and loaded processor from ${processorUrl}`);
    return newAudioContext;
}

async function getWorkletAudioContext(): Promise<AudioContext> {
    if (!workletAudioContext) {
        workletAudioContext = initAudioContext();
    }
    return workletAudioContext;
}

function createWorkletNode(context: AudioContext, sourceNode: MediaElementAudioSourceNode): AudioWorkletNode {
    const workletNode = new AudioWorkletNode(context, PROCESSOR_NAME, {
        channelCount: sourceNode.channelCount,
        outputChannelCount: [context.destination.channelCount],
        parameterData: {
            pitchValue: settings.pitchValue,
        },
    });
    sourceNode.connect(workletNode);
    workletNode.connect(context.destination);
    return workletNode;
}

async function applyWorklet(element: HTMLMediaElement) {
    if (nodesMap.get(element)) {
        return;
    }
    if (!settings.enabled) {
        console.error('Adding element when the extension is disabled', element);
        return;
    }

    try {
        const context = await getWorkletAudioContext();
        if (nodesMap.get(element)) {
            // Re-check if a concurrent applyWorklet already added the element.
            return;
        }

        // This event registers as user interaction in Chrome apparently.
        element.addEventListener('play', () => {
            if (!settings.enabled) {
                return;
            }
            if (context.state === 'suspended') {
                // Required for Chrome
                debugLog('Context got suspended, resuming');
                context.resume();
            }
        });

        const sourceNode = context.createMediaElementSource(element);
        const workletNode = createWorkletNode(context, sourceNode);
        nodesMap.set(element, { sourceNode: sourceNode, workletNode: workletNode });
        debugLog(
            `Added worklet to element ${element.id}, channelCount ${sourceNode.channelCount}, outputChannelCount ${context.destination.channelCount}`,
        );
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
        state.sourceNode.disconnect();
        if (state.workletNode) {
            state.workletNode.disconnect();
        }

        nodesMap.delete(element);
        debugLog(`Removed worklet from element ${element.id}`);
    } catch (error) {
        console.error('Error removing worklet from element:', error);
    }
}

function applyWorkletToChildren(rootNode: Element | Document) {
    const elements = rootNode.querySelectorAll('audio, video');
    if (elements.length == 0) {
        debugLog(`Found no audio/video elements in ${rootNode}`);
        return;
    }
    debugLog(`Found ${elements.length} audio/video elements in ${rootNode}, adding worklet`);

    elements.forEach(async (e) => {
        if (e instanceof HTMLMediaElement) {
            if (!nodesMap.get(e)) {
                await applyWorklet(e);
            }
        } else {
            console.error('Got element which is not audio/video', e);
        }
    });
}

function removeWorkletFromChildren(rootNode: Element) {
    const elements = rootNode.querySelectorAll('audio, video');
    if (elements.length == 0) {
        debugLog(`Found no audio/video elements in ${rootNode}`);
        return;
    }
    debugLog(`Found ${elements.length} audio/video elements in ${rootNode}, removing worklet`);

    elements.forEach(async (e) => {
        if (e instanceof HTMLMediaElement) {
            if (nodesMap.get(e)) {
                removeWorklet(e);
            }
        } else {
            console.error('Got element which is not audio/video', e);
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
    debugLog('Applying settings', newSettings);
    const gotEnabled = !settings.enabled && newSettings.enabled;
    settings = newSettings;

    // Run the rest in the background so that this function exits as soon as possible.
    setTimeout(() => applySettingsImpl(gotEnabled), 0);
}

async function applySettingsImpl(gotEnabled: boolean) {
    // Re-scan the HTML elements to find new audio/video elements not in nodesMap.
    if (gotEnabled) {
        debugLog('Enabling pitch shift for page');
        applyWorkletToChildren(document);
    }

    // Apply the new pitch value to the worklet nodes and creates worklet nodes if required.
    const context = await getWorkletAudioContext();
    const currentTime = context.currentTime;
    for (const node of nodesMap.values()) {
        if (settings.enabled) {
            if (!node.workletNode) {
                node.sourceNode.disconnect();
                node.workletNode = createWorkletNode(context, node.sourceNode);
            } else {
                //@ts-expect-error AudioParamMap does not currently have full interface described in TypeScript
                (node.workletNode.parameters.get('pitchValue') as AudioParam).setValueAtTime(
                    settings.pitchValue,
                    currentTime,
                );
            }
        } else {
            if (node.workletNode) {
                node.sourceNode.disconnect();
                node.sourceNode.connect(context.destination);
                node.workletNode = null;
            }
        }
    }
}

function getStats(): StatsResult {
    const response: StatsResult = {
        numAudioElements: 0,
        numVideoElements: 0,
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
    debugLog('Audio Pitch Changer: Initializing main content script');

    applyStoredSettings();

    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (settings.enabled && mutation.addedNodes) {
                mutation.addedNodes.forEach((node) => {
                    if (node instanceof Element) {
                        applyWorkletToChildren(node);
                    }
                });
            }
            if (mutation.removedNodes) {
                mutation.removedNodes.forEach((node) => {
                    if (node instanceof Element) {
                        removeWorkletFromChildren(node);
                    }
                });
            }
        }
    });

    observer.observe(document, { subtree: true, childList: true });
}

init();
