import {
    type ExtensionSettings,
    loadSettings,
    PROCESSOR_NAME,
    type ApplySettingsMessage,
    type GetStatsResponse,
} from './common.js';

function debugLog(...args: unknown[]): void {
    if (settings?.debugLogging) {
        console.debug(...args);
    }
}

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
        const workletNode = new AudioWorkletNode(context, PROCESSOR_NAME, {
            channelCount: sourceNode.channelCount,
            outputChannelCount: [context.destination.channelCount],
            parameterData: {
                pitchValue: settings.pitchValue,
            },
        });
        sourceNode.connect(workletNode);
        workletNode.connect(context.destination);
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

async function applyParameters() {
    debugLog('Applying settings', settings);
    const context = await getWorkletAudioContext();
    const currentTime = context.currentTime;
    for (const node of nodesMap.values()) {
        if (settings.enabled) {
            if (!node.workletNode) {
                node.workletNode = new AudioWorkletNode(context, PROCESSOR_NAME, {
                    channelCount: node.sourceNode.channelCount,
                    outputChannelCount: [context.destination.channelCount],
                    parameterData: {
                        pitchValue: settings.pitchValue,
                    },
                });
                node.sourceNode.disconnect();
                node.sourceNode.connect(node.workletNode);
                node.workletNode.connect(context.destination);
            } else {
                //@ts-expect-error AudioParamMap does not currently have full interface in TypeScript
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

async function init(): Promise<void> {
    settings = await loadSettings();
    debugLog('Audio Pitch Changer: Initializing content script');

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

chrome.runtime.onMessage.addListener(async (rawMessage, _sender, sendResponse) => {
    switch (rawMessage.type) {
        case 'ApplySettingsMessage': {
            const message = rawMessage as ApplySettingsMessage;
            const gotEnabled = !settings.enabled && message.settings.enabled;
            settings = message.settings;
            if (gotEnabled) {
                debugLog('Enabling pitch shift for page');
                applyWorkletToChildren(document);
            }
            // Do not remove elements when extension got disabled: there is currently no way to re-route processing
            await applyParameters();
            break;
        }

        case 'GetStatsRequest': {
            const response: GetStatsResponse = {
                type: 'GetStatsResponse',
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
            debugLog(`Sending response from ${document.URL}`, response);
            sendResponse(response);
            break;
        }

        default:
            console.log('pitch-changer: Unknown message type:', rawMessage.type);
    }
    return true;
});
