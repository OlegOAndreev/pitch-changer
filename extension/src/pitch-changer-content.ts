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
// * The AudioWorklet overriding script must be ran as soon as possible, before the page script: the easiest way to do
//   it is by adding to manifest.json. The alternative of calling executeScript() is harder to pull off: getting current
//   tab is an async operation, getting current frame is non-trivial.
//
// * The standard way of communicating between scripts (e.g. popup <-> content scripts) is sendMessage, but a) this
//   is a very verbose way of doing it, b) you can't sendMessage into a MAIN environment on Firefox and c) you have to
//   enumerate frames if we need to get results from all frames. Instead we call functions using executeScript(). In
//   order to do that we store the references to those functions in globalThis, which looks like a hack but works.
//
// * We run processing of all audio and video elements through a single AudioWorkletNode, that the elements connect to.
//   It is lazily initialized.

let settings: ExtensionSettings;

let globalAudioContext: Promise<AudioContext> | null = null;
let globalWorkletNode: AudioWorkletNode | null = null;

// WeakMap would have been nice here, but it is not iterable and we have a MutationObserver anyway.
//
// Because of the unsolved https://github.com/WebAudio/web-audio-api/issues/1202 this map may contain somewhat stale
// data: we add the elements to this map when the processing is enabled, but not when it is disabled. If the extension
// is enabled after the page was loaded, we re-scan all elements and add the new ones to the map.
//
// Basically, we try not to add worklets to elements if extension is disabled, but we never remove them from active
// elements.
const nodesMap = new Map<HTMLMediaElement, MediaElementAudioSourceNode>();

async function initAudioContext(): Promise<AudioContext> {
    const newAudioContext = new AudioContext();
    const processorUrl = chrome.runtime.getURL('pitch-changer-processor.js');
    await newAudioContext.audioWorklet.addModule(processorUrl);
    debugLog(`Created shared AudioContext and loaded processor from ${processorUrl}`);
    return newAudioContext;
}

async function getWorkletAudioContext(): Promise<AudioContext> {
    if (!globalAudioContext) {
        globalAudioContext = initAudioContext();
    }
    return globalAudioContext;
}

function getWorkletNode(context: AudioContext): AudioWorkletNode {
    if (globalWorkletNode) {
        return globalWorkletNode;
    }

    const destChannelCount = context.destination.channelCount;

    globalWorkletNode = new AudioWorkletNode(context, PROCESSOR_NAME, {
        // Force the WebAudio do up/downmixing for us.
        channelCount: destChannelCount,
        channelCountMode: 'explicit',
        outputChannelCount: [destChannelCount],
        parameterData: {
            pitchValue: settings.pitchValue,
        },
    });
    globalWorkletNode.connect(context.destination);

    debugLog(`Created shared worklet node, channelCount ${destChannelCount}`);
    return globalWorkletNode;
}

async function applyWorklet(element: HTMLMediaElement) {
    if (nodesMap.get(element)) {
        return;
    }
    if (!settings.enabled) {
        console.error('Trying to add element when the extension is disabled', element);
        return;
    }

    try {
        const context = await getWorkletAudioContext();
        const workletNode = getWorkletNode(context);
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
        sourceNode.connect(workletNode);
        nodesMap.set(element, sourceNode);
        debugLog(`Added source to shared worklet for element ${element.id}, channelCount ${sourceNode.channelCount}`);
    } catch (error) {
        console.error('Error adding worklet to node', error);
    }
}

function removeWorklet(element: HTMLMediaElement) {
    const sourceNode = nodesMap.get(element);
    if (!sourceNode) {
        console.error('Removing non-added element', element);
        return;
    }

    try {
        sourceNode.disconnect();
        nodesMap.delete(element);
        debugLog(`Removed source from shared worklet for element ${element.id}`);
    } catch (error) {
        console.error('Error removing worklet from element:', error);
    }
}

function applyWorkletToChildren(rootNode: Element | Document) {
    const elements = rootNode.querySelectorAll('audio, video');
    if (elements.length === 0) {
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
    if (elements.length === 0) {
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
    debugLog(`Applying new settings ${JSON.stringify(newSettings)}, current settings ${JSON.stringify(settings)}`);
    const gotEnabled = settings && !settings.enabled && newSettings.enabled;
    const gotDisabled = settings && settings.enabled && !newSettings.enabled;
    settings = newSettings;

    // TODO: Run the applySettings function in the MAIN world.

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
            const worklet = getWorkletNode(context);
            for (const sourceNode of nodesMap.values()) {
                sourceNode.disconnect();
                sourceNode.connect(worklet);
            }
        }
    }
    if (gotDisabled) {
        // Route all nodes through default destination.
        const context = await getWorkletAudioContext();
        for (const sourceNode of nodesMap.values()) {
            sourceNode.disconnect();
            sourceNode.connect(context.destination);
        }
    }

    // Update the parameters of the worklet if it exists.
    if (globalWorkletNode) {
        const context = await getWorkletAudioContext();
        //@ts-expect-error AudioParamMap does not currently have full interface described in TypeScript
        (globalWorkletNode.parameters.get('pitchValue') as AudioParam).setValueAtTime(
            settings.pitchValue,
            context.currentTime,
        );
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
    debugLog('Audio Pitch Changer: Initializing ISOLATED content script');
    // TODO:
    // 1. Inject the pitch-changer-override-ac here.
    // 2. Pass the current settings and processor URL to initialize the rest (call the function?)
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
