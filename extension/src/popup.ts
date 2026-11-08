import {
    type ExtensionSettings,
    type ProcessingMode,
    type StatsResult,
    type TargetLatency,
    loadSettings,
    SETTINGS_KEY,
    type ApplySettings,
    type GetStats,
    type Quality,
} from './common.js';
// Note: yes, this is an intended import
import { anyToString, debounce, setupWindowOnError, sleep } from '../../src/common-utils.js';

const currentSettings: ExtensionSettings = await loadSettings();

function debugLog(...args: unknown[]): void {
    if (currentSettings.debugLogging) {
        console.debug(...args);
    }
}

const toggleEnabled = document.getElementById('toggleEnabled') as HTMLInputElement;
const pitchSlider = document.getElementById('pitchSlider') as HTMLInputElement;
const pitchValue = document.getElementById('pitchValue') as HTMLDivElement;
const noteValue = document.getElementById('noteValue') as HTMLDivElement;
const modeButtons = document.querySelectorAll('#modeSelector button');
const latencyButtons = document.querySelectorAll('#latencySelector button');
const qualityButtons = document.querySelectorAll('#qualitySelector button');
const advancedSection = document.getElementById('advancedSection') as HTMLDetailsElement;
const passthroughCheckbox = document.getElementById('enablePassthroughOptimization') as HTMLInputElement;
const debugLoggingCheckbox = document.getElementById('debugLogging') as HTMLInputElement;
const numPlayingElementsValue = document.getElementById('numPlayingElements') as HTMLSpanElement;
const numAudioContextDestinationsValue = document.getElementById('numAudioContextDestinations') as HTMLSpanElement;
const fastPathActiveValue = document.getElementById('fastPathActive') as HTMLSpanElement;
const maxCurrentLatencyMsValue = document.getElementById('maxCurrentLatencyMs') as HTMLSpanElement;
const numUnderrunsValue = document.getElementById('numUnderruns') as HTMLSpanElement;
const maxQueueLengthValue = document.getElementById('maxQueueLength') as HTMLSpanElement;

const SAVE_SETTINGS_DEBOUNCE = 50;
const STATS_UPDATE_INTERVAL = 100;

const saveSettings = debounce(SAVE_SETTINGS_DEBOUNCE, async () => {
    try {
        await chrome.storage.local.set({ [SETTINGS_KEY]: currentSettings });
    } catch (error) {
        console.error('Error saving settings:', error);
    }
});

function setEnabled() {
    if (currentSettings.enabled) {
        modeButtons.forEach((btn) => btn.classList.remove('disabled'));
        latencyButtons.forEach((btn) => btn.classList.remove('disabled'));
        qualityButtons.forEach((btn) => btn.classList.remove('disabled'));
        pitchSlider.classList.remove('disabled');
        passthroughCheckbox.classList.remove('disabled');
        debugLoggingCheckbox.classList.remove('disabled');
    } else {
        modeButtons.forEach((btn) => btn.classList.add('disabled'));
        latencyButtons.forEach((btn) => btn.classList.add('disabled'));
        qualityButtons.forEach((btn) => btn.classList.add('disabled'));
        pitchSlider.classList.add('disabled');
        passthroughCheckbox.classList.add('disabled');
        debugLoggingCheckbox.classList.add('disabled');
    }
}

// See https://en.wikipedia.org/wiki/Piano_key_frequencies
function frequencyToNote(frequency: number): string {
    const noteNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'];
    const noteNumber = Math.round(12 * Math.log2(frequency / 440) + 49);
    const noteIdx = (noteNumber - 1) % noteNames.length;
    const octave = Math.floor((noteNumber + 8) / noteNames.length);
    return `${noteNames[noteIdx]}${octave}`;
}

function updatePitchDisplay(): void {
    pitchValue.textContent = `${currentSettings.pitchValue.toFixed(2)}x`;
    pitchSlider.value = currentSettings.pitchValue.toString();

    const baseFrequency = 440;
    const resultingFrequency = baseFrequency * currentSettings.pitchValue;
    noteValue.textContent = `${frequencyToNote(baseFrequency)} → ${frequencyToNote(resultingFrequency)}`;
}

function updateActiveModeDisplay(mode: ProcessingMode): void {
    modeButtons.forEach((btn) => {
        const button = btn as HTMLButtonElement;
        if (button.dataset.mode === mode) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

function updateActiveLatencyDisplay(latency: TargetLatency): void {
    latencyButtons.forEach((btn) => {
        const button = btn as HTMLButtonElement;
        if (button.dataset.latency === latency) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

function updateQualityDisplay(quality: Quality): void {
    qualityButtons.forEach((btn) => {
        const button = btn as HTMLButtonElement;
        if (button.dataset.quality === quality) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

function shouldApplyToTab(tab: chrome.tabs.Tab | undefined): boolean {
    if (!tab) {
        return false;
    }
    if (!tab.id) {
        return false;
    }
    // See https://stackoverflow.com/questions/70948787/how-to-detect-if-a-tab-is-unloaded-after-you-restart-the-chrome-browser
    if (tab.discarded || tab.status === 'unloaded') {
        return false;
    }
    // See https://github.com/w3c/webextensions/issues/527
    if (tab.frozen) {
        return false;
    }
    if (tab.incognito) {
        return false;
    }
    const url = tab.url || '';
    if (!url.startsWith('http')) {
        return false;
    }
    return true;
}

async function applySettingsToTabs() {
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
        if (!shouldApplyToTab(tab)) {
            continue;
        }

        debugLog(`Applying to ${tab.url}`);
        chrome.tabs
            .sendMessage(tab.id!, {
                type: 'pitchChangerExtensionApplySettings',
                settings: currentSettings,
            } satisfies ApplySettings)
            .catch((error: unknown) => {
                const message = anyToString(error);
                // Skip the pages which have not been properly initialized.
                if (message.includes('Receiving end does not exist')) {
                    console.debug(`Could not apply settings to tab ${tab.url}, skipping`, error);
                } else {
                    console.error(`Could not apply settings to tab ${tab.url}, ${anyToString(error)}`, error);
                }
            });
    }
}

async function updateDebugStats() {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    let numPlayingElements = 0;
    let numAudioContextDestinations = 0;
    let fastPathActive = true;
    let maxCurrentLatencyMs = 0;
    let numUnderruns = 0;
    let maxQueueLength = 0;
    if (shouldApplyToTab(tab)) {
        chrome.tabs
            .sendMessage(tab.id!, {
                type: 'pitchChangerExtensionGetStats',
            } satisfies GetStats)
            .catch((error: unknown) => {
                const message = anyToString(error);
                // Skip the pages which have not been properly initialized.
                if (message.includes('Receiving end does not exist')) {
                    console.debug(`Could not apply settings to tab ${tab.url}, skipping`, error);
                } else {
                    console.error(`Could not apply settings to tab ${tab.url}, ${anyToString(error)}`, error);
                }
            });

        // This is hack: we request stats from all frames and then wait for some fixed amount of time until they
        // respond.
        const onMessage = (data: StatsResult) => {
            if (data.type !== 'pitchChangerExtensionStatsResult') {
                return;
            }
            numPlayingElements += data.numPlayingElements;
            numAudioContextDestinations += data.numAudioContexts;
            fastPathActive = fastPathActive && data.fastPathActive;
            maxCurrentLatencyMs = Math.max(maxCurrentLatencyMs, data.currentLatencyMs);
            numUnderruns += data.numUnderruns;
            maxQueueLength = Math.max(maxQueueLength, data.queueLength);
        };
        chrome.runtime.onMessage.addListener(onMessage);
        await sleep(STATS_UPDATE_INTERVAL / 2);
        chrome.runtime.onMessage.removeListener(onMessage);
    }
    numPlayingElementsValue.textContent = numPlayingElements.toString();
    numAudioContextDestinationsValue.textContent = numAudioContextDestinations.toString();
    fastPathActiveValue.textContent = fastPathActive.toString();
    maxCurrentLatencyMsValue.textContent = Math.round(maxCurrentLatencyMs).toString();
    numUnderrunsValue.textContent = numUnderruns.toString();
    maxQueueLengthValue.textContent = maxQueueLength.toString();
}

function init() {
    debugLog('Audio Pitch Changer popup initialized');
    setupWindowOnError();

    toggleEnabled.checked = currentSettings.enabled;
    passthroughCheckbox.checked = currentSettings.enablePassthroughOptimization;
    debugLoggingCheckbox.checked = currentSettings.debugLogging;
    setEnabled();
    updatePitchDisplay();
    updateActiveModeDisplay(currentSettings.processingMode);
    updateActiveLatencyDisplay(currentSettings.targetLatency);
    updateQualityDisplay(currentSettings.quality);

    toggleEnabled.addEventListener('change', async () => {
        currentSettings.enabled = toggleEnabled.checked;
        setEnabled();
        await applySettingsToTabs();
        // Do not await saveSettings because it is wrapped in debounce.
        void saveSettings();
    });

    pitchSlider.addEventListener('input', async () => {
        currentSettings.pitchValue = parseFloat(pitchSlider.value);
        updatePitchDisplay();
        await applySettingsToTabs();
        // Do not await saveSettings because it is wrapped in debounce.
        void saveSettings();
    });

    debugLoggingCheckbox.addEventListener('change', async () => {
        currentSettings.debugLogging = debugLoggingCheckbox.checked;
        await applySettingsToTabs();
        // Do not await saveSettings because it is wrapped in debounce.
        void saveSettings();
    });

    passthroughCheckbox.addEventListener('change', async () => {
        currentSettings.enablePassthroughOptimization = passthroughCheckbox.checked;
        await applySettingsToTabs();
        // Do not await saveSettings because it is wrapped in debounce.
        void saveSettings();
    });

    advancedSection.addEventListener('toggle', async () => {
        if (advancedSection.open) {
            await updateDebugStats();
        }
    });

    setInterval(async () => {
        if (advancedSection.open) {
            await updateDebugStats();
        }
    }, STATS_UPDATE_INTERVAL);

    modeButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            const mode = (btn as HTMLButtonElement).dataset.mode;
            if (mode === 'pitch' || mode === 'formant-preserving-pitch') {
                currentSettings.processingMode = mode;
                updateActiveModeDisplay(mode);
                await applySettingsToTabs();
                // Do not await saveSettings because it is wrapped in debounce.
                void saveSettings();
            } else {
                console.warn('Invalid mode:', mode);
            }
        });
    });

    latencyButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            const latency = (btn as HTMLButtonElement).dataset.latency;
            if (latency === 'normal' || latency === 'high') {
                currentSettings.targetLatency = latency;
                updateActiveLatencyDisplay(latency);
                await applySettingsToTabs();
                // Do not await saveSettings because it is wrapped in debounce.
                void saveSettings();
            } else {
                console.warn('Invalid latency:', latency);
            }
        });
    });

    qualityButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            const quality = (btn as HTMLButtonElement).dataset.quality;
            if (quality === 'low' || quality === 'high') {
                currentSettings.quality = quality;
                updateQualityDisplay(quality);
                await applySettingsToTabs();
                // Do not await saveSettings because it is wrapped in debounce.
                void saveSettings();
            } else {
                console.warn('Invalid quality:', quality);
            }
        });
    });
}

init();
