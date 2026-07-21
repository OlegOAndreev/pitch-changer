import {
    type ContentScriptExports,
    type ExtensionSettings,
    type OverrideScriptExports,
    type OverrideStatsResult,
    type ProcessingMode,
    type StatsResult,
    loadSettings,
    SETTINGS_KEY,
} from './common.js';
// Note: yes, this is an intended import
import { debounce } from '../../src/common-utils.js';

const currentSettings: ExtensionSettings = await loadSettings();

function debugLog(...args: unknown[]): void {
    if (currentSettings.debugLogging) {
        console.debug(...args);
    }
}

const statusValue = document.getElementById('status') as HTMLDivElement;
const toggleEnabled = document.getElementById('toggleEnabled') as HTMLInputElement;
const pitchSlider = document.getElementById('pitchSlider') as HTMLInputElement;
const pitchValue = document.getElementById('pitchValue') as HTMLDivElement;
const noteValue = document.getElementById('noteValue') as HTMLDivElement;
const modeButtons = document.querySelectorAll('.mode-btn');
const advancedSection = document.getElementById('advancedSection') as HTMLDetailsElement;
const debugLoggingCheckbox = document.getElementById('debugLogging') as HTMLInputElement;
const numAudioElementsValue = document.getElementById('numAudioElements') as HTMLSpanElement;
const numVideoElementsValue = document.getElementById('numVideoElements') as HTMLSpanElement;
const numAudioContextDestinationsValue = document.getElementById('numAudioContextDestinations') as HTMLSpanElement;

const SAVE_SETTINGS_DEBOUNCE = 50;
const HIDE_ERROR_AFTER = 10000;

function showStatus(message: string) {
    statusValue.style.display = 'flex';
    statusValue.textContent = message;
    setTimeout(() => {
        statusValue.style.display = 'none';
    }, HIDE_ERROR_AFTER);
}

const saveSettings = debounce(SAVE_SETTINGS_DEBOUNCE, async () => {
    try {
        await chrome.storage.local.set({ [SETTINGS_KEY]: currentSettings });
    } catch (error) {
        console.error('Error saving settings:', error);
        showStatus(error as string);
    }
});

function setEnabled() {
    if (currentSettings.enabled) {
        modeButtons.forEach((btn) => btn.classList.remove('disabled'));
        pitchSlider.classList.remove('disabled');
    } else {
        modeButtons.forEach((btn) => btn.classList.add('disabled'));
        pitchSlider.classList.add('disabled');
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

function shouldApplyToTab(tab: chrome.tabs.Tab): boolean {
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
        try {
            chrome.scripting.executeScript({
                func: (settings) => {
                    const applySettings = (globalThis as unknown as ContentScriptExports).exportApplySettings;
                    // Skip the frames we did not get injected into for whatever reason.
                    if (applySettings) {
                        applySettings(settings);
                    }
                },
                args: [currentSettings],
                target: { tabId: tab.id!, allFrames: true },
                world: 'ISOLATED',
            });
            chrome.scripting.executeScript({
                func: (settings) => {
                    const applySettings = (globalThis as unknown as OverrideScriptExports).exportPitchChangerOverrideApplySettings;
                    // Skip the frames we did not get injected into for whatever reason.
                    if (applySettings) {
                        applySettings(settings);
                    }
                },
                args: [currentSettings],
                target: { tabId: tab.id!, allFrames: true },
                world: 'MAIN',
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            // Injection into restricted/quarantined hosts always fails; this is expected, so just log it.
            if (message.includes('Cannot access contents of the page')) {
                console.debug(`Could not apply settings to tab ${tab.url}, skipping`, error)
            } else {
                console.error(`Could not apply settings to tab ${tab.url} ISOLATED`, error);
                showStatus(error as string);
            }
        }
    }
}

async function updateDebugStats() {
    const [tab] = await chrome.tabs.query({ active: true });
    let numAudioElements = 0;
    let numVideoElements = 0;
    let numAudioContextDestinations = 0;
    if (shouldApplyToTab(tab)) {
        console.debug('Running for tab', tab);
        try {
            // We need to call two functions: one in pitch-changer-content.ts and one in pitch-changer-override.ac.ts
            const results = await chrome.scripting.executeScript({
                func: () => {
                    const getStats = (globalThis as unknown as ContentScriptExports).exportGetStats;
                    // Skip the frames we did not get injected into for whatever reason.
                    if (getStats) {
                        return getStats();
                    } else {
                        return { numAudioElements: 0, numVideoElements: 0 } as StatsResult;
                    }
                },
                target: { tabId: tab.id!, allFrames: true },
                world: 'ISOLATED',
                // Set to true, otherwise the execution may hang for a looong time while the various iframes are being
                // loaded.
                injectImmediately: true,
            });
            for (const result of results) {
                const data = result.result as StatsResult;
                numAudioElements += data.numAudioElements;
                numVideoElements += data.numVideoElements;
            }

            const overrideResults = await chrome.scripting.executeScript({
                func: () => {
                    const getStats = (globalThis as unknown as OverrideScriptExports)
                        .exportPitchChangerOverrideGetStats;
                    // Skip the frames we did not get injected into for whatever reason.
                    if (getStats) {
                        return getStats();
                    } else {
                        return { numAudioContexts: 0 } as OverrideStatsResult;
                    }
                },
                target: { tabId: tab.id!, allFrames: true },
                world: 'MAIN',
                // Set to true, otherwise the execution may hang for a looong time while the various iframes are being
                // loaded.
                injectImmediately: true,
            });
            for (const result of overrideResults) {
                const data = result.result as OverrideStatsResult;
                numAudioContextDestinations += data.numAudioContexts;
            }
        } catch (error) {
            console.error(`Could not get stats from tab ${tab.url}`, error);
            showStatus(error as string);
        }
    }
    numAudioElementsValue.textContent = numAudioElements.toString();
    numVideoElementsValue.textContent = numVideoElements.toString();
    numAudioContextDestinationsValue.textContent = numAudioContextDestinations.toString();
}

async function init(): Promise<void> {
    debugLog('Audio Pitch Changer popup initialized');

    toggleEnabled.checked = currentSettings.enabled !== false;
    debugLoggingCheckbox.checked = currentSettings.debugLogging !== false;
    setEnabled();
    updatePitchDisplay();
    updateActiveModeDisplay(currentSettings.processingMode);

    toggleEnabled.addEventListener('change', async () => {
        currentSettings.enabled = toggleEnabled.checked;
        setEnabled();
        saveSettings();
        applySettingsToTabs();
    });

    pitchSlider.addEventListener('input', () => {
        currentSettings.pitchValue = parseFloat(pitchSlider.value);
        updatePitchDisplay();
        saveSettings();
        applySettingsToTabs();
    });

    debugLoggingCheckbox.addEventListener('change', () => {
        currentSettings.debugLogging = debugLoggingCheckbox.checked;
        saveSettings();
        applySettingsToTabs();
    });

    advancedSection.addEventListener('toggle', () => {
        if (advancedSection.open) {
            updateDebugStats();
        }
    });

    modeButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const mode = (btn as HTMLButtonElement).dataset.mode;
            if (mode === 'pitch' || mode === 'formant-preserving-pitch') {
                currentSettings.processingMode = mode;
                updateActiveModeDisplay(mode);
                saveSettings();
                applySettingsToTabs();
            } else {
                console.warn('Invalid mode:', mode);
            }
        });
    });
}

init();
