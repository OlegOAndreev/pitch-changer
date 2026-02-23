import {
    type ApplySettingsMessage,
    type ExtensionSettings,
    type GetStatsRequest,
    type GetStatsResponse,
    type ProcessingMode,
    loadSettings,
    SETTINGS_KEY,
} from './common.js';
// Note: yes, this is an intended import
import { debounce } from '../../src/utils.js';

function debugLog(...args: unknown[]): void {
    if (currentSettings?.debugLogging) {
        console.debug(...args);
    }
}

const toggleEnabled = document.getElementById('toggleEnabled') as HTMLInputElement;
const pitchSlider = document.getElementById('pitchSlider') as HTMLInputElement;
const pitchValue = document.getElementById('pitchValue') as HTMLDivElement;
const noteValue = document.getElementById('noteValue') as HTMLDivElement;
const modeButtons = document.querySelectorAll('.mode-btn');
const advancedSection = document.getElementById('advancedSection') as HTMLDetailsElement;
const debugLoggingCheckbox = document.getElementById('debugLogging') as HTMLInputElement;
const numAudioElementsValue = document.getElementById('numAudioElements') as HTMLSpanElement;
const numVideoElementsValue = document.getElementById('numVideoElements') as HTMLSpanElement;

const SAVE_SETTINGS_DEBOUNCE = 500;

const currentSettings: ExtensionSettings = await loadSettings();

const saveSettings = debounce(SAVE_SETTINGS_DEBOUNCE, async () => {
    try {
        await chrome.storage.local.set({ [SETTINGS_KEY]: currentSettings });
    } catch (error) {
        console.error('Error saving settings:', error);
    }
});

function setEnabledClass() {
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
    if (tab.incognito) {
        return false;
    }
    const url = tab.url || '';
    if (!url.startsWith('http')) {
        return false;
    }
    return true;
}

async function applyToTabs() {
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
        if (!shouldApplyToTab(tab)) {
            continue;
        }
        debugLog(`Applying to ${tab.url}`);
        const request: ApplySettingsMessage = {
            type: 'ApplySettingsMessage',
            settings: currentSettings,
        };
        chrome.tabs.sendMessage(tab.id!, request);
    }
}

async function updateDebugStats() {
    const [tab] = await chrome.tabs.query({ active: true });
    let numAudioElements = 0;
    let numVideoElements = 0;
    if (shouldApplyToTab(tab)) {
        const frames = await chrome.webNavigation.getAllFrames({ tabId: tab.id! });
        if (frames) {
            for (const frame of frames) {
                try {
                    const request: GetStatsRequest = { type: 'GetStatsRequest' };
                    const response: GetStatsResponse = await chrome.tabs.sendMessage(tab.id!, request, {
                        frameId: frame.frameId,
                    });
                    numAudioElements += response.numAudioElements;
                    numVideoElements += response.numVideoElements;
                } catch (error) {
                    console.error('Could not message frame', frame, error);
                }
            }
        }
    }
    numAudioElementsValue.textContent = numAudioElements.toString();
    numVideoElementsValue.textContent = numVideoElements.toString();
}

async function init(): Promise<void> {
    debugLog('Audio Pitch Changer popup initialized');

    toggleEnabled.checked = currentSettings.enabled !== false;
    debugLoggingCheckbox.checked = currentSettings.debugLogging !== false;
    setEnabledClass();
    updatePitchDisplay();
    updateActiveModeDisplay(currentSettings.processingMode);

    toggleEnabled.addEventListener('change', async () => {
        currentSettings.enabled = toggleEnabled.checked;
        setEnabledClass();
        saveSettings();
        applyToTabs();
    });

    pitchSlider.addEventListener('input', () => {
        currentSettings.pitchValue = parseFloat(pitchSlider.value);
        updatePitchDisplay();
        saveSettings();
        applyToTabs();
    });

    debugLoggingCheckbox.addEventListener('change', () => {
        currentSettings.debugLogging = debugLoggingCheckbox.checked;
        saveSettings();
        applyToTabs();
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
                applyToTabs();
            } else {
                console.warn('Invalid mode:', mode);
            }
        });
    });
}

init();
