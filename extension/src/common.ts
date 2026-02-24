export const SETTINGS_KEY = 'pitch-changer-settings';

export const PROCESSOR_NAME = 'pitch-changer-extension-processor';

export type ProcessingMode = 'pitch' | 'formant-preserving-pitch';

export interface ExtensionSettings {
    enabled: boolean;
    debugLogging: boolean;
    processingMode: ProcessingMode;
    pitchValue: number;
}

export interface StatsResult {
    numAudioElements: number;
    numVideoElements: number;
}

export interface OverrideStatsResult {
    numAudioContextDestinations: number;
}

const defaultSettings: ExtensionSettings = {
    enabled: true,
    debugLogging: false,
    processingMode: 'pitch',
    pitchValue: 1.0,
};

export async function loadSettings(): Promise<ExtensionSettings> {
    const result = structuredClone(defaultSettings);
    try {
        const stored = await chrome.storage.local.get(SETTINGS_KEY);
        if (stored[SETTINGS_KEY]) {
            Object.assign(result, stored[SETTINGS_KEY]);
        }
    } catch (error) {
        console.error('Error loading settings:', error);
    }
    return result;
}

// The following two interfaces are hacks for allowing calling functions in content scripts from popup script. We store
// function pointers in globalThis properties via those interfaces. An alternative would be either a) passing messages
// or b) exporting those functions using modules. Both options require more code and look more fragile.
export interface ContentScriptExports {
    exportGetStats?(): StatsResult;
    exportApplySettings?(newSettings: ExtensionSettings): void;
}

export interface OverrideScriptExports {
    // The function name must be unique: we are injecting into global user-visible namespace.
    exportPitchShifterOverrideGetStats?(): OverrideStatsResult;
}
