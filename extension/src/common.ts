export const SETTINGS_KEY = 'pitch-changer-settings';

export const PROCESSOR_NAME = 'pitch-changer-extension-processor';

export type ProcessingMode = 'pitch' | 'formant-preserving-pitch';

export interface ExtensionSettings {
    enabled: boolean;
    debugLogging: boolean;
    processingMode: ProcessingMode;
    pitchValue: number;
}

export interface ApplySettingsMessage {
    type: 'ApplySettingsMessage';
    settings: ExtensionSettings;
}

export interface GetStatsRequest {
    type: 'GetStatsRequest';
}

export interface GetStatsResponse {
    type: 'GetStatsResponse';
    numAudioElements: number;
    numVideoElements: number;
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
