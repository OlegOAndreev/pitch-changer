export const SETTINGS_KEY = 'pitch-changer-settings';

export const PROCESSOR_NAME = 'pitch-changer-extension-processor';

export type ProcessingMode = 'pitch' | 'formant-preserving-pitch';

// Latency matters if your system is under load: the chances of skipping/cracks are higher with lower latencies.
export type TargetLatency = 'normal' | 'high';

export interface ExtensionSettings {
    enabled: boolean;
    debugLogging: boolean;
    processingMode: ProcessingMode;
    pitchValue: number;
    targetLatency: TargetLatency;
    enablePassthroughOptimization: boolean;
}

export interface StatsResult {
    type: 'pitch-changer-extension-stats-result';
    numAudioContexts: number;
    numPlayingElements: number;
    fastPathActive: boolean;
    currentLatencyMs: number;
    numUnderruns: number;
    queueLength: number;
}

const defaultSettings: ExtensionSettings = {
    enabled: true,
    debugLogging: false,
    processingMode: 'pitch',
    pitchValue: 1.0,
    targetLatency: 'normal',
    enablePassthroughOptimization: true,
};

export interface MessageWithType {
    type: string;
}

export interface MainWorldInit {
    type: 'pitch-changer-extension-main-world-init';
    processorUrl: string;
    workerIframeUrl: string;
    audioProcessorWorkerUrl: string;
    settings: ExtensionSettings;
}

export interface ApplySettings {
    type: 'pitch-changer-extension-apply-settings';
    settings: ExtensionSettings;
}

export interface GetStats {
    type: 'pitch-changer-extension-get-stats';
}

export interface WorkerIframeInit {
    type: 'pitch-changer-extension-worker-iframe-init';
    audioProcessorClientPort: MessagePort;
}

export interface ProcessorInit {
    type: 'pitch-changer-extension-processor-init';
    numChannels: number;
    audioProcessorClientPort: MessagePort;
}

export interface ProcessorSetParams {
    type: 'pitch-changer-extension-processor-set-params';
    processingMode: ProcessingMode;
    pitchValue: number;
    targetLatency: TargetLatency;
    enablePassthroughOptimization: boolean;
}

export interface ProcessorStats {
    type: 'pitch-changer-extension-processor-stats';
    fastPathActive: boolean;
    currentLatencyMs: number;
    numUnderruns: number;
    queueLength: number;
}

export type ProcessorRequest = ProcessorInit | ProcessorSetParams;

// Initialize settings with default values, but leave enabled: false.
export function disabledSettings(): ExtensionSettings {
    return {
        ...defaultSettings,
        enabled: false,
    };
}

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

export function isMessageWithType(data: unknown): data is MessageWithType {
    return typeof data === 'object' && data !== null && 'type' in data;
}
