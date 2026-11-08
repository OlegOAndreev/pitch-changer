export const SETTINGS_KEY = 'pitch-changer-settings';

export const PROCESSOR_NAME = 'pitch-changer-extension-processor';

export type ProcessingMode = 'pitch' | 'formant-preserving-pitch';

// Latency matters if your system is under load: the chances of skipping/cracks are higher with lower latencies.
export type TargetLatency = 'normal' | 'high';

export type Quality = 'low' | 'high';

export interface ExtensionSettings {
    enabled: boolean;
    debugLogging: boolean;
    processingMode: ProcessingMode;
    pitchValue: number;
    targetLatency: TargetLatency;
    quality: Quality;
    enablePassthroughOptimization: boolean;
}

export interface StatsResult {
    type: 'pitchChangerExtensionStatsResult';
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
    quality: 'high',
    enablePassthroughOptimization: true,
};

export interface MessageWithType {
    type: string;
}

export interface MainWorldInit {
    type: 'pitchChangerExtensionMainWorldInit';
    processorUrl: string;
    workerIframeUrl: string;
    audioProcessorWorkerUrl: string;
    settings: ExtensionSettings;
}

export interface ApplySettings {
    type: 'pitchChangerExtensionApplySettings';
    settings: ExtensionSettings;
}

export interface GetStats {
    type: 'pitchChangerExtensionGetStats';
}

export interface WorkerIframeInit {
    type: 'pitchChangerExtensionWorkerIframeInit';
    audioProcessorClientPort: MessagePort;
}

export interface ProcessorInit {
    type: 'pitchChangerExtensionProcessorInit';
    numChannels: number;
    audioProcessorClientPort: MessagePort;
}

export interface ProcessorSetParams {
    type: 'pitchChangerExtensionProcessorSetParams';
    processingMode: ProcessingMode;
    pitchValue: number;
    targetLatency: TargetLatency;
    quality: Quality;
    enablePassthroughOptimization: boolean;
}

export interface ProcessorStats {
    type: 'pitchChangerExtensionProcessorStats';
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
