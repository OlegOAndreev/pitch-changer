import { Recorder } from './recorder';

import MainModuleFactory from '../wasm/build/main-wasm-module';
import { encodeToBlob } from './media-encoder';

await MainModuleFactory();

// Settings interface and implementation
interface AppSettings {
    processingMode: 'pitch' | 'time';
    pitchValue: number;
}

// Lazily initialize audio context to prevent warning logs in Firefox.
let globalAudioContext: AudioContext;
function getAudioContext(): AudioContext {
    if (!globalAudioContext) {
        globalAudioContext = new AudioContext();
        console.log(`AudioContext created, sample rate ${globalAudioContext.sampleRate}Hz`);
    }
    return globalAudioContext;
}

let globalRecorder: Recorder;
async function getRecorder(): Promise<Recorder> {
    if (!globalRecorder) {
        globalRecorder = await Recorder.create(getAudioContext());
    }
    return globalRecorder;
}

let sourceData: Float32Array;
let sourceSampleRate: number;

let isPlaying = false;
let playingBufferSource: AudioBufferSourceNode | null;

function getById<T extends HTMLElement>(elementId: string): T {
    const element = document.getElementById(elementId);
    if (!element) {
        throw new Error(`Element with id "${elementId}" not found`);
    }
    return element as T;
}

const contentContainer = getById<HTMLDivElement>('content-сontainer');
const recordBtn = getById<HTMLButtonElement>('record-btn');
const recordBtnEmoji = getById<HTMLButtonElement>('record-btn-emoji');
const playBtn = getById<HTMLButtonElement>('play-btn');
const playBtnEmoji = getById<HTMLButtonElement>('play-btn-emoji');
const loadBtn = getById<HTMLButtonElement>('load-btn');
const saveBtn = getById<HTMLButtonElement>('save-btn');
const pitchSlider = getById<HTMLInputElement>('pitch-slider');
const pitchLabel = getById<HTMLElement>('pitch-label');
const pitchModeRadio = getById<HTMLInputElement>('pitch-mode');
const timeModeRadio = getById<HTMLInputElement>('time-mode');

const messageLabel = getById<HTMLElement>('message-label');

const fileInput = getById<HTMLInputElement>('file-input');

const saveDialogOverlay = getById<HTMLDivElement>('save-dialog-overlay');
const saveInput = getById<HTMLInputElement>('save-filename-input');
const saveOkBtn = getById<HTMLButtonElement>('save-dialog-ok-btn');
const saveCancelBtn = getById<HTMLButtonElement>('save-dialog-cancel-btn');

const SETTINGS_KEY = 'pitch-changer-settings';
const settings = loadSettings();

function loadSettings(): AppSettings {
    const settings: AppSettings = {
        processingMode: 'pitch',
        pitchValue: 1.25
    };
    Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}'));
    if (settings.processingMode !== 'time') {
        settings.processingMode = 'pitch';
    }
    pitchModeRadio.checked = settings.processingMode === 'pitch';
    timeModeRadio.checked = settings.processingMode === 'time';
    if (settings.pitchValue > 2.0 || settings.pitchValue < 0.5) {
        settings.pitchValue = 1.25;
    }
    pitchSlider.value = settings.pitchValue.toString();
    pitchLabel.textContent = settings.pitchValue + 'x';

    // Show the content container after settings are applied
    contentContainer.style.visibility = 'visible';

    return settings;
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

async function startPlayback() {
    if (!sourceData) {
        console.log('Error: no audio data to play');
        return;
    }

    const audioContext = getAudioContext();
    // At least Chrome requires this
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    const audioBuffer = audioContext.createBuffer(1, sourceData.length, sourceSampleRate);
    audioBuffer.getChannelData(0).set(sourceData);
    playingBufferSource = audioContext.createBufferSource();
    playingBufferSource.buffer = audioBuffer;
    playingBufferSource.connect(audioContext.destination);
    playingBufferSource.onended = (_e: Event) => stopPlayback();
    playingBufferSource.start();
    isPlaying = true;

    console.log(`Playing ${audioBuffer.length} samples at ${audioBuffer.sampleRate}Hz`);

    playBtnEmoji.textContent = '⏹';
    playBtn.title = 'Pause';
    playBtn.classList.add('playing');
}

function stopPlayback() {
    if (!isPlaying) {
        return;
    }

    // No need to disconnect the source: "The nodes will automatically get disconnected from the graph and will be
    // deleted when they have no more references" from https://webaudio.github.io/web-audio-api/#dynamic-lifetime-background
    playingBufferSource!.stop();
    playingBufferSource = null;
    isPlaying = false;

    console.log('Stopped playing audio');

    playBtnEmoji.textContent = '▶';
    playBtn.title = 'Play';
    playBtn.classList.remove('playing');
}

recordBtn.addEventListener('click', async () => {
    const recorder = await getRecorder();
    if (!recorder.isRecording) {
        stopPlayback();
        try {
            await recorder.start();
        } catch (error) {
            console.error('Could not start recording:', error);
            alert(`Could not start recording: ${(error as Error).message}`);
        }

        recordBtnEmoji.textContent = '⏹';
        recordBtn.title = 'Stop';
        recordBtn.classList.add('recording');
        playBtn.disabled = true;
        loadBtn.disabled = true;
        saveBtn.disabled = true;
    } else {
        sourceData = await recorder.stop();
        sourceSampleRate = getAudioContext().sampleRate;

        recordBtnEmoji.textContent = '⏺';
        recordBtn.title = 'Record';
        recordBtn.classList.remove('recording');
        playBtn.disabled = false;
        loadBtn.disabled = false;
        saveBtn.disabled = false;
    }
});

playBtn.addEventListener('click', async () => {
    if (!isPlaying) {
        startPlayback();
    } else {
        stopPlayback();
    }
});

fileInput.addEventListener('change', async () => {
    const file = fileInput.files![0];
    if (!file) {
        loadBtn.disabled = false;
        return;
    }

    try {
        const startTime = performance.now();
        messageLabel.textContent = `Decoding ${file.name}...`;
        const arrayBuffer = await file.arrayBuffer();
        // Use decoding with global AudioContext to resample data into target sample rate for playback.
        const audioBuffer = await getAudioContext().decodeAudioData(arrayBuffer);
        sourceData = audioBuffer.getChannelData(0);
        sourceSampleRate = audioBuffer.sampleRate;
        console.log(`Loaded ${file.name} in ${performance.now() - startTime}ms: ${audioBuffer.numberOfChannels}`
            + ` channels, ${audioBuffer.length} samples at ${audioBuffer.sampleRate}Hz`);
    } catch (error) {
        console.error('Error loading audio file:', error);
        alert(`Error loading audio file: ${(error as Error).message}`);
        messageLabel.textContent = '';
        loadBtn.disabled = false;
        return;
    }

    messageLabel.textContent = '';
    playBtn.disabled = false;
    loadBtn.disabled = false;
    saveBtn.disabled = false;
    stopPlayback();
});

loadBtn.addEventListener('click', async () => {
    loadBtn.disabled = true;
    fileInput.click();
});

// This callback is set when the dialog is open.
let saveDialogResolve: ((filename: string | null) => void) | null;

saveOkBtn.addEventListener('click', () => {
    const filename = saveInput.value.trim();
    saveDialogOverlay.style.display = 'none';
    if (!saveDialogResolve) {
        console.log('Save button is clicked even though the resolve is not set');
        return;
    }
    saveDialogResolve(filename);
});

saveCancelBtn.addEventListener('click', () => {
    saveDialogOverlay.style.display = 'none';
    if (!saveDialogResolve) {
        console.log('Save button is clicked even though the resolve is not set');
        return;
    }
    saveDialogResolve(null);
});

saveInput.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter') {
        saveOkBtn.click();
    } else if (e.key === 'Escape') {
        saveCancelBtn.click();
    }
});

// Unlike the file upload, the browsers are split on saving files: Chrome supports file system access API, while Safari
// and Firefox do not and need workarounds with clicking on <a> elements. We want to support both ways of saving files.
async function showSaveDialog(): Promise<[string | null, FileSystemFileHandle | null]> {
    if ('showSaveFilePicker' in window) {
        try {
            const fileHandle = await window.showSaveFilePicker({
                suggestedName: 'scaled.mp3',
                types: [
                    {
                        description: 'Audio Files',
                        accept: {
                            'audio/mpeg': ['.mp3'],
                            'audio/wav': ['.wav']
                        }
                    }
                ]
            });
            return [fileHandle.name, fileHandle];
        } catch (error) {
            if ((error as Error).name === 'AbortError') {
                return [null, null];
            }
            throw error;
        }
    } else {
        saveDialogOverlay.style.display = 'flex';
        saveInput.focus();
        saveInput.select();
        return new Promise((resolve) => {
            saveDialogResolve = (filename) => resolve([filename, null]);
        });
    }
}

async function saveFile(filename: string, fileHandle: FileSystemFileHandle | null, blob: Blob) {
    if (fileHandle) {
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        console.log(`Saved ${filename} successfully`);
    } else {
        const url = URL.createObjectURL(blob);
        const element = getById<HTMLAnchorElement>('save-dialog-link');
        element.href = url;
        element.download = filename;
        element.click();
        URL.revokeObjectURL(url);
        console.log(`Downloaded ${filename} successfully`);
    }
}

saveBtn.addEventListener('click', async () => {
    if (!sourceData) {
        console.log('No audio data to save');
        return;
    }

    try {
        saveBtn.disabled = true;
        const [filename, fileHandle] = await showSaveDialog();
        if (!filename) {
            saveBtn.disabled = false;
            return;
        }

        const fileType = filename.split('.').pop()?.toLowerCase();
        if (!fileType || (fileType !== 'mp3' && fileType !== 'wav')) {
            alert('Unsupported file format. Please use .mp3 or .wav');
            saveBtn.disabled = false;
            return;
        }

        console.log(`Encoding audio to ${fileType} with sample rate ${sourceSampleRate}Hz`);
        messageLabel.textContent = `Encoding audio to ${fileType}...`;
        const blob = await encodeToBlob(fileType, sourceData, sourceSampleRate);
        messageLabel.textContent = `Saving ${filename}...`;
        await saveFile(filename, fileHandle, blob);
    } catch (error) {
        console.error('Error saving audio file:', error);
        alert(`Error saving audio file: ${(error as Error).message}`);
    }
    messageLabel.textContent = '';
    saveBtn.disabled = false;
})

pitchModeRadio.addEventListener('change', () => {
    if (pitchModeRadio.checked) {
        settings.processingMode = 'pitch';
        saveSettings();
    }
});

timeModeRadio.addEventListener('change', () => {
    if (timeModeRadio.checked) {
        settings.processingMode = 'time';
        saveSettings();
    }
});

pitchSlider.addEventListener('input', () => {
    pitchLabel.textContent = pitchSlider.value + 'x';
    settings.pitchValue = parseFloat(pitchSlider.value);
    saveSettings();
});
