import { Recorder } from './recorder';

import initRustModule from '../wasm/build/wasm_main_module';
import { encodeToBlob } from './media-encoder';
import { Player } from './player';
import { saveFile, showSaveDialog } from './save-dialog';
import { getById } from './utils';
import { shiftPitch, timeStretch } from './process-audio';

await initRustModule();

const MAX_PITCH_VALUE = 2.0;
const MIN_PITCH_VALUE = 0.5;
const PITCH_VALUE_STEP = 0.125;
const DEFAULT_PITCH_VALUE = 1.25;
const DEFAULT_PROCESSING_MODE = 'pitch';

// Settings interface and implementation
interface AppSettings {
    processingMode: 'pitch' | 'time';
    pitchValue: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onError(message: string, error: any) {
    console.error(message, error);
    if (error instanceof Error) {
        alert(`${message}: ${error.message}`);
    } else {
        alert(message);
    }
}

// Lazily initialize audio context to prevent warning logs in Firefox.
let globalAudioContext: AudioContext;
function getAudioContext(): AudioContext {
    if (!globalAudioContext) {
        try {
            globalAudioContext = new AudioContext();
            console.log(`AudioContext created, sample rate ${globalAudioContext.sampleRate}Hz`);
        } catch (error) {
            onError('Error creating AudioContext', error);
        }
    }
    return globalAudioContext;
}

let globalRecorder: Promise<Recorder>;
async function getRecorder(): Promise<Recorder> {
    if (!globalRecorder) {
        try {
            globalRecorder = Recorder.create(getAudioContext());
        } catch (error) {
            onError('Error creating Recorder', error);
        }
    }
    return globalRecorder;
}

let globalPlayer: Player;
function getPlayer(): Player {
    if (!globalPlayer) {
        globalPlayer = new Player(getAudioContext());
    }
    return globalPlayer;
}

let sourceData: Float32Array;
let sourceSampleRate: number;
let processedData: Float32Array;


const contentContainer = getById<HTMLDivElement>('content-container');
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

const SETTINGS_KEY = 'pitch-changer-settings';
const settings = loadSettings();

function loadSettings(): AppSettings {
    const settings: AppSettings = {
        processingMode: DEFAULT_PROCESSING_MODE,
        pitchValue: DEFAULT_PITCH_VALUE
    };
    Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}'));
    if (settings.processingMode !== 'time') {
        settings.processingMode = DEFAULT_PROCESSING_MODE;
    }
    pitchModeRadio.checked = settings.processingMode === 'pitch';
    timeModeRadio.checked = settings.processingMode === 'time';
    if (settings.pitchValue > MAX_PITCH_VALUE || settings.pitchValue < MIN_PITCH_VALUE) {
        settings.pitchValue = DEFAULT_PITCH_VALUE;
    }
    pitchSlider.min = MIN_PITCH_VALUE.toString();
    pitchSlider.max = MAX_PITCH_VALUE.toString();
    pitchSlider.step = PITCH_VALUE_STEP.toString();
    pitchSlider.value = settings.pitchValue.toString();
    pitchLabel.textContent = settings.pitchValue + 'x';
    const pitchSliderMarkers = getById<HTMLDataListElement>('pitch-slider-markers');
    for (let v = MIN_PITCH_VALUE; v <= MAX_PITCH_VALUE; v += PITCH_VALUE_STEP) {
        const node = new Option();
        node.value = v.toString();
        pitchSliderMarkers.appendChild(node);
    }

    // Show the content container after settings are applied
    contentContainer.style.visibility = 'visible';

    return settings;
}

let saveSettingsTimer: number | null = null;
function saveSettings() {
    // Debounce saving settings.
    if (saveSettingsTimer) {
        clearTimeout(saveSettingsTimer);
    }
    saveSettingsTimer = setTimeout(() => {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    }, 500);
}

function processSourceAudio() {
    if (!sourceData) {
        console.error('processAudio called with null source data');
        return;
    }
    try {
        switch (settings.processingMode) {
            case 'pitch':
                processedData = shiftPitch(sourceData, sourceSampleRate, settings.pitchValue);
                break;
            case 'time':
                processedData = timeStretch(sourceData, sourceSampleRate, settings.pitchValue);
                break;
        }
        console.log(`Processed audio: got ${processedData.length} samples`);
    } catch (error) {
        console.error('Error processing audio:', error);
        alert(`Error processing audio: ${(error as Error).message}`);
    }
}

function onStopPlayback() {
    console.log('Stopped playing audio');
    playBtnEmoji.textContent = '▶';
    playBtn.title = 'Play';
    playBtn.classList.remove('playing');
}

recordBtn.addEventListener('click', async () => {
    const recorder = await getRecorder();
    if (!recorder.isRecording) {
        getPlayer().stop();
        try {
            await recorder.start();
        } catch (error) {
            console.error('Could not start recording:', error);
            alert(`Could not start recording: ${(error as Error).message}`);
        }

        messageLabel.textContent = 'Recording...';
        recordBtnEmoji.textContent = '⏹';
        recordBtn.title = 'Stop';
        recordBtn.classList.add('recording');
        playBtn.disabled = true;
        loadBtn.disabled = true;
        saveBtn.disabled = true;
    } else {
        sourceData = await recorder.stop();
        sourceSampleRate = getAudioContext().sampleRate;

        const sourceSeconds = Math.round(sourceData.length / sourceSampleRate);
        messageLabel.textContent = `Recorded ${sourceSeconds}s`;
        processSourceAudio();

        messageLabel.textContent = '';
        recordBtnEmoji.textContent = '⏺';
        recordBtn.title = 'Record';
        recordBtn.classList.remove('recording');
        playBtn.disabled = false;
        loadBtn.disabled = false;
        saveBtn.disabled = false;

    }
});

playBtn.addEventListener('click', async () => {
    const player = getPlayer();
    if (!player.isPlaying) {
        if (!sourceData) {
            console.error('Error: no audio data to play');
            return;
        }

        await getPlayer().play(processedData, sourceSampleRate, onStopPlayback);

        playBtnEmoji.textContent = '⏹';
        playBtn.title = 'Pause';
        playBtn.classList.add('playing');
    } else {
        player.stop();
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
        messageLabel.textContent = `Decoded ${file.name}`;
    } catch (error) {
        console.error('Error loading audio file:', error);
        alert(`Error loading audio file: ${(error as Error).message}`);
        messageLabel.textContent = '';
        loadBtn.disabled = false;
        return;
    }

    processSourceAudio();

    playBtn.disabled = false;
    loadBtn.disabled = false;
    saveBtn.disabled = false;
    getPlayer().stop();
});

loadBtn.addEventListener('click', async () => {
    loadBtn.disabled = true;
    fileInput.click();
});

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
        if (!fileType || (fileType !== 'mp3' && fileType != 'ogg' && fileType !== 'wav')) {
            alert('Unsupported file format. Please use .mp3, .ogg or .wav');
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
});

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
