import initWasmModule, { get_settings } from '../wasm/build/wasm_main_module';
import { AudioProcessor } from './audio-processor';
import { decodeAudioFromBlob } from './media-decoder';
import { encodeAudioToBlob } from './media-encoder';
import { Player } from './player';
import { Recorder } from './recorder';
import { saveFile, showSaveDialog } from './save-dialog';
import { Spectrogram } from './spectrogram';
import { drainRingBuffer, Float32RingBuffer } from './sync';
import { getAudioLength, getAudioSeconds, type InterleavedAudio, type ProcessingMode } from './types';
import { debounce, getById, secondsToString, sleep, withButtonsDisabled } from './utils';
import { runBenchmark, type BenchmarkResults } from './benchmark';

const MAX_PITCH_VALUE = 2.0;
const MIN_PITCH_VALUE = 0.5;
const PITCH_VALUE_STEP = 0.125;
const DEFAULT_PITCH_VALUE = 1.25;
const DEFAULT_PROCESSING_MODE = 'pitch';

const SAVE_SETTINGS_DEBOUNCE = 500;

const SETTINGS_KEY = 'pitch-changer-settings';

window.onerror = (event: Event | string, source?: string, lineno?: number, colno?: number, error?: Error) => {
    console.error('Error:', event, source, lineno, colno, error);
    alert(event);
};

window.onunhandledrejection = (event: PromiseRejectionEvent) => {
    console.error('Unhandled rejection:', event);
    alert(event.reason);
};

if (!window.crossOriginIsolated) {
    throw new Error('This app works only in cross-origin isolated environments, configure the web server');
}

// Settings stored locally
interface AppSettings {
    processingMode: ProcessingMode;
    pitchValue: number;
}

//
// Global app state
//

class AppState {
    // Audio data
    sourceAudio: InterleavedAudio | null = null;
    processor: AudioProcessor = new AudioProcessor();
    spectrogram: Spectrogram | null = null;
    benchmarkTimes: BenchmarkResults | null = null;

    // Audio player and recorder are lazy initialized because they affect the browser (and OS) UI
    private audioContext: AudioContext | null = null;
    private recorder: Promise<Recorder> | null = null;
    private player: Promise<Player> | null = null;

    // Settings
    settings: AppSettings;

    constructor() {
        this.settings = this.loadSettings();
    }

    async initProcessor(): Promise<void> {
        await this.processor.init();
        this.processor.setParams(this.settings.processingMode, this.settings.pitchValue);
    }

    private loadSettings(): AppSettings {
        const settings: AppSettings = {
            processingMode: DEFAULT_PROCESSING_MODE,
            pitchValue: DEFAULT_PITCH_VALUE
        };
        Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}'));
        if (!settings.processingMode) {
            settings.processingMode = DEFAULT_PROCESSING_MODE;
        }
        return settings;
    }

    updateSettings() {
        this.saveSettings();
        this.processor.setParams(this.settings.processingMode, this.settings.pitchValue);
    }

    private saveSettings = debounce(SAVE_SETTINGS_DEBOUNCE, () => {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings));
    });

    getAudioContext(): AudioContext {
        if (!this.audioContext) {
            this.audioContext = new AudioContext({ latencyHint: 'interactive' });
            console.log(`AudioContext created, sample rate ${this.audioContext.sampleRate}Hz`);
        }
        return this.audioContext!;
    }

    async getRecorder(): Promise<Recorder> {
        if (!this.recorder) {
            this.recorder = Recorder.create(this.getAudioContext());
        }
        return this.recorder!;
    }

    async getPlayer(): Promise<Player> {
        if (!this.player) {
            this.player = Player.create(this.getAudioContext());
        }
        return this.player;
    }
}

await initWasmModule();
const appState = new AppState();
await appState.initProcessor();

//
// UI elements
//

const contentContainer = getById<HTMLDivElement>('content-container');
const recordBtn = getById<HTMLButtonElement>('record-btn');
const recordBtnEmoji = getById<HTMLButtonElement>('record-btn-emoji');
const recordingBtnClass = 'fa-microphone';
const stopBtnClass = 'fa-stop';
const playBtn = getById<HTMLButtonElement>('play-btn');
const playBtnEmoji = getById<HTMLButtonElement>('play-btn-emoji');
const playingBtnClass = 'fa-play';
const loadBtn = getById<HTMLButtonElement>('load-btn');
const saveBtn = getById<HTMLButtonElement>('save-btn');
const pitchSlider = getById<HTMLInputElement>('pitch-slider');
const pitchLabel = getById<HTMLElement>('pitch-label');
const pitchModeRadio = getById<HTMLInputElement>('pitch-mode');
const timeModeRadio = getById<HTMLInputElement>('time-mode');
const formantPitchModeRadio = getById<HTMLInputElement>('formant-pitch-mode');
const sourceLabel = getById<HTMLElement>('source-label');
const processingSpinner = getById<HTMLDivElement>('processing-spinner');
const processingLabel = getById<HTMLDivElement>('processing-label');
const fileInput = getById<HTMLInputElement>('file-input');
const spectrogramCanvas = getById<HTMLCanvasElement>('spectrogram-canvas');

// Debug UI elements
const debugToggleBtn = getById<HTMLButtonElement>('debug-toggle-btn');
const debugPanel = getById<HTMLDivElement>('debug-panel');
const debugInfo = getById<HTMLSpanElement>('debug-info');
const copyDebugBtn = getById<HTMLButtonElement>('copy-debug-btn');
const benchmarkBtn = getById<HTMLButtonElement>('benchmark-btn');

// Reset all buttons to default state
async function onBeforeSourceAudioDataSet() {
    recordBtn.disabled = true;
    loadBtn.disabled = true;
    playBtn.disabled = true;
    playBtnEmoji.classList.remove(stopBtnClass);
    playBtnEmoji.classList.add(playingBtnClass);
    playBtn.title = 'Play';
    playBtn.classList.remove('playing');
    saveBtn.disabled = true;
    const player = await appState.getPlayer();
    player.stop();
}

// Re-enable play and save buttons
function onAfterSourceAudioDataSet() {
    recordBtn.disabled = false;
    loadBtn.disabled = false;
    playBtn.disabled = false;
    saveBtn.disabled = false;
}

// Initialize UI from stored settings
function applySettingsToUI(settings: AppSettings): void {
    pitchModeRadio.checked = settings.processingMode === 'pitch';
    timeModeRadio.checked = settings.processingMode === 'time';
    formantPitchModeRadio.checked = settings.processingMode === 'formant-preserving-pitch';
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
}

applySettingsToUI(appState.settings);
appState.spectrogram = new Spectrogram(spectrogramCanvas);

//
// Event handlers
//

async function handleRecordClick(): Promise<void> {
    const recorder = await appState.getRecorder();
    if (!recorder.isRecording) {
        await onBeforeSourceAudioDataSet();

        sourceLabel.textContent = 'Recording...';
        recordBtn.disabled = false;
        recordBtn.title = 'Stop';
        recordBtn.classList.add('recording');
        recordBtnEmoji.classList.remove(recordingBtnClass);
        recordBtnEmoji.classList.add(stopBtnClass);

        appState.sourceAudio = await recorder.record();

        sourceLabel.textContent = `Recorded ${secondsToString(getAudioSeconds(appState.sourceAudio))}`;
        recordBtn.title = 'Record';
        recordBtn.classList.remove('recording');
        recordBtnEmoji.classList.remove(stopBtnClass);
        recordBtnEmoji.classList.add(recordingBtnClass);

        onAfterSourceAudioDataSet();
    } else {
        recorder.stop();
    }
}

async function runPlay(player: Player): Promise<void> {
    const SPECTROGRAM_INTERVAL = 30; // ms

    const sampleRate = appState.sourceAudio!.sampleRate;
    const numChannels = appState.sourceAudio!.numChannels;
    // Use smaller ring buffer of 0.5s. This means we react faster to changes in ratio or processing mode.
    const bufferSize = sampleRate * numChannels / 2;

    console.log(`Start playing audio with buffer size ${bufferSize}`);
    const buffer = Float32RingBuffer.withCapacity(bufferSize);

    // Synchronization for playing is rather complicated:
    //   1. Start the processor
    //   2. Wait until the buffer is at least partially filled to prevent underruns
    //   3. Only after that start playing audio
    //   4. If all source data is processed, AudioProcessor closes the buffer
    //   5. If a user clicks the stop button, Player closes the buffer
    //   6. The Player promise completes once the AudioWorklet signals it has processed all data from the ring buffer
    const processPromise = appState.processor.process(appState.sourceAudio!, buffer);
    await buffer.waitPopAsync(bufferSize / 2);
    const playerPromise = player.play(buffer, numChannels);

    const spectrogramTimer = setInterval(() => {
        const latestData = player.getLatestSamples(appState.spectrogram!.getSamples(), numChannels);
        appState.spectrogram!.draw(latestData, numChannels, sampleRate);
    }, SPECTROGRAM_INTERVAL);

    // We do not need to wait for this promise, but let's do it for symmetry.
    await processPromise;
    await playerPromise;

    clearInterval(spectrogramTimer);
    appState.spectrogram!.clear();

    console.log('Stopped playing audio');
}

async function handlePlayClick(): Promise<void> {
    const player = await appState.getPlayer();
    if (!player.isPlaying) {
        if (!appState.sourceAudio) {
            console.error('Error: no audio data to play');
            return;
        }

        playBtnEmoji.classList.remove(playingBtnClass);
        playBtnEmoji.classList.add(stopBtnClass);
        playBtn.title = 'Pause';
        playBtn.classList.add('playing');

        await runPlay(player);

        playBtn.title = 'Play';
        playBtn.classList.remove('playing');
        playBtnEmoji.classList.remove(stopBtnClass);
        playBtnEmoji.classList.add(playingBtnClass);
    } else {
        player.stop();
    }
}


function handleLoadClick(): void {
    // Oh https://developer.mozilla.org/en-US/docs/Web/API/Window/showOpenFilePicker, where are you
    fileInput.click();
}

async function handleFileInputChange(file: File): Promise<void> {
    await onBeforeSourceAudioDataSet();
    const audioContext = appState.getAudioContext();
    try {
        const startTime = performance.now();
        processingSpinner.style.display = 'inline-block';
        processingLabel.textContent = `Decoding ${file.name}...`;
        appState.sourceAudio = await decodeAudioFromBlob(file, audioContext);

        console.log(`Loaded ${file.name} in ${performance.now() - startTime}ms: ${getAudioLength(appState.sourceAudio)} samples per channel, ${appState.sourceAudio.numChannels} channels at ${appState.sourceAudio.sampleRate}Hz`);
        sourceLabel.textContent = `Loaded ${secondsToString(getAudioSeconds(appState.sourceAudio))} of ${file.name}`;
        onAfterSourceAudioDataSet();
    } catch (error) {
        sourceLabel.textContent = `Error loading audio file: ${String(error)}`;
        // In case of error, play and save buttons should stay disabled.
        recordBtn.disabled = false;
        loadBtn.disabled = false;
    } finally {
        processingSpinner.style.display = 'none';
        processingLabel.textContent = '';
    }
}

async function processAllAudio(): Promise<InterleavedAudio> {
    // This is a background task, use larger buffers.
    const BUFFER_SIZE = 1024 * 1024;
    const startTime = performance.now();

    const output = Float32RingBuffer.withCapacity(BUFFER_SIZE);
    const processPromise = appState.processor.process(appState.sourceAudio!, output);
    const drainPromise = drainRingBuffer(output);

    await processPromise;
    //  processor.process() closes the output and forces drainRingBuffer completion.
    const data = await drainPromise;

    const deltaTime = performance.now() - startTime;
    console.log(`Process all audio, got ${data.length} samples in ${deltaTime}ms`);
    return {
        data,
        sampleRate: appState.sourceAudio!.sampleRate,
        numChannels: appState.sourceAudio!.numChannels
    };
}

async function handleSaveClick(): Promise<void> {
    const [filename, fileHandle] = await showSaveDialog();
    if (!filename) {
        return;
    }

    const fileType = filename.split('.').pop()?.toLowerCase();
    if (!fileType || (fileType !== 'mp3' && fileType !== 'ogg' && fileType !== 'wav')) {
        alert('Unsupported file format. Please use .mp3, .ogg or .wav');
        return;
    }

    try {
        processingSpinner.style.display = 'inline-block';
        processingLabel.textContent = 'Processing...';
        const processedAudio = await processAllAudio();
        console.log(`Encoding audio to ${fileType} with sample rate ${processedAudio.sampleRate}Hz`);
        processingLabel.textContent = 'Encoding...';
        const blob = await encodeAudioToBlob(fileType, processedAudio);
        await saveFile(filename, fileHandle, blob);
    } finally {
        processingSpinner.style.display = 'none';
        processingLabel.textContent = `Saved ${filename}`;
    }
}

function handlePitchModeChange(): void {
    if (pitchModeRadio.checked) {
        appState.settings.processingMode = 'pitch';
        appState.updateSettings();
    }
}

function handleTimeModeChange(): void {
    if (timeModeRadio.checked) {
        appState.settings.processingMode = 'time';
        appState.updateSettings();
    }
}

function handleFormantModeChange(): void {
    if (formantPitchModeRadio.checked) {
        appState.settings.processingMode = 'formant-preserving-pitch';
        appState.updateSettings();
    }
}

function handlePitchSliderInput(): void {
    pitchLabel.textContent = pitchSlider.value + 'x';
    appState.settings.pitchValue = parseFloat(pitchSlider.value);
    appState.updateSettings();
}

async function handleDebugPanelClick() {
    if (debugPanel.style.display === 'none') {
        let info = `User-agent: ${navigator.userAgent}\n\ndevicePixelRatio: ${window.devicePixelRatio}`;
        try {
            const wasm = await initWasmModule();
            info += `\n\nWASM settings: ${get_settings()}\n\nWASM memory: ${wasm.memory.buffer.byteLength}`;
        } catch (error) {
            info += `\n\nWASM error: ${error}`;
        }
        // Add benchmark results if available
        if (appState.benchmarkTimes) {
            info += `\n\nBenchmark (processing-to-realtime):`;
            const { 'time': timeStretch, pitch, 'formant-preserving-pitch': formantPitch } = appState.benchmarkTimes;
            info += `\n  time-stretch: ${timeStretch.toFixed(2)}`;
            info += `\n  pitch: ${pitch.toFixed(2)}`;
            info += `\n  formant-preserving-pitch: ${formantPitch.toFixed(2)}`;
        }
        debugInfo.textContent = info;
        debugPanel.style.display = 'flex';
    } else {
        debugPanel.style.display = 'none';
    }
}

async function handleBenchmarkClick() {
    const sampleRate = appState.getAudioContext().sampleRate;
    const numChannels = 2;
    const durationSeconds = 30;
    const pitchValue = 1.25;

    const origText = benchmarkBtn.textContent;
    benchmarkBtn.disabled = true;
    benchmarkBtn.textContent = origText + " (running...)";
    // This is a hacky way to force button change the content/disable status while still blocking the main thread.
    await sleep(0);
    try {
        const results = runBenchmark(sampleRate, numChannels, durationSeconds, pitchValue);
        appState.benchmarkTimes = results;
        // Refresh the debug panel in the simplest way =)
        handleDebugPanelClick();
        handleDebugPanelClick();
        console.log('Benchmark completed:', results);
    } catch (error) {
        console.error('Benchmark failed:', error);
        alert(`Benchmark failed: ${error}`);
    } finally {
        benchmarkBtn.disabled = false;
        benchmarkBtn.textContent = origText;
    }
}

async function handleCopyDebugClick() {
    const text = debugInfo.textContent;
    try {
        await navigator.clipboard.writeText(text);
    } catch (error) {
        console.error('Failed to copy debug info:', error);
        alert('Failed to copy to clipboard: ' + error);
    }
}

recordBtn.addEventListener('click', () => handleRecordClick());
playBtn.addEventListener('click', () => handlePlayClick());
loadBtn.addEventListener('click', () => handleLoadClick());
saveBtn.addEventListener('click', withButtonsDisabled([playBtn, saveBtn], () => handleSaveClick()));
pitchModeRadio.addEventListener('change', () => handlePitchModeChange());
timeModeRadio.addEventListener('change', () => handleTimeModeChange());
formantPitchModeRadio.addEventListener('change', () => handleFormantModeChange());
pitchSlider.addEventListener('input', () => handlePitchSliderInput());
fileInput.addEventListener('change', withButtonsDisabled([loadBtn], async () => {
    const file = fileInput.files![0];
    if (!file) {
        return;
    }
    await handleFileInputChange(file);
}));
debugToggleBtn.addEventListener('click', () => handleDebugPanelClick());
copyDebugBtn.addEventListener('click', () => handleCopyDebugClick());
benchmarkBtn.addEventListener('click', () => handleBenchmarkClick());
debugInfo.addEventListener('click', () => handleDebugPanelClick());
