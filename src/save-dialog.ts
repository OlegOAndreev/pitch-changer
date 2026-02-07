import { getById } from "./utils";

const saveDialogOverlay = getById<HTMLDivElement>('save-dialog-overlay');
const saveInput = getById<HTMLInputElement>('save-filename-input');
const saveOkBtn = getById<HTMLButtonElement>('save-dialog-ok-btn');
const saveCancelBtn = getById<HTMLButtonElement>('save-dialog-cancel-btn');

// This callback is set when the dialog is open.
let saveDialogResolve: ((filename: string | null) => void) | null;

saveOkBtn.addEventListener('click', () => {
    const filename = saveInput.value.trim();
    saveDialogOverlay.style.display = 'none';
    if (!saveDialogResolve) {
        console.error('Save button is clicked even though the resolve is not set');
        return;
    }
    saveDialogResolve(filename);
    saveDialogResolve = null;
});

saveCancelBtn.addEventListener('click', () => {
    saveDialogOverlay.style.display = 'none';
    if (!saveDialogResolve) {
        console.error('Save button is clicked even though the resolve is not set');
        return;
    }
    saveDialogResolve(null);
    saveDialogResolve = null;
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
export async function showSaveDialog(): Promise<[string | null, FileSystemFileHandle | null]> {
    if ('showSaveFilePicker' in window) {
        try {
            const fileHandle = await window.showSaveFilePicker({
                suggestedName: 'scaled.mp3',
                types: [
                    {
                        description: 'Audio Files',
                        accept: {
                            'audio/mpeg': ['.mp3'],
                            'audio/ogg': ['.ogg'],
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
        const { promise, resolve } = Promise.withResolvers<[string | null, FileSystemFileHandle | null]>();
        saveDialogResolve = (filename) => resolve([filename, null]);
        return promise;
    }
}

export async function saveFile(filename: string, fileHandle: FileSystemFileHandle | null, blob: Blob) {
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
