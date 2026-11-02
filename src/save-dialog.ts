import { getById } from './common-utils';

const saveDialog = getById<HTMLDialogElement>('save-dialog');
const saveInput = getById<HTMLInputElement>('save-filename-input');

// This callback is set when the dialog is open.
let saveDialogResolve: ((filename: string | null) => void) | null;

saveDialog.addEventListener('close', () => {
    if (!saveDialogResolve) {
        throw new Error('Save dialog is closed even though the resolve is not set');
    }
    if (saveDialog.returnValue === 'save') {
        const filename = saveInput.value.trim();
        saveDialogResolve(filename);
    } else {
        saveDialogResolve(null);
    }
    saveDialogResolve = null;
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
                            'audio/wav': ['.wav'],
                        },
                    },
                ],
            });
            return [fileHandle.name, fileHandle];
        } catch (error) {
            if ((error as Error).name === 'AbortError') {
                return [null, null];
            }
            throw error;
        }
    } else {
        saveDialog.returnValue = '';
        saveDialog.showModal();
        saveInput.focus();
        saveInput.select();
        const promise = new Promise<[string | null, FileSystemFileHandle | null]>((res) => {
            saveDialogResolve = (filename) => res([filename, null]);
        });
        return promise;
    }
}

// Save blob to file using FileSystem API or fallback download
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
