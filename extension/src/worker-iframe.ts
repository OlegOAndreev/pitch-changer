import { AudioProcessorManager } from '../../src/audio-processor.js';
import { anyToString } from '../../src/common-utils.js';
import { type WorkerIframeInit } from './common.js';

async function init() {
    let iframeUrl: URL;
    try {
        iframeUrl = new URL(location.href);
    } catch (error) {
        throw new Error(`Failed to parse the worker iframe URL ${location.href}: ${anyToString(error)}`, {
            cause: error,
        });
    }
    const workerUrl = iframeUrl.searchParams.get('worker_url');
    if (!workerUrl) {
        throw new Error(`worker_url not defined in ${location.href}`);
    }

    let manager: AudioProcessorManager;
    try {
        manager = await AudioProcessorManager.create(workerUrl, (e) => console.error(e));
    } catch (error) {
        throw new Error(`Failed to create the audio processor manager for ${workerUrl}: ${anyToString(error)}`, {
            cause: error,
        });
    }

    let clientPort: MessagePort;
    try {
        clientPort = manager.createClientPort();
    } catch (error) {
        throw new Error(`Failed to create the audio processor client port: ${anyToString(error)}`, { cause: error });
    }

    try {
        window.parent.postMessage(
            {
                type: 'pitch-changer-extension-worker-iframe-init',
                audioProcessorClientPort: clientPort,
            } satisfies WorkerIframeInit,
            '*',
            [clientPort],
        );
    } catch (error) {
        throw new Error(`Failed to send the client port to the parent frame: ${anyToString(error)}`, {
            cause: error,
        });
    }
}

init().catch((error: unknown) => {
    console.error('Audio Pitch Changer: worker iframe initialization failed', error);
});
