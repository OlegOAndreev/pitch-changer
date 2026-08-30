import { AudioProcessorManager } from '../../src/audio-processor.js';
import type { WorkerIframeInit } from './common.js';

async function init() {
    const iframeUrl = new URL(location.href);
    const workerUrl = iframeUrl.searchParams.get('worker_url');
    if (!workerUrl) {
        console.error(`worker_url not defined in ${location.href}`);
        return;
    }
    const manager = await AudioProcessorManager.create(workerUrl, (e) => console.error(e));
    const clientPort = manager.createClientPort();

    window.parent.postMessage(
        {
            type: 'pitch-changer-extension-worker-iframe-init',
            audioProcessorClientPort: clientPort,
        } as WorkerIframeInit,
        '*',
        [clientPort],
    );
}

init();
