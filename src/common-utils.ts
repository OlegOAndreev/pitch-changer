// Returns "base name" from file path or URL
export function getBaseName(name: string): string {
    if (name.endsWith('/')) {
        name = name.substring(0, name.length - 1);
    }
    // This works for -1 as well
    const start = name.lastIndexOf('/') + 1;
    const end = name.substring(start).lastIndexOf('.') + start;
    if (end === start - 1 || end === start) {
        return name.substring(start);
    }
    return name.substring(start, end);
}

// Get HTML element by ID with type safety, throws if element not found
export function getById<T extends HTMLElement>(elementId: string): T {
    const element = document.getElementById(elementId);
    if (!element) {
        throw new Error(`Element with id '${elementId}' not found`);
    }
    return element as T;
}

// Converts a function into a function which is run after a fixed timeout.
//
// Source code taken from https://tech.reverse.hr/articles/debounce-function-in-typescript
//
// Modified to support both sync and async functions.
export function debounce<T extends unknown[]>(
    delay: number,
    callback: (...args: T) => void | Promise<void>,
): (...args: T) => Promise<void> {
    // If curTimer is not undefined, the function is scheduled to be called.
    let curTimer: ReturnType<typeof setTimeout> | undefined;
    // We store the list of all resolves/rejects.
    let curResolves: ((value: void) => void)[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let curRejects: ((reason?: any) => void)[] = [];

    return (...args: T): Promise<void> => {
        let resolve: (value: void) => void;
        let reject: (reason?: unknown) => void;
        const promise = new Promise<void>((res, rej) => {
            resolve = res;
            reject = rej;
        });
        curResolves.push(resolve!);
        curRejects.push(reject!);
        if (curTimer) {
            clearTimeout(curTimer);
        }

        curTimer = setTimeout(async () => {
            // Copy the resolves/rejects so that they do not get changesd
            const resolves = curResolves;
            const rejects = curRejects;
            curResolves = [];
            curRejects = [];
            curTimer = undefined;
            try {
                const result = callback(...args);
                if (result instanceof Promise) {
                    await result;
                }
                for (const resolve of resolves) {
                    resolve();
                }
            } catch (error) {
                for (const reject of rejects) {
                    reject(error);
                }
            }
        }, delay);

        return promise as Promise<void>;
    };
}

// Wraps an async event handler to disable a button while the handler is executing.
export function withButtonsDisabled<T extends Event>(
    buttons: HTMLButtonElement[],
    handler: (event: T) => Promise<void> | void,
): (event: T) => Promise<void> {
    return async (event: T) => {
        // Only re-enable buttons if they weren't already disabled before
        const wasDisabled = [];
        for (let i = 0; i < buttons.length; i++) {
            wasDisabled.push(buttons[i].disabled);
            buttons[i].disabled = true;
        }
        try {
            await handler(event);
        } finally {
            for (let i = 0; i < buttons.length; i++) {
                buttons[i].disabled = wasDisabled[i];
            }
        }
    };
}

// Return a promise which resolves in delay milliseconds.
export async function sleep(delay: number): Promise<void> {
    let resolve: (value?: void) => void;
    const promise = new Promise<void>((res) => {
        resolve = res;
    });
    setTimeout(() => resolve(), delay);
    return promise;
}

// Setup a few common error handlers (onerror must still be setup in each worker/worklet).
export function setupWindowOnError() {
    window.onerror = (event: Event | string, source?: string, lineno?: number, colno?: number, error?: Error) => {
        console.error('Error:', event, source, lineno, colno, error);
        alert(`Error: ${event}, ${source}:${lineno}, ${error}`);
    };

    window.onunhandledrejection = (event: PromiseRejectionEvent) => {
        console.error('Unhandled rejection:', event);
        alert(`Unhandled rejection: ${event.reason}`);
    };
}

// Log error, suitable for global error catching
export function logError(context: string, event: ErrorEvent) {
    console.error(`Error from ${context}: ${event.message}, ${event.filename}:${event.lineno}, ${event.error}`);
    alert(`Error from ${context}: ${event.message}, ${event.filename}:${event.lineno}, ${event.error}`);
}
