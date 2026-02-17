// Get HTML element by ID with type safety, throws if element not found
export function getById<T extends HTMLElement>(elementId: string): T {
    const element = document.getElementById(elementId);
    if (!element) {
        throw new Error(`Element with id "${elementId}" not found`);
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
    let curResolves: ((value: void | PromiseLike<void>) => void)[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let curRejects: ((reason?: any) => void)[] = [];

    return (...args: T): Promise<void> => {
        const { promise, resolve, reject } = Promise.withResolvers<void>();
        curResolves.push(resolve);
        curRejects.push(reject);
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
    }
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
        };
        try {
            await handler(event);
        } finally {
            for (let i = 0; i < buttons.length; i++) {
                buttons[i].disabled = wasDisabled[i];
            }
        }
    };
}

// Convert number of seconds into a string representation.
export function secondsToString(sec: number): string {
    const rounded = Math.max(0, Math.round(sec));
    const minutes = Math.floor(rounded / 60);
    const seconds = rounded % 60;
    if (minutes > 0) {
        return `${minutes}m${seconds}s`;
    } else {
        return `${seconds}s`;
    }
}

// Return a promise which resolves in delay milliseconds.
export async function sleep(delay: number): Promise<void> {
    const { promise, resolve } = Promise.withResolvers<void>();
    setTimeout(() => resolve(), delay);
    return promise;
} 
