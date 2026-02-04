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
    callback: (...args: T) => void | Promise<void>,
    delay: number,
): (...args: T) => Promise<void> {
    // If curTimer is not undefined, the function is scheduled to be called.
    let curTimer: ReturnType<typeof setTimeout> | undefined;
    // We store the list of all resolves/rejects.
    let curResolves: ((value: void | PromiseLike<void>) => void)[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let curRejects: ((reason?: any) => void)[] = [];

    return (...args: T): Promise<void> => {
        const promise = new Promise<void>((resolve, reject) => {
            curResolves.push(resolve);
            curRejects.push(reject);
        });
        if (curTimer) {
            clearTimeout(curTimer);
        }

        curTimer = setTimeout(async () => {
            try {
                const result = callback(...args);
                if (result instanceof Promise) {
                    await result;
                }
                for (const resolve of curResolves) {
                    resolve();
                }
            } catch (error) {
                for (const reject of curRejects) {
                    reject(error);
                }
            } finally {
                curTimer = undefined;
                curResolves = [];
                curRejects = [];
            }
        }, delay);

        return promise;
    }
}

// Wraps an async event handler to disable a button while the handler is executing.
export function withButtonDisabled<T extends Event>(
    handler: (event: T) => Promise<void> | void,
    button?: HTMLButtonElement,
): (event: T) => Promise<void> {
    return async (event: T) => {
        if (!button) {
            button = (event.target as HTMLButtonElement);
        }
        const wasDisabled = button.disabled;
        button.disabled = true;
        try {
            await handler(event);
        } finally {
            // Only re-enable if it wasn't already disabled before
            if (!wasDisabled) {
                button.disabled = false;
            }
        }
    };
}
