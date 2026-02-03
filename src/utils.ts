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
export function debounce<T extends unknown[]>(
    callback: (...args: T) => void,
    delay: number,
) {
    let timeoutTimer: ReturnType<typeof setTimeout>;

    return (...args: T) => {
        clearTimeout(timeoutTimer);

        timeoutTimer = setTimeout(() => {
            callback(...args);
        }, delay);
    };
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
