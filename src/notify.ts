// A tiny module for showing closeable error messages.

export type NotificationPosition = 'bottom-left' | 'bottom-right';

const CONTAINER_ID = 'notification-container';
const STYLES_ID = 'notification-styles';

// Configure once per app.
const position: NotificationPosition = 'bottom-left';

const STYLES = `
#${CONTAINER_ID} {
    position: fixed;
    bottom: 20px;
    display: grid;
    gap: 8px;
    max-width: min(400px, calc(100vw - 40px));
    max-height: calc(100vh - 40px);
    overflow-y: auto;
    z-index: 10001;
}

#${CONTAINER_ID}[data-position='bottom-left'] {
    left: 20px;
}

#${CONTAINER_ID}[data-position='bottom-right'] {
    right: 20px;
}

.notification {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    padding: 8px 12px;
    background: #cc2222;
    color: white;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    font-size: 14px;
}

.notification-message {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.notification-close {
    margin-left: auto;
    padding: 0;
    border: none;
    background: none;
    color: white;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
}
`;

function getContainer(): HTMLDivElement {
    const existing = document.getElementById(CONTAINER_ID);
    if (existing) {
        return existing as HTMLDivElement;
    }

    if (!document.getElementById(STYLES_ID)) {
        const style = document.createElement('style');
        style.id = STYLES_ID;
        style.textContent = STYLES;
        document.head.append(style);
    }

    const container = document.createElement('div');
    container.id = CONTAINER_ID;
    container.dataset.position = position;
    document.body.append(container);
    return container;
}

// Show an error notification. It stays on screen until the user closes it.
export function showError(message: string): void {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.setAttribute('role', 'alert');

    const messageElement = document.createElement('span');
    messageElement.className = 'notification-message';
    messageElement.textContent = message;

    const closeButton = document.createElement('button');
    closeButton.className = 'notification-close';
    closeButton.type = 'button';
    closeButton.setAttribute('aria-label', 'Close');
    closeButton.textContent = '×';
    closeButton.addEventListener('click', () => notification.remove());

    notification.append(messageElement, closeButton);
    getContainer().append(notification);
}
