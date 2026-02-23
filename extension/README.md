# Web Extension Template

A simple web extension template written in TypeScript, compatible with both Chrome and Firefox.

## Features

- **Cross-browser compatibility**: Works with Chrome (Manifest V3) and Firefox (with browser_specific_settings)
- **TypeScript**: Full TypeScript support with type definitions for Chrome API
- **Content Script**: Runs on every website (`<all_urls>`)
- **Background Script**: Service worker for extension logic
- **Popup UI**: Interactive popup with toggle controls and actions
- **Storage API**: Uses `chrome.storage.local` for persistent settings

## Project Structure

```
extension/
├── src/
│   ├── content.ts          # Content script (runs on web pages)
│   ├── background.ts       # Background service worker
│   ├── popup.ts           # Popup script
│   └── popup.html         # Popup HTML
├── dist/                  # Compiled JavaScript (generated)
├── icons/                 # Extension icons (placeholder)
├── manifest.json          # Extension manifest
├── package.json          # NPM dependencies and scripts
├── tsconfig.json         # TypeScript configuration
└── README.md             # This file
```

## Getting Started

### 1. Install Dependencies

```bash
cd extension
npm install
```

### 2. Build the Extension

```bashч]ч]
npm run build
```

This compiles TypeScript files from `src/` to `dist/`.

### 3. Load in Browser

#### Chrome

1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension` directory

#### Firefox

1. Open `about:debugging`
2. Click "This Firefox"
3. Click "Load Temporary Add-on"
4. Select `extension/manifest.json`

## Development

- `npm run build` - Build once
- `npm run build:watch` - Watch for changes and rebuild
- `npm run package` - Create a ZIP file for distribution

## Manifest Features

- **Manifest V3**: Uses service workers for background scripts
- **Permissions**: `activeTab`, `scripting`, `<all_urls>`
- **Content Scripts**: Injected into all websites
- **Icons**: Placeholder icons included (create your own in `icons/`)

## Content Script

The content script (`content.ts`) runs on every website and:

- Adds a temporary indicator when active
- Responds to messages from background/popup
- Can highlight page elements
- Logs page information to console

## Customization

1. Update `manifest.json` with your extension details
2. Replace placeholder icons in `icons/` directory
3. Modify content script logic in `src/content.ts`
4. Update popup UI in `src/popup.html` and `src/popup.ts`

## Browser Compatibility Notes

- **Chrome**: Uses Manifest V3 with service workers
- **Firefox**: Requires `browser_specific_settings` in manifest
- **Safari**: Would require separate Safari Web Extension conversion

## License

MIT
