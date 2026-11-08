# Audio Pitch Changer Extension

This is an extension for browsers, which allows changing the pitch of audion in web pages. The extension supports not
only the standard media elements, but also web pages that are using WebAudio directly (e.g. Yandex Music).

## Development

### Preparing environment
See `Preparing environment` in the upper-level README for the pitch-changer project

### Building and testing
Full build, output in `dist/`:
```bash
npm run build
```

### Installing

**Chrome**

1. Enable `Developer mode` in browser://extensions/
2. Click `Load Unpacked` and open the extension/dist/ directory.

**Firefox**

For testing:
1. Go to about:addons, click `Debug Add-ons`
2. Click `Load Temporary Add-on...` and open the extension/dist/ directory.

For regular use:
1. Get add-on developer credentials from https://addons.mozilla.org/en-US/developers/addon/api/key/
2. Run `export WEB_EXT_API_KEY=...; export WEB_EXT_API_SECRET=...; npm run package:extension`
3. Install the generated and signed .xpi

## Tested in
* Firefox
* Chrome
