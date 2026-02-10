# Pitch-change

This is a web interface to allow stretching time and shifting pitch of audio. The core 

## Browser support
For now (Feb 2026) the up to date browser is required, see https://caniuse.com/wf-atomics-wait-async


## Development

### Preparing environment
Install node.js and run to install all the required libraries and scripts.
```bash
npm install
./wasm/install-build-deps.sh
```

### Building
```bash
npm run build
```

Builds in the `dist/` directory.

### Running dev server
```
npx vite
```

Starts the local dev server with URL http://localhost:5173/pitch-changer/

## License
MIT: https://opensource.org/license/mit
