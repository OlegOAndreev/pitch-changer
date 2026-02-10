# Audio Pitch Changer

This is a web interface to allow stretching time and shifting pitch of audio. The audio can be recorded via microphone
or uploaded from a file. The core is written in Rust and implements a phase vocoder using STFT, the web frontend is
written in TypeScript and uses Web Audio + WASM in Web Workers.

## Browser support
For now (Feb 2026) the up to date browser is required, see https://caniuse.com/wf-atomics-wait-async

## Development

### General instructions

See AGENTS.md for details on how to build and test the project.

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
