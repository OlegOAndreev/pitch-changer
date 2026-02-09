# Brief instructions for code agents

## General
* This is a web audio pitch modifier, which can either record audio from microphone or upload from file
* General information about this project is in README.md
* The code is written in TypeScript and Rust with minimal dependencies, pure CSS and HTML, bundled using esbuild

## Code location
* TypeScript files are located in src/
* HTML and CSS Files are located in public/
* Build results are located in public/
* Rust files are located in wasm/

## HTML/CSS rules
* Do not embed HTML or CSS in typescript files
* Do absolutely minimal changes to other CSS rules if you do any CSS modification
* When adding new CSS rules, prefer adding id-based rules instead of class-based rules
* Add class-based CSS rules only if you see duplication in rules

## TypeScript rules
* We use SharedArrayBuffer for synchronization with workers and audio worklets

## Audio
* WebAudio produces and accepts all audio in deinterleaved (planar) formats, one Float32Array per channel.
* Internally all data is interleaved for passing around:
  * when using Float32RingBuffer
  * when passing to/from WASM module

## Build & testing commands
* Only attempt running any commands for building and testing as outlined here
* Install JavaScript/TypeScript libraries with `npm install`, do not run any other commands
* When changing TypeScript or html/css files, run `npm run build:ts` and `npm run test:ts`, do not run any other commands
* When changing Rust files, run `npm run build` and `npm run test`
* Do not attempt running any other commands, including `npx`, unless absolutely required (and then ask the user first)
* Run `npm run build` to check the final result in dist/
* Do not attempt opening browser for UI testing, always request the user to test the UI
* Do not attempt API calls, always request the user to test the API
