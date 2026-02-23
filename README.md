# Audio Pitch Changer

This is a web interface to allow stretching time and shifting pitch of audio. The audio can be recorded via microphone
or uploaded from a file. The core is written in Rust and implements a phase vocoder using STFT, the web frontend is
written in TypeScript and uses Web Audio + WASM in Web Workers. All processing is done locally.

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

## References

Original source: https://github.com/OlegOAndreev/scratch/blob/master/audio-stretch/audio-stretch.cpp. Unlike the
original code, we currently do time stretching using STFT and pitch shift by resampling the stretched audio, just like
almost every other implementation.

The reason for going stretch -> resample way is that I could not find the way to get stable results from shifting
frequencies: we either get bin aliasing and troubles with finding a good phase for them (when pitch_shift < 1.0) or get
empty bins (when pitch_shift > 1.0). See
https://fileadmin.cs.lth.se/cs/Personal/Pierre_Nugues/memoires/erik/polyphonic_pitch_modification.pdf for description of
various interpolations to try when altering frequencies.

See also:
  * very nice presentation https://www.youtube.com/watch?v=fJUmmcGKZMI
  * https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/ (and
    http://downloads.dspdimension.com/smbPitchShift.cpp)
  * New Phase-Vocoder Techniques for Pitch-Shifting, Harmonizing and Other Exotic Effects by Jean Laroche and Mark
    Dolson
  * https://www.panix.com/~jens/pvoc-dolson.par
  * https://nl.mathworks.com/help/audio/ug/pitch-shifting-and-time-dilation-using-a-phase-vocoder-in-matlab.html
  * https://www.mdpi.com/2076-3417/6/2/57
  * Phase Vocoder Done Right by Zdeneˇk Pru ̊ša and Nicki Holighaus
  * https://github.com/oramics/dsp-kit/blob/master/docs/phase-vocoder.md
  * https://www.mdpi.com/2076-3417/6/2/57#TSM_Based_on_the_Phase_Vocoder_PVTS
  * http://recherche.ircam.fr/equipes/analyse-synthese/roebel/paper/trueenv_dafx2005.pdf
  * https://www.diva-portal.org/smash/get/diva2:1381398/FULLTEXT01.pdf
  * https://danishsoundcluster.dk/wp-content/uploads/2023/03/Danish_Sound_Vocoder_Report.pdf
  * https://github.com/jurihock/stftPitchShift

## License
MIT: https://opensource.org/license/mit
