mod phase_gradient_time_stretch;
mod stft;
mod time_stretcher;
mod util;
mod web;
mod window;

pub use time_stretcher::{StretchParams, TimeStretcher};
pub use util::generate_sine_wave;
pub use window::WindowType;

// Original source: https://github.com/OlegOAndreev/scratch/blob/master/audio-stretch/audio-stretch.cpp. Unlike the
// original code, we currently do time stretching using STFT and pitch shift by resampling the stretched audio, just
// like almost every other implementation.
//
// The reason for going stretch -> resample way is that I could not find the way to get stable results from shifting
// frequencies: we either get bin aliasing and troubles with finding a good phase for them (when pitch_shift < 1.0) or
// get empty bins (when pitch_shift > 1.0). See
// https://fileadmin.cs.lth.se/cs/Personal/Pierre_Nugues/memoires/erik/polyphonic_pitch_modification.pdf for description
// of various interpolations to try when altering frequencies.
//
// See also:
//   * very nice presentation https://www.youtube.com/watch?v=fJUmmcGKZMI
//   * https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/ (and
//     http://downloads.dspdimension.com/smbPitchShift.cpp)
//   * New Phase-Vocoder Techniques for Pitch-Shifting, Harmonizing and Other Exotic Effects by Jean Laroche and Mark
// Dolson
//   * https://www.panix.com/~jens/pvoc-dolson.par
//   * https://nl.mathworks.com/help/audio/ug/pitch-shifting-and-time-dilation-using-a-phase-vocoder-in-matlab.html
//   * https://www.mdpi.com/2076-3417/6/2/57
//   * Phase Vocoder Done Right by Zdeneˇk Pru ̊ša and Nicki Holighaus
//   * https://danishsoundcluster.dk/wp-content/uploads/2023/03/Danish_Sound_Vocoder_Report.pdf
//   * https://github.com/oramics/dsp-kit/blob/master/docs/phase-vocoder.md
