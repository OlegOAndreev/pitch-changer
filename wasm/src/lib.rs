mod basic_vocoder;
mod phase_gradient_vocoder;
mod stft;
mod pitch_shifter;
mod util;
mod web;

pub use pitch_shifter::{PitchShifter, StretchMethod, StretchParams};
pub use util::{cross_correlation, generate_sine_wave};

// Original source: https://github.com/OlegOAndreev/scratch/blob/master/audio-stretch/audio-stretch.cpp

// General notes on various pitch-shifting methods:
// http://blogs.zynaptiq.com/bernsee/time-pitch-overview/
//
// Basic phase vocoder implementation: http://downloads.dspdimension.com/smbPitchShift.cpp
//
// Desription of the phase gradient approach to improve the phase vocoder:
// Phase Vocoder Done Right, Zdeneˇk Pru ̊ša and Nicki Holighaus, Acoustics Research Institute,
// Austrian Academy of Sciences Vienna, Austria
