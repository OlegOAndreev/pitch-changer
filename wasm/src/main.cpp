#include <emscripten/bind.h>

#include "podarray.h"

// NOTE: Adapted from https://github.com/OlegOAndreev/scratch/tree/master/audio-stretch

// General notes on various pitch-shifting methods:
// http://blogs.zynaptiq.com/bernsee/time-pitch-overview/
//
// Basic phase vocoder implementation: http://downloads.dspdimension.com/smbPitchShift.cpp
//
// Desription of the phase gradient approach to improve the phase vocoder:
// Phase Vocoder Done Right, Zdeneˇk Pru ̊ša and Nicki Holighaus, Acoustics Research Institute,
// Austrian Academy of Sciences Vienna, Austria

// enum class StretchMethod {
//     Simple = 1,
//     Stft = 2
// };

// struct StretchParams {
//     double pitchShift = 1.0;
//     double timeStretch = 1.0;
//     int rate = 0;
//     size_t fftSize = 2048;
//     int overlap = 4;
//     bool phaseGradient = false;
// };

// struct SoundData {
//     int rate = 0;
//     int numChannels = 0;
//     size_t numSamples = 0;
//     std::vector<uint8_t> samples;

//     size_t getByteLength() const
//     {
//         return numChannels * numSamples * sizeof(float);
//     }
// };

// // A simple array_view-like structure, presenting a linear array as a row-major matrix.
// template <typename T>
// struct Span2d {
//     T* begin;
//     size_t rows;
//     size_t columns;

//     Span2d(T* begin_, size_t rows_, size_t columns_)
//         : begin(begin_)
//         , rows(rows_)
//         , columns(columns_)
//     {
//     }

//     // Returns the pointer to the row n.
//     FORCE_INLINE T* row(size_t n)
//     {
//         return begin + n * columns;
//     }

//     // Returns the reference to the element at the position (row, column).
//     FORCE_INLINE T& operator()(size_t row, size_t column)
//     {
//         return begin[row * columns + column];
//     }
// };


// // Returns clamped values, assumes that ST can hold all values from DT.
// template<typename ST, typename DT>
// DT castAndClamp(ST v)
// {
//     static ST dstMin = (ST)std::numeric_limits<DT>::min();
//     static ST dstMax = (ST)std::numeric_limits<DT>::max();
//     return std::min(std::max(v, dstMin), dstMax);
// }

// // Struct, holding the necessary state to resample audio chunk by chunk. The main use-case for
// // the struct is passing the last position (and, potentially, last value) from resample
// // for the previous chunk to resample for the next chunk.
// template<typename ST>
// struct LinearResampleState {
//     int numChannels;

//     // Iterator position, stored to continue the resampling for the next block. Positions
//     // and deltas are encoded as int+float pair. If nextSrcIntPos == SIZE_MAX,
//     // lastSample must be used as a first element before processing next src block.
//     size_t nextSrcIntPos = 0;
//     double nextSrcFloatPos = 0.0;
//     // This value is used if the nextSrcIntPos == SIZE_MAX.
//     std::vector<ST> lastSample;

//     LinearResampleState(int numChannels_)
//         : numChannels(numChannels_)
//     {
//         lastSample.resize(numChannels);
//     }
// };

// // Does the cheapest of the cheapest resamples from src to dst, writes ~ srcSamples * stretch
// // samples. The resample works very poorly for small stretch values (e.g. < 0.5) as it basically
// // starts ignoring every second src sample.
// template<typename ST, typename DT>
// size_t resampleChunk(LinearResampleState<ST>* state, const ST* src, size_t srcSamples,
//                      double stretch, DT* dst, size_t dstRemaining)
// {
//     int numChannels = state->numChannels;
//     Span2d<const ST> srcS(src, srcSamples, numChannels);
//     Span2d<DT> dstS(dst, dstRemaining, numChannels);
//     if (stretch == 1.0) {
//         if (srcSamples > dstRemaining) {
//             printf("Overflow dst: %d vs %d\n", (int)srcSamples, (int)dstRemaining);
//             exit(1);
//         }
//         if (state->nextSrcIntPos != 0 || state->nextSrcFloatPos != 0.0) {
//             printf("Stretch 1.0 should always keep lastSrcIntPos/lastSrcFloatPos equal to zero\n");
//             exit(1);
//         }
//         for (size_t pos = 0; pos < srcSamples; pos++) {
//             for (int ch = 0; ch < numChannels; ch++) {
//                 dstS(pos, ch) = castAndClamp<ST, DT>(srcS(pos, ch));
//             }
//         }
//         return srcSamples;
//     } else {
//         double stretchRcp = 1.0 / stretch;
//         size_t srcIntDelta = (size_t)stretchRcp;
//         double srcFloatDelta = stretchRcp - srcIntDelta;
//         size_t dstPos = 0;
//         size_t srcIntPos = state->nextSrcIntPos;
//         double srcFloatPos = state->nextSrcFloatPos;
//         // Process src positions between the last sample from the previous block and first sample
//         // from the current block.
//         if (srcIntPos == SIZE_MAX) {
//             while (true) {
//                 // Sanity check that dst is allocated correctly.
//                 if (dstPos >= dstRemaining) {
//                     printf("Overflow dst: %d vs %d\n", (int)dstPos, (int)dstRemaining);
//                     exit(1);
//                 }
//                 for (int ch = 0; ch < numChannels; ch++) {
//                     ST d = state->lastSample[ch] * (1 - srcFloatPos) + srcS(0, ch) * srcFloatPos;
//                     dstS(dstPos, ch) = castAndClamp<ST, DT>(d);
//                 }
//                 dstPos++;
//                 srcIntPos += srcIntDelta;
//                 srcFloatPos += srcFloatDelta;
//                 if (srcFloatPos >= 1.0) {
//                     srcIntPos++;
//                     srcFloatPos -= 1.0;
//                 }
//                 // Use defined overflow for size_t. NOTE: This would certainly be cleaner
//                 // if ssize_t has been used for srcIntPos and lastSrcIntPos.
//                 if (srcIntPos != SIZE_MAX) {
//                     break;
//                 }
//             }
//         }
//         // Main loop for processing samples from src.
//         while (true) {
//             if (srcIntPos >= srcSamples - 1) {
//                 break;
//             }
//             // Sanity check that dst is allocated correctly.
//             if (dstPos >= dstRemaining) {
//                 printf("Overflow dst: %d vs %d\n", (int)dstPos, (int)dstRemaining);
//                 exit(1);
//             }
//             for (int ch = 0; ch < numChannels; ch++) {
//                 ST d = srcS(srcIntPos, ch) * (1 - srcFloatPos)
//                         + srcS(srcIntPos + 1, ch) * srcFloatPos;
//                 dstS(dstPos, ch) = castAndClamp<ST, DT>(d);
//             }
//             dstPos++;
//             srcIntPos += srcIntDelta;
//             srcFloatPos += srcFloatDelta;
//             if (srcFloatPos >= 1.0) {
//                 srcIntPos++;
//                 srcFloatPos -= 1.0;
//             }
//         }
//         if (srcIntPos == srcSamples - 1) {
//             // The src position is between the last sample from the current block and
//             // the first sample of the next block. Store the last sample for interpolating
//             // during next call to resampleChunk.
//             state->nextSrcIntPos = SIZE_MAX;
//             for (int ch = 0; ch < numChannels; ch++) {
//                 state->lastSample[ch] = srcS(srcIntPos, ch);
//             }
//         } else {
//             state->nextSrcIntPos = srcIntPos - srcSamples;
//         }
//         state->nextSrcFloatPos = srcFloatPos;
//         return dstPos;
//     }
// }

// template<typename T>
// size_t simpleStretchSoundSamples(const T* src, size_t numSamples, int numChannels,
//                                  const StretchParams& params, T* dst, size_t dstNumSamples)
// {
//     double stretch;
//     if (params.pitchShift != 1.0 && params.timeStretch != 1.0) {
//         printf("Simple stretch method only works if you specify either time stretching"
//                " or pitch shifting\n");
//         exit(1);
//     } else if (params.pitchShift != 1.0) {
//         stretch = 1 / params.pitchShift;
//     } else {
//         stretch = params.timeStretch;
//     }
//     LinearResampleState<T> state(numChannels);
//     return resampleChunk(&state, src, numSamples, stretch, dst, dstNumSamples);
// }

// void fillHannWindow(kiss_fft_scalar* window, size_t fftSize)
// {
//     for (size_t k = 0; k < fftSize; k++) {
//         window[k] = -0.5 * std::cos(2 * M_PI * k / fftSize) + 0.5;
//     }
// }

// // Helper struct for transforming blocks of data with STFT.
// struct StftState {
//     size_t fftSize;
//     size_t offset;
//     int numChannels;
//     std::vector<kiss_fft_scalar> window;
//     kiss_fftr_cfg fftCfg;
//     kiss_fftr_cfg fftiCfg;

//     // Buffers modified at each STFT step.
//     std::vector<kiss_fft_scalar> srcBuf;
//     std::vector<kiss_fft_cpx> freqBuf;
//     std::vector<kiss_fft_cpx> dstFreqBuf;
//     std::vector<kiss_fft_scalar> dstBuf;

//     StftState(size_t fftSize_, size_t offset_, int numChannels_)
//         : fftSize(fftSize_)
//         , offset(offset_)
//         , numChannels(numChannels_)
//     {
//         window.resize(fftSize);
//         fftCfg = kiss_fftr_alloc(fftSize, 0, nullptr, nullptr);
//         fftiCfg = kiss_fftr_alloc(fftSize, 1, nullptr, nullptr);

//         srcBuf.resize(fftSize * numChannels);
//         dstBuf.resize(fftSize * numChannels);
//         freqBuf.resize(fftSize / 2 + 1);
//         dstFreqBuf.resize(fftSize / 2 + 1);
//     }

//     ~StftState()
//     {
//         kiss_fftr_free(fftCfg);
//         kiss_fftr_free(fftiCfg);
//     }

//     kiss_fft_scalar* srcBufCh(int ch)
//     {
//         return srcBuf.data() + fftSize * ch;
//     }

//     kiss_fft_scalar* dstBufCh(int ch)
//     {
//         return dstBuf.data() + fftSize * ch;
//     }
// };

// // Helper struct, containing state for the simple phase vocoder.
// struct SimpleVocoderState {
//     size_t freqSize;
//     // Phases from analysis from previous frame.
//     std::vector<kiss_fft_scalar> prevAnaPhases;
//     // Phases from synthesis from previous frame.
//     std::vector<kiss_fft_scalar> prevSynPhases;

//     // Scratch buffers.
//     std::vector<kiss_fft_scalar> synMagnitudes;
//     std::vector<kiss_fft_scalar> synPhaseDiffs;

//     SimpleVocoderState(size_t fftSize, int numChannels)
//     {
//         freqSize = fftSize / 2 + 1;
//         prevAnaPhases.resize(freqSize * numChannels);
//         prevSynPhases.resize(freqSize * numChannels);
//         synMagnitudes.resize(freqSize);
//         synPhaseDiffs.resize(freqSize);
//     }

//     kiss_fft_scalar* prevAnaPhasesCh(int channel)
//     {
//         return prevAnaPhases.data() + freqSize * channel;
//     }

//     kiss_fft_scalar* prevSynPhasesCh(int channel)
//     {
//         return prevSynPhases.data() + freqSize * channel;
//     }
// };

// // Returns the phaseDiff normalized back to [-M_PI, M_PI] range.
// kiss_fft_scalar normalizePhase(kiss_fft_scalar phaseDiff)
// {
//     // Assumes that phaseDiff is not that far the range so that we do not use fmod, which
//     // can be pretty slow.
// #if 1
//     if (phaseDiff < -M_PI) {
//         do {
//             phaseDiff += M_PI * 2;
//         } while (phaseDiff < -M_PI);
//     } else if (phaseDiff > M_PI) {
//         do {
//             phaseDiff -= M_PI * 2;
//         } while (phaseDiff > M_PI);
//     }
//     return phaseDiff;
// #else
//     if (phaseDiff > -M_PI) {
//         return std::fmod(phaseDiff + M_PI, M_PI * 2) - M_PI;
//     } else {
//         return M_PI + std::fmod(phaseDiff + M_PI, M_PI * 2);
//     }
// #endif
// }

// // Standard simple phase vocoder implementation: takes stft->freq and computes stft->dstFreq
// // for given channel with given pitchShift.
// void stretchFreqSimple(StftState* stft, SimpleVocoderState* vocoder, double pitchShift, int channel)
// {
//     size_t freqSize = stft->fftSize / 2 + 1;
//     kiss_fft_cpx* freq = stft->freqBuf.data();
//     kiss_fft_scalar* prevAnaPhases = vocoder->prevAnaPhasesCh(channel);
//     kiss_fft_scalar* prevSynPhases = vocoder->prevSynPhasesCh(channel);
//     kiss_fft_scalar* synPhaseDiffs = vocoder->synPhaseDiffs.data();
//     kiss_fft_scalar* synMagnitudes = vocoder->synMagnitudes.data();

//     size_t overlap = stft->fftSize / stft->offset;
//     // Overlap must be power-of-two, so use this fact for optimization:
//     // replace the k % overlap with k & overlapMask. You can always replace k * origPhaseMult with
//     // (k % overlap) * origPhaseMult.
//     size_t overlapMask = overlap - 1;
//     if ((overlapMask & overlap) != 0) {
//         printf("Overlap must be pow-of-2\n");
//         exit(1);
//     }
//     // origPhaseMult is used to calculate the expected phase difference for a basis for
//     // frequency bin k: e^(2 * pi * k * t / freqSize) = e^(2 * pi * k / overlap)
//     // when t = stft->offset, therefore the expected phase at t = stft->offset equals
//     // to origPhaseMult * k. In order to constrain the phase difference to [0, 2 * pi), simply
//     // replace the k with (k % overlap): origPhaseMult * (k % overlap). As noted above,
//     // this is equivalent to origPhaseMulti * (k & overlapMask).
//     double origPhaseMult = 2 * M_PI / overlap;

//     std::fill(synMagnitudes, synMagnitudes + freqSize, 0.0);
//     std::fill(synPhaseDiffs, synPhaseDiffs + freqSize, 0.0);

//     for (size_t k = 0; k < freqSize; k++) {
//         size_t newk = k * pitchShift;
//         if (newk >= freqSize) {
//             break;
//         }

//         kiss_fft_scalar magn = sqrt(freq[k].r * freq[k].r + freq[k].i * freq[k].i);
//         bool largeMagn = (magn > synMagnitudes[newk]);
//         synMagnitudes[newk] += magn;
//         kiss_fft_scalar phase = std::atan2(freq[k].i, freq[k].r);
//         // If multiple old phase bins stretch into one new phase bin (when pitchShift < 1.0),
//         // we add their magnitudes but want to want to choose only one phase (summing up phases
//         // from multiple bins make no sense). Therefore, we separately compute newPhaseDiffs
//         // and then add them to prevNewPhases in the end when finally computing dstFreq. Ideally
//         // we would choose the phase from the bin with the highest magnitude here.
//         if (largeMagn) {
//             // Original phase diff is the difference between the potential phase (phase of
//             // the frequency bin k at the end of the previous block) and the actual phase
//             // for the frequency bin k at the start of the new block. This phase diff is then
//             // applied to the stretched frequencies. The final formula is simplified version of
//             // the one from smbPitchShift.cpp. Modulo overlapMask is important here, because
//             // otherwise the origPhaseMult * k can be very large, while the normalizePhase
//             // assumes the parameter value to be near 0.
//             kiss_fft_scalar anaPhaseDiff = normalizePhase(phase - prevAnaPhases[k]
//                                                           - origPhaseMult * (k & overlapMask));
//             // The following two lines are equivalent to
//             //   synPhaseDiffs[newk] = pitchShift * (anaPhaseDiff + origPhaseMult * k).
//             // The main difference is that the synPhaseDiffs is much closer to zero, so that
//             // normalizePhase would take less time later.
//             kiss_fft_scalar synPhaseDiff = anaPhaseDiff * pitchShift
//                     + (k * pitchShift - newk) * origPhaseMult;
//             synPhaseDiffs[newk] = origPhaseMult * (newk & overlapMask) + synPhaseDiff;
//         }
//         prevAnaPhases[k] = phase;
//     }

//     kiss_fft_cpx* dstFreq = stft->dstFreqBuf.data();
//     for (size_t k = 0; k < freqSize; k++) {
//         // Do the normalizePhase here so that the newPhases do not become too large, reducing the
//         // the floating point error.
//         prevSynPhases[k] = normalizePhase(prevSynPhases[k] + synPhaseDiffs[k]);
//         dstFreq[k].r = synMagnitudes[k] * std::cos(prevSynPhases[k]);
//         dstFreq[k].i = synMagnitudes[k] * std::sin(prevSynPhases[k]);
//     }
// }


// // Helper struct, containing the state for the phase vocoder with phase gradient method.
// struct PhaseGradientVocoderState {
//     struct HeapElem {
//         // Frequency bin.
//         size_t freqIdx;
//         // Magnitude for the frequency.
//         kiss_fft_scalar magn;
//         // True if the element is from the previous STFT frame, false if from the current frame.
//         bool prevFrame;

//         bool operator<(HeapElem const& other) const {
//             return magn < other.magn;
//         }
//     };

//     size_t freqSize;
//     std::vector<kiss_fft_scalar> prevAnaMagnitudes;
//     std::vector<kiss_fft_scalar> prevAnaPhases;
//     std::vector<kiss_fft_scalar> prevSynPhases;

//     // Scratch buffers.
//     std::vector<kiss_fft_scalar> anaMagnitudes;
//     std::vector<kiss_fft_scalar> anaPhases;
//     std::vector<kiss_fft_scalar> synMagnitudes;
//     std::vector<kiss_fft_scalar> synPhases;
//     std::priority_queue<HeapElem> maxHeap;
//     // If true, synPhases is assigned.
//     std::vector<uint8_t> phaseAssigned;

//     PhaseGradientVocoderState(size_t fftSize, int numChannels)
//     {
//         freqSize = fftSize / 2 + 1;
//         prevAnaMagnitudes.resize(freqSize * numChannels);
//         prevAnaPhases.resize(freqSize * numChannels);
//         prevSynPhases.resize(freqSize * numChannels);

//         anaMagnitudes.resize(freqSize);
//         anaPhases.resize(freqSize);
//         synMagnitudes.resize(freqSize);
//         synPhases.resize(freqSize);
//         phaseAssigned.resize(freqSize);
//     }

//     kiss_fft_scalar* prevAnaMagnitudesCh(int channel)
//     {
//         return prevAnaMagnitudes.data() + freqSize * channel;
//     }

//     kiss_fft_scalar* prevAnaPhasesCh(int channel)
//     {
//         return prevAnaPhases.data() + freqSize * channel;
//     }

//     kiss_fft_scalar* prevSynPhasesCh(int channel)
//     {
//         return prevSynPhases.data() + freqSize * channel;
//     }
// };

// // Phase vocoder with the phase gradient impl: takes stft->freq and compute sstft->dstFreq for given
// // channel with given pitchShift.
// void stretchFreqPhaseGradient(StftState* stft, PhaseGradientVocoderState* vocoder,
//                               double pitchShift, int channel)
// {
//     // Ignore the frequencies with magnitude < (max magnitude * kMaxMagnitudeTolerance).
//     static const float kMinMagnitudeTolerance = 1e-3;

//     size_t freqSize = stft->fftSize / 2 + 1;
//     kiss_fft_cpx* freq = stft->freqBuf.data();
//     kiss_fft_scalar* prevAnaMagnitudes = vocoder->prevAnaMagnitudesCh(channel);
//     kiss_fft_scalar* prevAnaPhases = vocoder->prevAnaPhasesCh(channel);
//     kiss_fft_scalar* prevSynPhases = vocoder->prevSynPhasesCh(channel);
//     kiss_fft_scalar* anaMagnitudes = vocoder->anaMagnitudes.data();
//     kiss_fft_scalar* anaPhases = vocoder->anaPhases.data();
//     kiss_fft_scalar* synMagnitudes = vocoder->synMagnitudes.data();
//     kiss_fft_scalar* synPhases = vocoder->synPhases.data();
//     std::priority_queue<PhaseGradientVocoderState::HeapElem>& maxHeap = vocoder->maxHeap;
//     uint8_t* phaseAssigned = vocoder->phaseAssigned.data();

//     size_t overlap = stft->fftSize / stft->offset;
//     // Overlap must be power-of-two, so use this fact for optimization:
//     // replace the k % overlap with k & overlapMask. You can always replace k * origPhaseMult with
//     // (k % overlap) * origPhaseMult.
//     size_t overlapMask = overlap - 1;
//     if ((overlapMask & overlap) != 0) {
//         printf("Overlap must be pow-of-2\n");
//         exit(1);
//     }
//     double origPhaseMult = 2 * M_PI / overlap;

//     kiss_fft_scalar maxMagn = 0.0;
//     for (size_t k = 0; k < freqSize; k++) {
//         kiss_fft_scalar magn = sqrt(freq[k].r * freq[k].r + freq[k].i * freq[k].i);
//         anaMagnitudes[k] = magn;
//         maxMagn = std::max(std::max(maxMagn, anaMagnitudes[k]), prevAnaMagnitudes[k]);
//     }
//     kiss_fft_scalar minMagn = maxMagn * kMinMagnitudeTolerance;

//     // std::priroity_queue has no clear().
//     while (!maxHeap.empty()) {
//         maxHeap.pop();
//     }

//     std::fill(synMagnitudes, synMagnitudes + freqSize, 0.0);
//     std::fill(synPhases, synPhases + freqSize, 0.0);
//     // Number of zeroes in phaseAssigned array.
//     int numUnassigned = 0;
//     for (size_t k = 0; k < freqSize; k++) {
//         size_t newk = k * pitchShift;
//         if (newk >= freqSize) {
//             break;
//         }

//         kiss_fft_scalar magn = anaMagnitudes[k];
//         // If pitchShift < 1.0, several analysis frequency bins may correspond to one synthesis bin,
//         // therefore, add, not assign.
//         synMagnitudes[newk] += magn;

//         // Optimization: do not compute phase for frequencies below the minMagn threshold.
//         if (magn > minMagn) {
//             anaPhases[k] = std::atan2(freq[k].i, freq[k].r);
//             phaseAssigned[k] = false;
//             numUnassigned++;
//             maxHeap.push({k, prevAnaMagnitudes[k], true});
//         } else {
//             // The original paper assigns random values to frequencies below the min magnitude,
//             // but we simply leave the phase to be 0.0 (see std::fill above for synPhases).
//             anaPhases[k] = 0.0;
//             phaseAssigned[k] = true;
//         }
//     }

//     while (numUnassigned > 0) {
//         if (maxHeap.empty()) {
//             printf("INTERNAL ERROR: no more elements remaining in the heap, %d still unassigned\n",
//                    numUnassigned);
//             break;
//         }
//         PhaseGradientVocoderState::HeapElem topElem = maxHeap.top();
//         maxHeap.pop();
//         size_t k = topElem.freqIdx;
//         if (topElem.prevFrame) {
//             if (!phaseAssigned[k]) {
//                 phaseAssigned[k] = true;
//                 numUnassigned--;
//                 maxHeap.push({k, anaMagnitudes[k], false});

//                 size_t newk = k * pitchShift;
//                 if (newk < freqSize) {
//                     // Original phase diff is the difference between the potential phase (phase
//                     // of the frequency bin k at the end of the previous block) and the actual
//                     // phase for the frequency bin k at the start of the new block. This phase
//                     // diff is then applied to the stretched frequencies. The final formula
//                     // is simplified a bit from the one from smbPitchShift.cpp
//                     // Modulo overlapMask is important here, because otherwise the origPhaseMult * k
//                     // can be very large and normalizePhase expects values close to 0.
//                     kiss_fft_scalar anaPhaseDiff = normalizePhase(
//                                 anaPhases[k] - prevAnaPhases[k]
//                                 - origPhaseMult * (k & overlapMask));
//                     // The following two lines are basically equivalent to
//                     //   synPhaseDiffs[newk] = pitchShift * (anaPhaseDiff + origPhaseMult * k).
//                     // The main difference is that the synPhaseDiffs is much closer to zero, so that
//                     // normalizePhase would take less time later.
//                     kiss_fft_scalar synPhaseDiff = anaPhaseDiff * pitchShift
//                             + (k * pitchShift - newk) * origPhaseMult;
//                     synPhases[newk] = prevSynPhases[newk] + synPhaseDiff
//                             + origPhaseMult * (newk & overlapMask);
//                 }
//             }
//         } else {
//             if (k > 0 && !phaseAssigned[k - 1]) {
//                 phaseAssigned[k - 1] = true;
//                 numUnassigned--;
//                 maxHeap.push({k - 1, anaMagnitudes[k - 1], false});

//                 size_t newk1 = (k - 1) * pitchShift;
//                 size_t newk = k * pitchShift;
//                 if (newk < freqSize && newk1 != newk) {
//                     synPhases[newk1] = synPhases[newk] - anaPhases[k] + anaPhases[k - 1];
//                 }
//             }
//             if (k < freqSize - 1 && !phaseAssigned[k + 1]) {
//                 phaseAssigned[k + 1] = true;
//                 numUnassigned--;
//                 maxHeap.push({k + 1, anaMagnitudes[k + 1], false});

//                 size_t newk1 = (k + 1) * pitchShift;
//                 size_t newk = k * pitchShift;
//                 if (newk1 < freqSize && newk1 != newk) {
//                     synPhases[newk1] = synPhases[newk] - anaPhases[k] + anaPhases[k + 1];
//                 }
//             }
//         }
//     }

//     kiss_fft_cpx* dstFreq = stft->dstFreqBuf.data();
//     for (size_t k = 0; k < freqSize; k++) {
//         // Do the normalizePhase here so that the prevNewPhases does not become too large,
//         // reducing the the floating point error.
//         prevSynPhases[k] = normalizePhase(synPhases[k]);
//         dstFreq[k].r = synMagnitudes[k] * std::cos(prevSynPhases[k]);
//         dstFreq[k].i = synMagnitudes[k] * std::sin(prevSynPhases[k]);
//     }

//     std::copy(anaMagnitudes, anaMagnitudes + freqSize, prevAnaMagnitudes);
//     std::copy(anaPhases, anaPhases + freqSize, prevAnaPhases);
// }

// // Takes stft->srcBuf, does STFT, changes pitch by pitchShift and does inverse STFT to stft->dstBuf.
// void doStftPitchChange(StftState* stft, SimpleVocoderState* simpleVocoder,
//                        PhaseGradientVocoderState* gradientVocoder,
//                        double pitchShift, bool phaseGradient)
// {
//     for (int channel = 0; channel < stft->numChannels; channel++) {
//         kiss_fftr(stft->fftCfg, stft->srcBuf.data() + stft->fftSize * channel,
//                   stft->freqBuf.data());
//         if (phaseGradient) {
//             stretchFreqPhaseGradient(stft, gradientVocoder, pitchShift, channel);
//         } else {
//             stretchFreqSimple(stft, simpleVocoder, pitchShift, channel);
//         }
//         kiss_fftri(stft->fftiCfg, stft->dstFreqBuf.data(), stft->dstBuf.data()
//                    + stft->fftSize * channel);
//     }
// }

// // The simple STFT pitch shifter: changes frequencies (shifts frequencies for each individual
// // STFT block separately) and then basically does the linear interpolation like the simple stretch.
// // The algorithm is described in http://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/
// template<typename T>
// size_t stftStretchSoundSamples(const T* src, size_t numSamples, int numChannels,
//                                const StretchParams& params, T* dst, size_t dstNumSamples)
// {
//     // See comments when defining the window below for explanation.
//     if (params.overlap < 4) {
//         printf("STFT requires overlap >= 4\n");
//         exit(1);
//     }

//     // Offset of next block to the previous one must always be fftSize / (n * 2) so that
//     // sum of offset hann windows is always constant.
//     size_t fftSize = params.fftSize;
//     size_t offset = fftSize / params.overlap;
//     // If we are time stretching, we have to compensate with corresponding pitch shift.
//     double finalPitchShift = params.pitchShift * params.timeStretch;

//     Span2d<const T> srcV(src, numSamples, numChannels);
//     Span2d<T> dstV(dst, numSamples, numChannels);
//     StftState stft(fftSize, offset, numChannels);
//     SimpleVocoderState simpleVocoder(fftSize, numChannels);
//     PhaseGradientVocoderState phaseGradientVocoder(fftSize, numChannels);
//     // STFT outputs are accumulated in circular buffer in dstAccumBuf. The part of dstAccumBuf,
//     // for which all overlapped blocks have been summed, will be stretched and written to dst.
//     std::vector<kiss_fft_scalar> dstAccumBuf;
//     dstAccumBuf.resize(fftSize * numChannels);
//     Span2d<kiss_fft_scalar> dstAccumBufV(dstAccumBuf.data(), fftSize, numChannels);

//     size_t dstWritten = 0;

//     // One window will be used both for analysis and synthesis (i.e. forward and inverse
//     // transforms). This idea is taken from smbPitchShift.cpp and is based on the fact that
//     // for overlaps >= 4 the sum of squared hann window is constant.
//     // The explanation why having windows both for forward and inverse transform is more optimal
//     // (instead of just forward transform) is given here: https://gauss256.github.io/blog/cola.html:
//     //
//     // > As mentioned above, it is most common to choose a = 1. The reason is that in the case
//     // > where we did modify the STFT, there may not be a time-domain signal whose STFT matches
//     // > our modified version. Choosing a = 1 gives the Griffin-Lim
//     // > [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.7858&rep=rep1&type=pdf]
//     // > optimal estimate (optimal in a least-squares sense) for a time-domain signal from
//     // > a modified STFT. SciPy implements a = 1 for the signal reconstruction in the istft routine.
//     kiss_fft_scalar* window = stft.window.data();
//     fillHannWindow(window, fftSize);

//     LinearResampleState<kiss_fft_scalar> resampleState(numChannels);
//     // Start with negative block offset, because each element must be overlapped by all possible
//     // offsets windows. Real block offset is block - fftSize, but we do not want to use
//     // signed integers here.
//     for (size_t block = offset; block < fftSize; block += offset) {
//         size_t prefix = fftSize - block;
//         // read != fftSize-prefix holds only when the numSamples < fftSize-offset.
//         size_t read = std::min(fftSize - prefix, numSamples);
//         for (int channel = 0; channel < numChannels; channel++) {
//             // Prepare srcBuf. The first part for negative block offsets is prefilled with zeros.
//             kiss_fft_scalar *srcBuf = stft.srcBufCh(channel);
//             for (size_t k = 0; k < prefix; k++) {
//                 srcBuf[k] = 0.0;
//             }
//             for (size_t k = prefix; k < prefix + read; k++) {
//                 srcBuf[k] = (kiss_fft_scalar)srcV(k - prefix, channel) * window[k];
//             }
//             for (size_t k = prefix + read; k < fftSize; k++) {
//                 srcBuf[k] = 0.0;
//             }
//         }

//         doStftPitchChange(&stft, &simpleVocoder, &phaseGradientVocoder, finalPitchShift,
//                           params.phaseGradient);

//         // Write dstBuf to dstAccumBuf. For blocks with negative offset we never wrap around
//         // the dstAccumBuf, always start writing at index 0.
//         for (int channel = 0; channel < numChannels; channel++) {
//             kiss_fft_scalar *dstBuf = stft.dstBuf.data();
//             for (size_t k = 0; k < read; k++) {
//                 dstAccumBufV(k, channel) += dstBuf[k + prefix] / fftSize;
//             }
//         }
//     }

//     // Process the full blocks.
//     for (size_t block = 0; block < numSamples; block += offset) {
//         size_t read = std::min(fftSize, numSamples - block);
//         size_t accumStart = block % fftSize;
//         for (int channel = 0; channel < numChannels; channel++) {
//             // Prepare srcBuf.
//             kiss_fft_scalar* srcBuf = stft.srcBufCh(channel);
//             for (size_t k = 0; k < read; k++) {
//                 srcBuf[k] = (kiss_fft_scalar)srcV(block + k, channel) * window[k];
//             }
//             for (size_t k = read; k < fftSize; k++) {
//                 srcBuf[k] = 0.0;
//             }
//         }

//         doStftPitchChange(&stft, &simpleVocoder, &phaseGradientVocoder, finalPitchShift,
//                           params.phaseGradient);

//         for (int channel = 0; channel < numChannels; channel++) {
//             kiss_fft_scalar* dstBuf = stft.dstBufCh(channel);
//             // Write dstBuf to dstAccumBuf. We want to write dstBuf[k]
//             // to dstAccumBufS((block + k) % fftSize). The last part dstBuf[fftSize - offset,
//             // fftSize) must overwrite, not add to the dstAccumBuf. Thus, we have to do two
//             // or three separate loops, the last one overwriting, not adding the data.
// #define DST_BUF(k) (dstBuf[k] * window[k] * 4) / (fftSize * params.overlap)
//             if (accumStart != 0) {
//                 // We wrap around the dstAccumBuf, do two loops when writing.
//                 for (size_t k = accumStart; k < fftSize; k++) {
//                     dstAccumBufV(k, channel) += DST_BUF(k - accumStart);
//                 }
//                 for (size_t k = 0; k < accumStart - offset; k++) {
//                     dstAccumBufV(k, channel) += DST_BUF(k + fftSize - accumStart);
//                 }
//                 for (size_t k = accumStart - offset; k < accumStart; k++) {
//                     dstAccumBufV(k, channel) = DST_BUF(k + fftSize - accumStart);
//                 }
//             } else {
//                 for (size_t k = 0; k < fftSize - offset; k++) {
//                     dstAccumBufV(k, channel) += DST_BUF(k);
//                 }
//                 for (size_t k = fftSize - offset; k < fftSize; k++) {
//                     dstAccumBufV(k, channel) = DST_BUF(k);
//                 }
//             }
// #undef DST_BUF
//         }

//         // We can take the dstAccumBuf[accumStart:accumStart + offset], stretch and write it
//         // to the dst.
//         size_t numOutput = std::min(offset, numSamples - block);
//         dstWritten += resampleChunk(&resampleState, dstAccumBufV.row(accumStart), numOutput,
//                                     params.timeStretch, dstV.row(dstWritten),
//                                     dstNumSamples - dstWritten);
//     }

//     return dstWritten;
// }

// template<typename T>
// size_t doStretchSound(const T* src, size_t numSamples, int numChannels, StretchMethod method,
//                       const StretchParams& params, T* dst, size_t dstNumSamples)
// {
//     switch (method) {
//     case StretchMethod::Simple:
//         return simpleStretchSoundSamples(src, numSamples, numChannels, params, dst, dstNumSamples);
//     case StretchMethod::Stft:
//         return stftStretchSoundSamples(src, numSamples, numChannels, params, dst, dstNumSamples);
//     default:
//         ENSURE(false, "Not supported format");
//     }
// }

// struct Magnitude {
//     double magnitude;
//     size_t freq;
//     kiss_fft_scalar r;
//     kiss_fft_scalar i;
// };

// template<typename T>
// void printFreq(T* data, size_t numSamples, int numChannels, int rate)
// {
//     size_t n = 1 << ((size_t)log2(numSamples) + 1);
//     printf("Analyzing %d samples (channel 0)\n", (int)n);

//     kiss_fftr_cfg cfg = kiss_fftr_alloc(n, 0, nullptr, nullptr);
//     std::vector<kiss_fft_scalar> dataCopy;
//     dataCopy.resize(n);
//     for (size_t k = 0; k < std::min(n, numSamples); k++) {
//         dataCopy[k] = (kiss_fft_scalar)data[k * numChannels];
//     }
//     std::vector<kiss_fft_cpx> freq;
//     freq.resize(n / 2 + 1);
//     kiss_fftr(cfg, dataCopy.data(), freq.data());

//     std::vector<Magnitude> magnitudes;
//     magnitudes.resize(n / 2 + 1);
//     for (size_t k = 0; k < n / 2 + 1; k++) {
//         magnitudes[k].magnitude = sqrt(freq[k].r * freq[k].r + freq[k].i * freq[k].i);
//         magnitudes[k].freq = k;
//         magnitudes[k].r = freq[k].r;
//         magnitudes[k].i = freq[k].i;
// #if 0
//         printf("[%d (%d-%dHz)]: %g (%g %g)\n", (int)k, (int)(k * rate / n),
//                (int)((k + 1) * rate / n), magnitudes[k].magnitude, freq[k].r, freq[k].i);
// #endif
//     }
//     std::sort(magnitudes.begin(), magnitudes.end(), [] (const Magnitude& m1, const Magnitude& m2) {
//         return m1.magnitude > m2.magnitude;
//     });
//     for (size_t k = 0; k < 15; k++) {
//         printf("top %d: %d (%d-%dHz) %g (%g %g: %g)\n", (int)k, (int)magnitudes[k].freq,
//                (int)(magnitudes[k].freq * rate / n), (int)((magnitudes[k].freq + 1) * rate / n),
//                magnitudes[k].magnitude, magnitudes[k].r, magnitudes[k].i,
//                std::atan2(magnitudes[k].i, magnitudes[k].r));
//     }

//     kiss_fftr_free(cfg);
// }

// SoundData stretchSound(const SoundData& src, StretchMethod method, const StretchParams& params)
// {
//     SoundData dst;
//     dst.format = src.format;
//     dst.rate = src.rate;
//     dst.numChannels = src.numChannels;
//     // NOTE: Calculating the actual number of samples, which will be written while stretching,
//     // can be quite error-prone due to floating point math errors. The easy way: overallocate
//     // the dst buffer and get the number of samples actually written from the stretching function.
//     dst.numSamples = src.numSamples * params.timeStretch * 1.1 + 1;
//     dst.samples.resize(dst.getByteLength());

//     size_t dstWritten;
//     switch (dst.format) {
//     case SampleFormat::Sint16:
//         dstWritten = doStretchSound((int16_t*)src.samples.data(), src.numSamples, src.numChannels,
//                                     method, params, (int16_t*)dst.samples.data(), dst.numSamples);
//         break;
//     case SampleFormat::Float:
//         dstWritten = doStretchSound((float*)src.samples.data(), src.numSamples, src.numChannels,
//                                     method, params, (float*)dst.samples.data(), dst.numSamples);
//         break;
//     default:
//         ENSURE(false, "Not supported format");
//     }

//     // Sanity check.
//     if (dstWritten > dst.numSamples) {
//         printf("ERROR: Number of samples written is greater than the allocated: %d vs %d\n",
//                (int)dstWritten, (int)dst.numSamples);
//         exit(1);
//     }
//     dst.numSamples = dstWritten;

// #if 0
//     switch (dst.format) {
//     case SampleFormat::Sint16:
//         printFreq((int16_t*)dst.samples.data(), dstWritten, dst.numChannels, dst.rate);
//         break;
//     case SampleFormat::Float:
//         printFreq((float*)dst.samples.data(), dstWritten, dst.numChannels, dst.rate);
//         break;
//     }
// #endif

//     return dst;
// }

EMSCRIPTEN_BINDINGS(main) {
    emscripten::class_<PodArray<float>>("Float32PodArray")
        .constructor<size_t>()
        .function("freeData", &PodArray<float>::freeData)
        .property("dataPtr", &PodArray<float>::dataPtr)
        .property("beginOffset", &PodArray<float>::beginOffset)
        .property("endOffset", &PodArray<float>::endOffset)
        .property("size", &PodArray<float>::getSize);
}
