use std::fs::File;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use argh::FromArgs;
use hound::{WavSpec, WavWriter};
use plotters::prelude::*;

use wasm_main_module::{
    EnvelopeShifter, MultiProcessor, PitchShiftParams, PitchShifter, SpectralHistogram, TimeStretchParams,
    TimeStretcher, WindowType, compute_dominant_frequency, generate_sine_wave, interleave_samples,
};

fn parse_window_type(s: &str) -> Result<WindowType> {
    match s.to_lowercase().as_str() {
        "hann" => Ok(WindowType::Hann),
        "sqrt-hann" => Ok(WindowType::SqrtHann),
        "sqrt-blackman" => Ok(WindowType::SqrtBlackman),
        _ => bail!("Unknown window type '{}'. Supported: hann, sqrt-hann, sqrt-blackman", s),
    }
}

struct AudioContents {
    /// Interleaved audio data
    data: Vec<f32>,
    sample_rate: u32,
    channels: usize,
}

impl AudioContents {
    /// Get the first channel data (mono) for plotting and frequency analysis
    fn first_channel(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.data.clone();
        }
        let num_samples = self.data.len() / self.channels;
        let mut result = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            result.push(self.data[i * self.channels]);
        }
        result
    }
}

fn load_file(filename: &str) -> Result<AudioContents> {
    use symphonia::core::audio::Signal as _;

    let file = File::open(filename)?;
    let mss = symphonia::core::io::MediaSourceStream::new(
        Box::new(file),
        symphonia::core::io::MediaSourceStreamOptions::default(),
    );
    let probed = symphonia::default::get_probe()
        .format(
            &symphonia::core::probe::Hint::new(),
            mss,
            &symphonia::core::formats::FormatOptions::default(),
            &symphonia::core::meta::MetadataOptions::default(),
        )
        .context("unsupported format")?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("no supported audio tracks")?;
    println!("Decoding {} as {:?},", filename, track.codec_params.codec);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &symphonia::core::codecs::DecoderOptions::default())
        .context("unsupported codec")?;

    let mut data = Vec::new();
    let mut sample_rate = 0;
    let mut channels = 0;

    while let Ok(packet) = format.next_packet() {
        let decoded = decoder.decode(&packet).context("failed decoding")?;
        if sample_rate == 0 {
            sample_rate = decoded.spec().rate;
            channels = decoded.spec().channels.count();
        } else if sample_rate != decoded.spec().rate {
            bail!("File contains multiple sample rates in one track: {} vs {}", sample_rate, decoded.spec().rate);
        } else if channels != decoded.spec().channels.count() {
            bail!("File contains varying channel counts: {} vs {}", channels, decoded.spec().channels.count());
        }

        let frames = decoded.frames();

        match decoded {
            symphonia::core::audio::AudioBufferRef::F32(buf) => {
                // Store interleaved
                for i in 0..frames {
                    for ch in 0..channels {
                        data.push(buf.chan(ch)[i]);
                    }
                }
            }
            symphonia::core::audio::AudioBufferRef::S16(buf) => {
                // Store interleaved, convert to f32
                for i in 0..frames {
                    for ch in 0..channels {
                        data.push(buf.chan(ch)[i] as f32 / 32768.0);
                    }
                }
            }
            _ => {
                bail!("Only f32 or s16 audio is supported");
            }
        }
    }

    if data.is_empty() {
        bail!("No audio data found in file");
    }

    println!(
        "Loaded {}: {} samples, {} channels, {} Hz ({} samples per channel)",
        filename,
        data.len(),
        channels,
        sample_rate,
        data.len() / channels
    );

    Ok(AudioContents { data, sample_rate, channels })
}

fn save_wav(data: &[f32], sample_rate: u32, channels: usize, filename: &str) -> Result<()> {
    let spec = WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in data {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    println!("WAV file saved to: {} ({} channels)", filename, channels);
    Ok(())
}

fn save_plot_audio_svg(input: &[f32], output: &[f32], sample_rate: u32, filename: &str) -> Result<()> {
    const INTERVAL_LENGTH: f32 = 0.05;

    let min_length = input.len().min(output.len());
    let min_duration = min_length as f32 / sample_rate as f32;
    let output_length = output.len() as f32 / sample_rate as f32;

    let interval1_start = 0.0;
    let interval1_end = interval1_start + INTERVAL_LENGTH;
    let interval2_start = min_duration * 0.5;
    let interval2_end = interval2_start + INTERVAL_LENGTH;
    let interval3_start = output_length - INTERVAL_LENGTH;
    let interval3_end = output_length;

    let from_sample1 = (sample_rate as f32 * interval1_start) as usize;
    let to_sample1 = (sample_rate as f32 * interval1_end) as usize;
    let from_sample2 = (sample_rate as f32 * interval2_start) as usize;
    let to_sample2 = (sample_rate as f32 * interval2_end) as usize;
    let from_sample3 = (sample_rate as f32 * interval3_start) as usize;
    let to_sample3 = (sample_rate as f32 * interval3_end) as usize;

    let root = SVGBackend::new(filename, (1200, 1800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (top_area, rest_area) = root.split_vertically(600);
    let (middle_area, bottom_area) = rest_area.split_vertically(600);

    let mut top_chart = ChartBuilder::on(&top_area)
        .caption(format!("Beginning: {:.3}-{:.3} s", interval1_start, interval1_end), ("sans-serif", 20))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(interval1_start..interval1_end, -1.2f32..1.2f32)?;
    top_chart.configure_mesh().label_style(("sans-serif", 20)).draw()?;
    top_chart
        .draw_series(LineSeries::new(
            (from_sample1..to_sample1)
                .into_iter()
                .map(|i| (i as f32 / sample_rate as f32, *input.get(i).unwrap_or(&0.0))),
            &BLUE,
        ))?
        .label("Input");
    top_chart
        .draw_series(LineSeries::new(
            (from_sample1..to_sample1)
                .into_iter()
                .map(|i| (i as f32 / sample_rate as f32, *output.get(i).unwrap_or(&0.0))),
            &RED,
        ))?
        .label("Output");

    let mut middle_chart = ChartBuilder::on(&middle_area)
        .caption(format!("Middle: {:.3}-{:.3} s", interval2_start, interval2_end), ("sans-serif", 40))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(interval2_start..interval2_end, -1.2f32..1.2f32)?;

    middle_chart.configure_mesh().label_style(("sans-serif", 20)).draw()?;
    middle_chart
        .draw_series(LineSeries::new(
            (from_sample2..to_sample2)
                .into_iter()
                .map(|i| (i as f32 / sample_rate as f32, *input.get(i).unwrap_or(&0.0))),
            &BLUE,
        ))?
        .label("Input");
    middle_chart
        .draw_series(LineSeries::new(
            (from_sample2..to_sample2)
                .into_iter()
                .map(|i| (i as f32 / sample_rate as f32, *output.get(i).unwrap_or(&0.0))),
            &RED,
        ))?
        .label("Output");

    let mut bottom_chart = ChartBuilder::on(&bottom_area)
        .caption(format!("End: {:.3}-{:.3} s", interval3_start, interval3_end), ("sans-serif", 40))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(interval3_start..interval3_end, -1.2f32..1.2f32)?;

    bottom_chart.configure_mesh().label_style(("sans-serif", 20)).draw()?;
    bottom_chart
        .draw_series(LineSeries::new(
            (from_sample3..to_sample3)
                .into_iter()
                .map(|i| (i as f32 / sample_rate as f32, *input.get(i).unwrap_or(&0.0))),
            &BLUE,
        ))?
        .label("Input");
    bottom_chart
        .draw_series(LineSeries::new(
            (from_sample3..to_sample3)
                .into_iter()
                .map(|i| (i as f32 / sample_rate as f32, *output.get(i).unwrap_or(&0.0))),
            &RED,
        ))?
        .label("Output");

    root.present()?;
    println!("Plot SVG saved to: {}", filename);
    Ok(())
}

fn save_spectrum_plot_svg(
    spectrum: &[f32],
    envelope: &[f32],
    sample_rate: u32,
    fft_size: usize,
    filename: &str,
) -> Result<()> {
    use plotters::prelude::*;

    let num_bins = spectrum.len();
    let max_freq = sample_rate as f32 / 2.0;
    let bin_width = max_freq / (num_bins as f32 - 1.0);

    let spectrum_points: Vec<(f32, f32)> = (0..num_bins).map(|i| (i as f32 * bin_width, spectrum[i])).collect();
    let envelope_points: Vec<(f32, f32)> = (0..envelope.len()).map(|i| (i as f32 * bin_width, envelope[i])).collect();

    let max_magnitude = spectrum.iter().chain(envelope.iter()).copied().fold(0.0f32, f32::max).max(1e-1).min(1000.0);

    let root = SVGBackend::new(filename, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Spectrum (FFT size: {}, Sample rate: {} Hz)", fft_size, sample_rate), ("sans-serif", 20))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_freq, 0.0..max_magnitude * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Magnitude")
        .label_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(LineSeries::new(spectrum_points.clone(), &BLUE))?.label("Spectrum");
    chart
        .draw_series(LineSeries::new(envelope_points.clone(), &RED))?
        .label("Spectral Envelope");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Spectrum plot SVG saved to: {}", filename);
    Ok(())
}

#[derive(FromArgs)]
/// Pitch shift CLI tool
struct Cli {
    #[argh(subcommand)]
    command: Commands,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum Commands {
    Generate(Generate),
    TimeStretch(TimeStretch),
    PitchShift(PitchShift),
    ShowSpectrum(ShowSpectrum),
}

/// Generate a sine wave and save it as WAV
#[derive(FromArgs)]
#[argh(subcommand, name = "generate")]
struct Generate {
    /// frequency in Hz
    #[argh(option, default = "440.0")]
    frequency: f32,

    /// sample rate in Hz
    #[argh(option, default = "44100.0")]
    sample_rate: f32,

    /// duration in seconds
    #[argh(option, default = "3.0")]
    duration: f32,

    /// number of channels (default: 1)
    #[argh(option, default = "1")]
    channels: usize,

    /// output WAV file path
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    output: String,
}

/// Apply time stretch to input
#[derive(FromArgs)]
#[argh(subcommand, name = "time-stretch")]
struct TimeStretch {
    /// input audio file
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    input: String,

    /// time stretch factor (e.g., 2.0 = twice as long, 0.5 = half length)
    #[argh(option, default = "1.0")]
    stretch: f32,

    /// overlap ratio
    #[argh(option, default = "8")]
    overlap: u32,

    /// FFT size
    #[argh(option, default = "4096")]
    fft_size: usize,

    /// window type: "hann", "sqrt-hann", or "sqrt-blackman"
    #[argh(option, default = "\"hann\".to_string()")]
    window: String,

    /// output WAV file for stretched audio
    #[argh(option, default = "\"stretched.wav\".to_string()")]
    output: String,

    /// save plot comparison SVG
    #[argh(switch)]
    plot: bool,
}

/// Apply pitch shift to input
#[derive(FromArgs)]
#[argh(subcommand, name = "pitch-shift")]
struct PitchShift {
    /// input audio file
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    input: String,

    /// pitch shift factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
    #[argh(option, default = "1.0")]
    shift: f32,

    /// overlap ratio
    #[argh(option, default = "8")]
    overlap: u32,

    /// FFT size
    #[argh(option, default = "4096")]
    fft_size: usize,

    /// window type: "hann", "sqrt-hann", or "sqrt-blackman"
    #[argh(option, default = "\"hann\".to_string()")]
    window: String,

    /// output WAV file for pitch shifterd audio
    #[argh(option, default = "\"shifted.wav\".to_string()")]
    output: String,

    /// cepstrum cutoff, either 0.0 to disable formant preservation or somewhere in 0.5-5.0 for preserving formants
    #[argh(option, default = "0.0")]
    quefrency_cutoff: f32,

    /// save plot comparison SVG
    #[argh(switch)]
    plot: bool,
}

/// Show spectrum analysis of audio file
#[derive(FromArgs)]
#[argh(subcommand, name = "show-spectrum")]
struct ShowSpectrum {
    /// input audio file
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    input: String,

    /// offset in seconds
    #[argh(option, default = "5.0")]
    offset: f32,

    /// FFT size
    #[argh(option, default = "4096")]
    fft_size: usize,

    /// cepstrum cutoff (quefrency)
    #[argh(option, default = "0.0")]
    quefrency_cutoff: f32,

    /// output SVG file for spectrum plot
    #[argh(option, default = "\"spectrum.svg\".to_string()")]
    output: String,
}

fn main() -> Result<()> {
    let cli: Cli = argh::from_env();

    match cli.command {
        Commands::Generate(Generate { frequency, sample_rate, duration, channels, output }) => {
            println!(
                "Generating sine wave: {} Hz, {} samples/sec, {} seconds, {} channels",
                frequency, sample_rate, duration, channels
            );
            let mut sine_wave = Vec::new();
            for ch in 0..channels {
                // Generate same frequency for all channels (could be modified)
                let channel_data = generate_sine_wave(frequency, sample_rate, 1.0, duration);
                if ch == 0 {
                    sine_wave = channel_data;
                } else {
                    sine_wave.extend(channel_data);
                }
            }
            // Interleave the channels
            let mut interleaved_sine = Vec::with_capacity(sine_wave.len());
            interleave_samples(&sine_wave, channels, &mut interleaved_sine);
            println!(
                "Generated {} samples ({} per channel)",
                interleaved_sine.len(),
                interleaved_sine.len() / channels
            );

            save_wav(&interleaved_sine, sample_rate as u32, channels, &output)?;
        }

        Commands::TimeStretch(TimeStretch {
            input: input_path,
            stretch,
            output: output_path,
            plot,
            overlap,
            fft_size,
            window,
        }) => {
            let mut start_time = Instant::now();
            let input = load_file(&input_path).with_context(|| format!("loading {} failed", input_path))?;
            println!("Loading {} took {}ms", input_path, Instant::now().duration_since(start_time).as_millis());

            println!(
                "Time stretching {}: {} Hz, {} channels, stretch: {}",
                input_path, input.sample_rate, input.channels, stretch
            );

            let window_type = parse_window_type(&window)?;
            let mut params = TimeStretchParams::new(input.sample_rate, stretch);
            params.overlap = overlap;
            params.fft_size = fft_size;
            params.window_type = window_type;

            let mut output_data = vec![];
            let mut stretcher = MultiProcessor::<TimeStretcher>::new(&params, input.channels)?;
            start_time = Instant::now();
            for chunk in input.data.chunks((fft_size - 1) * input.channels) {
                stretcher.process(chunk, &mut output_data);
            }
            stretcher.finish(&mut output_data);
            println!(
                "Output generated: {} samples in {}ms",
                output_data.len(),
                Instant::now().duration_since(start_time).as_millis()
            );

            save_wav(&output_data, input.sample_rate, input.channels, &output_path)
                .with_context(|| format!("saving {} failed", output_path))?;

            // Compute dominant frequency on first channel
            let input_first = input.first_channel();
            let output_first = if input.channels == 1 {
                output_data.clone()
            } else {
                let num_samples = output_data.len() / input.channels;
                let mut result = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    result.push(output_data[i * input.channels]);
                }
                result
            };

            let input_freq = compute_dominant_frequency(&input_first, input.sample_rate as f32);
            println!("Input dominant frequency (first channel): {:.2} Hz", input_freq);
            let output_freq = compute_dominant_frequency(&output_first, input.sample_rate as f32);
            println!("Output dominant frequency (first channel): {:.2} Hz", output_freq);

            if plot {
                let plot_filename = output_path.replace(".wav", "_plot.svg");
                save_plot_audio_svg(&input_first, &output_first, input.sample_rate, &plot_filename)
                    .with_context(|| format!("saving plot {} failed", plot_filename))?;
            }
        }

        Commands::PitchShift(PitchShift {
            input: input_path,
            shift,
            output: output_path,
            plot,
            overlap,
            fft_size,
            window,
            quefrency_cutoff,
        }) => {
            let mut start_time = Instant::now();
            let input = load_file(&input_path).with_context(|| format!("loading {} failed", input_path))?;
            println!("Loading {} took {}ms", input_path, Instant::now().duration_since(start_time).as_millis());

            println!(
                "Pitch shifting {}: {} Hz, {} channels, shift: {}, quefrency cutoff: {}",
                input_path, input.sample_rate, input.channels, shift, quefrency_cutoff
            );

            let window_type = parse_window_type(&window)?;
            let mut params = PitchShiftParams::new(input.sample_rate, shift);
            params.overlap = overlap;
            params.fft_size = fft_size;
            params.window_type = window_type;
            params.quefrency_cutoff = quefrency_cutoff;

            let mut output_data = vec![];
            let mut shifter = MultiProcessor::<PitchShifter>::new(&params, input.channels)?;
            start_time = Instant::now();
            for chunk in input.data.chunks((fft_size - 1) * input.channels) {
                shifter.process(chunk, &mut output_data);
            }
            shifter.finish(&mut output_data);
            println!(
                "Output generated: {} samples in {}ms",
                output_data.len(),
                Instant::now().duration_since(start_time).as_millis()
            );

            save_wav(&output_data, input.sample_rate, input.channels, &output_path)
                .with_context(|| format!("saving {} failed", output_path))?;

            // Compute dominant frequency on first channel
            let input_first = input.first_channel();
            let output_first = if input.channels == 1 {
                output_data.clone()
            } else {
                let num_samples = output_data.len() / input.channels;
                let mut result = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    result.push(output_data[i * input.channels]);
                }
                result
            };

            let input_freq = compute_dominant_frequency(&input_first, input.sample_rate as f32);
            println!("Input dominant frequency (first channel): {:.2} Hz", input_freq);
            let output_freq = compute_dominant_frequency(&output_first, input.sample_rate as f32);
            println!("Output dominant frequency (first channel): {:.2} Hz", output_freq);

            if plot {
                let plot_filename = output_path.replace(".wav", "_plot.svg");
                save_plot_audio_svg(&input_first, &output_first, input.sample_rate, &plot_filename)
                    .with_context(|| format!("saving plot {} failed", plot_filename))?;
            }
        }

        Commands::ShowSpectrum(ShowSpectrum {
            input: input_path,
            offset,
            fft_size,
            quefrency_cutoff,
            output: output_path,
        }) => {
            let start_time = Instant::now();
            let input = load_file(&input_path).with_context(|| format!("loading {} failed", input_path))?;
            println!("Loading {} took {}ms", input_path, Instant::now().duration_since(start_time).as_millis());

            println!(
                "Showing spectrum of {}: offset {}s, FFT size {}, quefrency cutoff {}",
                input_path, offset, fft_size, quefrency_cutoff
            );

            // Calculate start sample index
            let start_sample = (offset * input.sample_rate as f32) as usize * input.channels;
            let samples_needed = fft_size * input.channels;
            let end_sample = start_sample + samples_needed;

            if end_sample > input.data.len() {
                bail!(
                    "Not enough audio data: need {} samples ({} per channel) starting at sample {}, but file has only {} samples",
                    samples_needed,
                    fft_size,
                    start_sample,
                    input.data.len()
                );
            }

            let audio_slice = &input.data[start_sample..end_sample];
            let mut histogram = SpectralHistogram::new(fft_size);
            let mut spectrum = vec![];
            histogram.compute_vec(audio_slice, input.channels, &mut spectrum);

            let mut envelope = vec![];
            let cepstrum_cutoff_samples = (quefrency_cutoff * input.sample_rate as f32 / 1000.0) as usize;
            let mut envelope_shifter = EnvelopeShifter::new(spectrum.len(), cepstrum_cutoff_samples, 1.0);
            envelope_shifter.compute_envelope(&spectrum, &mut envelope);

            save_spectrum_plot_svg(&spectrum, &envelope, input.sample_rate, fft_size, &output_path)
                .with_context(|| format!("saving spectrum plot {} failed", output_path))?;
        }
    }

    Ok(())
}
