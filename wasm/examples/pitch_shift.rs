use anyhow::{Result, bail};
use argh::FromArgs;
use hound::{WavReader, WavSpec, WavWriter};
use plotters::prelude::*;
use realfft::RealFftPlanner;

use wasm_main_module::{PitchShifter, StretchMethod, StretchParams, generate_sine_wave};

struct WavContents {
    data: Vec<f32>,
    rate: u32,
}

fn load_wav(filename: &str) -> Result<WavContents> {
    let mut reader = WavReader::open(filename)?;
    let num_channels = reader.spec().channels;
    if num_channels != 1 {
        println!("Reading only first channel from {}", filename);
    }
    let mut data = vec![];
    for s in reader.samples::<f32>().step_by(num_channels as usize) {
        data.push(s?);
    }
    Ok(WavContents { data: data, rate: reader.spec().sample_rate })
}

fn save_wav(data: &[f32], sample_rate: u32, filename: &str) -> Result<()> {
    let spec = WavSpec { channels: 1, sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };

    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in data {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    println!("WAV file saved to: {}", filename);
    Ok(())
}

fn save_plot_data_svg(input: &[f32], output: &[f32], sample_rate: u32, filename: &str) -> Result<()> {
    const INTERVAL1: (f32, f32) = (0.0, 0.025);
    const INTERVAL2: (f32, f32) = (0.2, 0.225);

    let max_sample = input.len().min(output.len());
    let from_sample1 = ((sample_rate as f32 * INTERVAL1.0) as usize).min(max_sample);
    let to_sample1 = ((sample_rate as f32 * INTERVAL1.1) as usize).min(max_sample);
    let from_sample2 = ((sample_rate as f32 * INTERVAL2.0) as usize).min(max_sample);
    let to_sample2 = ((sample_rate as f32 * INTERVAL2.1) as usize).min(max_sample);

    let root = SVGBackend::new(filename, (1200, 1200)).into_drawing_area();
    root.fill(&WHITE)?;
    let (top_area, bottom_area) = root.split_vertically(600);

    let mut top_chart = ChartBuilder::on(&top_area)
        .caption(format!("{}-{} s", INTERVAL1.0, INTERVAL1.1), ("sans-serif", 40))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(INTERVAL1.0..INTERVAL1.1, -1.2f32..1.2f32)?;
    top_chart.configure_mesh().label_style(("sans-serif", 25)).draw()?;
    top_chart
        .draw_series(LineSeries::new(
            (from_sample1..to_sample1).into_iter().map(|i| (i as f32 / sample_rate as f32, input[i])),
            &BLUE,
        ))?
        .label("Input");
    top_chart
        .draw_series(LineSeries::new(
            (from_sample1..to_sample1).into_iter().map(|i| (i as f32 / sample_rate as f32, output[i])),
            &RED,
        ))?
        .label("Output");

    // Create second plot (bottom) for interval 2
    let mut bottom_chart = ChartBuilder::on(&bottom_area)
        .caption(format!("{}-{} s", INTERVAL2.0, INTERVAL2.1), ("sans-serif", 40))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(INTERVAL2.0..INTERVAL2.1, -1.2f32..1.2f32)?;

    bottom_chart.configure_mesh().label_style(("sans-serif", 25)).draw()?;
    bottom_chart
        .draw_series(LineSeries::new(
            (from_sample2..to_sample2).into_iter().map(|i| (i as f32 / sample_rate as f32, input[i])),
            &BLUE,
        ))?
        .label("Input");
    bottom_chart
        .draw_series(LineSeries::new(
            (from_sample2..to_sample2).into_iter().map(|i| (i as f32 / sample_rate as f32, output[i])),
            &RED,
        ))?
        .label("Output");

    root.present()?;
    println!("Plot SVG saved to: {}", filename);
    Ok(())
}

fn compute_dominant_frequency(signal: &[f32], sample_rate: u32) -> Result<f32> {
    let n = signal.len();
    let fft_size = n.next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(fft_size);

    let mut input = vec![0.0; fft_size];
    input[..n].copy_from_slice(signal);
    let mut freq = r2c.make_output_vec();
    r2c.process(&mut input, &mut freq)?;

    let mut max_magn = 0.0;
    let mut max_bin = 0;
    for i in 0..freq.len() {
        let mag = freq[i].norm();
        if mag > max_magn {
            max_magn = mag;
            max_bin = i;
        }
    }

    Ok(max_bin as f32 * sample_rate as f32 / fft_size as f32)
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
    PitchShift(PitchShift),
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

    /// output WAV file path
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    output: String,
}

/// Apply pitch shift to a sine wave
#[derive(FromArgs)]
#[argh(subcommand, name = "pitch-shift")]
struct PitchShift {
    /// input WAV file
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    input: String,

    /// pitch shift factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
    #[argh(option, default = "1.0")]
    pitch_shift: f32,

    /// time stretch factor (e.g., 2.0 = two times longer, 0.5 = two times shorter)
    #[argh(option, default = "1.0")]
    time_stretch: f32,

    /// pitch shifting method
    #[argh(option, default = "\"basic\".to_string()")]
    method: String,

    /// overlap ratio for pitch shifting
    #[argh(option, default = "8")]
    overlap: u32,

    /// FFT size for pitch shifting
    #[argh(option, default = "4096")]
    fft_size: usize,

    /// output WAV file for shifted audio
    #[argh(option, default = "\"shifted.wav\".to_string()")]
    output: String,

    /// save plot comparison SVG
    #[argh(switch)]
    plot: bool,
}

fn main() -> Result<()> {
    let cli: Cli = argh::from_env();

    match cli.command {
        Commands::Generate(Generate { frequency, sample_rate, duration, output }) => {
            println!("Generating sine wave: {} Hz, {} samples/sec, {} seconds", frequency, sample_rate, duration);
            let sine_wave = generate_sine_wave(frequency, sample_rate, duration);
            println!("Generated {} samples", sine_wave.len());

            save_wav(&sine_wave, sample_rate as u32, &output)?;
        }

        Commands::PitchShift(PitchShift {
            input: input_path,
            pitch_shift,
            time_stretch,
            method,
            output,
            plot,
            overlap,
            fft_size,
        }) => {
            let input = load_wav(&input_path)?;
            println!(
                "Pitch shifting {}: {} Hz, pitch shift: {}, time stretch: {}, method: {:?}",
                input_path, input.rate, pitch_shift, time_stretch, method
            );

            let mut params = StretchParams::new(input.rate, pitch_shift, time_stretch);
            if method == "basic" {
                params.method = StretchMethod::Basic;
            } else if method == "phase-gradient" {
                params.method = StretchMethod::PhaseGradient;
            } else {
                bail!("Unknown method {}", method);
            }
            params.overlap = overlap;
            params.fft_size = fft_size;
            let mut shifter = PitchShifter::new(&params);

            // Process all input
            let mut output_data = shifter.process(&input.data);
            output_data.append(&mut shifter.finish());
            println!("Output generated: {} samples", output_data.len());

            save_wav(&output_data, input.rate, &output)?;

            let input_freq = compute_dominant_frequency(&input.data, input.rate)?;
            println!("Input dominant frequency: {:.2} Hz", input_freq);
            let output_freq = compute_dominant_frequency(&output_data, input.rate)?;
            println!("Output dominant frequency: {:.2} Hz", output_freq);

            if plot {
                let plot_filename = output.replace(".wav", "_plot.svg");
                save_plot_data_svg(&input.data, &output_data, input.rate, &plot_filename)?;
            }
        }
    }

    Ok(())
}
