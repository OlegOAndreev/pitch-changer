use anyhow::{bail, Context, Result};
use argh::FromArgs;
use hound::{WavReader, WavSpec, WavWriter};
use plotters::prelude::*;
use realfft::RealFftPlanner;

use wasm_main_module::{generate_sine_wave, StretchParams, TimeStretcher, WindowType};

fn parse_window_type(s: &str) -> Result<WindowType> {
    match s.to_lowercase().as_str() {
        "hann" => Ok(WindowType::Hann),
        "sqrt-hann" => Ok(WindowType::SqrtHann),
        "sqrt-blackman" => Ok(WindowType::SqrtBlackman),
        _ => bail!("Unknown window type '{}'. Supported: hann, sqrt-hann, sqrt-blackman", s),
    }
}

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
    match reader.spec().sample_format {
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>().step_by(num_channels as usize) {
                data.push(s?);
            }
        }
        hound::SampleFormat::Int => {
            if reader.spec().bits_per_sample == 16 {
                for s in reader.samples::<i16>().step_by(num_channels as usize) {
                    data.push(s? as f32 / 65536.0);
                }
            }
        }
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
    const INTERVAL_LENGTH: f32 = 0.05;

    let min_length = input.len().min(output.len());
    let min_duration = min_length as f32 / sample_rate as f32;
    let output_length = output.len() as f32 / sample_rate as f32;

    let interval1_start = 0.0;
    let interval1_end = interval1_start + INTERVAL_LENGTH;
    let interval2_start = min_duration * 0.5;
    let interval2_end = interval2_start + INTERVAL_LENGTH;
    let interval3_start = output_length - INTERVAL_LENGTH * 2.0;
    let interval3_end = output_length - INTERVAL_LENGTH;

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
        .caption(format!("Beginning: {:.3}-{:.3} s", interval1_start, interval1_end), ("sans-serif", 40))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(interval1_start..interval1_end, -1.2f32..1.2f32)?;
    top_chart.configure_mesh().label_style(("sans-serif", 25)).draw()?;
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

    middle_chart.configure_mesh().label_style(("sans-serif", 25)).draw()?;
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

    bottom_chart.configure_mesh().label_style(("sans-serif", 25)).draw()?;
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
    TimeStretch(TimeStretch),
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

/// Apply time stretch to input
#[derive(FromArgs)]
#[argh(subcommand, name = "time-stretch")]
struct TimeStretch {
    /// input WAV file
    #[argh(option, default = "\"sine_wave.wav\".to_string()")]
    input: String,

    /// time stretch factor (e.g., 2.0 = one octave up, 0.5 = one octave down)
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

fn main() -> Result<()> {
    let cli: Cli = argh::from_env();

    match cli.command {
        Commands::Generate(Generate { frequency, sample_rate, duration, output }) => {
            println!("Generating sine wave: {} Hz, {} samples/sec, {} seconds", frequency, sample_rate, duration);
            let sine_wave = generate_sine_wave(frequency, sample_rate, 1.0, duration);
            println!("Generated {} samples", sine_wave.len());

            save_wav(&sine_wave, sample_rate as u32, &output)?;
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
            let input = load_wav(&input_path).with_context(|| format!("loading {} failed", input_path))?;
            println!("Time stretching {}: {} Hz, stretch: {}", input_path, input.rate, stretch);

            let window_type = parse_window_type(&window)?;
            let mut params = StretchParams::new(input.rate, stretch);
            params.overlap = overlap;
            params.fft_size = fft_size;
            params.window_type = window_type;
            let mut shifter = TimeStretcher::new(&params)?;

            // Process all input
            let mut output_data = shifter.process(&input.data);
            output_data.append(&mut shifter.finish());
            println!("Output generated: {} samples", output_data.len());

            save_wav(&output_data, input.rate, &output_path)
                .with_context(|| format!("saving {} failed", output_path))?;

            let input_freq = compute_dominant_frequency(&input.data, input.rate)?;
            println!("Input dominant frequency: {:.2} Hz", input_freq);
            let output_freq = compute_dominant_frequency(&output_data, input.rate)?;
            println!("Output dominant frequency: {:.2} Hz", output_freq);

            if plot {
                let plot_filename = output_path.replace(".wav", "_plot.svg");
                save_plot_data_svg(&input.data, &output_data, input.rate, &plot_filename)
                    .with_context(|| format!("saving plot {} failed", plot_filename))?;
            }
        }
    }

    Ok(())
}
