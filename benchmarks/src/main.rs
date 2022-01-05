use std::{
    collections::HashMap,
    fmt::Write,
    fs::File,
    io::BufReader,
    io::Cursor,
    marker::PhantomData,
    time::{Duration, Instant},
};

use arcode::{
    bitbit::{BitReader, BitWriter, MSB},
    decode::decoder::ArithmeticDecoder,
    encode::encoder::ArithmeticEncoder,
    util::source_model_builder::{EOFKind, SourceModelBuilder},
};
use compressed_dynamic_word_embeddings::embedding_file::EmbeddingFile;
use constriction::{
    backends::{BoundedReadWords, WriteWords},
    stream::{
        model::{
            DecoderModel, EncoderModel, LookupDecoderModel, NonContiguousCategoricalDecoderModel,
            NonContiguousCategoricalEncoderModel, NonContiguousSymbolTable,
        },
        queue::RangeEncoder,
        stack::AnsCoder,
        Code, Decode, Encode, IntoDecoder,
    },
    BitArray, NonZeroBitArray, Stack,
};
use num_traits::AsPrimitive;
use rand::seq::SliceRandom;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "evalcompress",
    about = "Evaluate the bitrates and run times of various compression methods."
)]
struct Args {
    /// report bitrates.
    #[structopt(long)]
    bitrates: bool,

    /// Measure and report runtimes.
    #[structopt(long)]
    runtimes: bool,

    /// Number of times that the runtime for each data point will be measured.
    #[structopt(long, short = "c", default_value = "5")]
    runtimes_count: u32,

    /// Number of seconds to pause between encoding steps.
    #[structopt(long, short = "p", default_value = "0")]
    pause: u32,
}

fn load_tlfc() -> Vec<Vec<i16>> {
    println!();
    println!("# Loading and decoding tlfc file ...");
    let file = BufReader::new(File::open("tlfc.dwe").unwrap());
    let file = EmbeddingFile::from_reader(file).unwrap();
    let symbols_per_timestep = (file.header().vocab_size * file.header().embedding_dim) as usize;
    println!("symbols_per_test = {}", symbols_per_timestep);

    (0..file.header().num_timesteps)
        .map(|t| {
            let timestep = file.timestep(t).unwrap();
            let (mut decoder, model) = timestep.into_inner();
            decoder
                .decode_iid_symbols(symbols_per_timestep, model)
                .map(Result::unwrap)
                .collect()
        })
        .collect()
}

fn histogram(data: &[i16]) -> (Vec<(i16, u32)>, f64) {
    let mut counts = HashMap::new();
    for &i in data {
        *counts.entry(i).or_default() += 1;
    }

    // Sort entries to make floating point operations deterministic.
    let mut counts_sorted = counts.iter().map(|(&s, &c)| (s, c)).collect::<Vec<_>>();
    counts_sorted.sort_unstable();

    let entropy = (data.len() as f64).log2()
        - counts_sorted
            .iter()
            .map(|&(_, c)| c as f64 * (c as f64).log2())
            .sum::<f64>()
            / data.len() as f64;

    (counts_sorted, entropy)
}

trait HasBitrate {
    fn bitrate(&self) -> usize;
}

impl<Word, State, Backend> HasBitrate for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: BoundedReadWords<Word, Stack>,
{
    fn bitrate(&self) -> usize {
        self.num_valid_bits()
    }
}

impl<Word, State, Backend> HasBitrate for RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: AsRef<[Word]> + WriteWords<Word>,
{
    fn bitrate(&self) -> usize {
        self.num_bits()
    }
}

fn cross_entropy<M, const PRECISION: usize>(
    symbols: &[i16],
    counts: &[f64],
    total_count: usize,
    model: M,
) -> f64
where
    M: EncoderModel<PRECISION, Symbol = i16>,
    M::Probability: AsPrimitive<f64>,
{
    PRECISION as f64
        - symbols
            .iter()
            .zip(counts)
            .map(|(&symbol, &count)| {
                let probability = model
                    .left_cumulative_and_probability(symbol)
                    .unwrap()
                    .1
                    .get();
                count as f64 * probability.as_().log2()
            })
            .sum::<f64>()
            / total_count as f64
}

struct TimestepStats {
    symbols: Vec<i16>,
    counts: Vec<u32>,
    counts_f64: Vec<f64>,
    entropy: f64,
}

trait DecoderModelBuilder<Encoder, const PRECISION: usize>
where
    Encoder: Code,
{
    type DecoderModel: DecoderModel<PRECISION, Probability = Encoder::Word, Symbol = i16>;

    fn build(symbols: &[i16], counts: &[f64]) -> Self::DecoderModel;
}

struct NonLookup<Encoder, const PRECISION: usize>(PhantomData<Encoder>);

impl<Encoder, const PRECISION: usize> DecoderModelBuilder<Encoder, PRECISION>
    for NonLookup<Encoder, PRECISION>
where
    Encoder: Encode<PRECISION>
        + Into<Vec<<Encoder as Code>::Word>>
        + Default
        + HasBitrate
        + IntoDecoder<PRECISION>,
    <Encoder as Code>::Word: AsPrimitive<usize>
        + AsPrimitive<f64>
        + std::convert::Into<f64>
        + AsPrimitive<<Encoder as Code>::Word>,
    f64: AsPrimitive<<Encoder as Code>::Word>,
    usize: AsPrimitive<<Encoder as Code>::Word>,
{
    #[allow(clippy::type_complexity)]
    type DecoderModel = NonContiguousCategoricalDecoderModel<
        i16,
        <Encoder as Code>::Word,
        Vec<(<Encoder as Code>::Word, i16)>,
        PRECISION,
    >;

    fn build(symbols: &[i16], counts: &[f64]) -> Self::DecoderModel {
        Self::DecoderModel::from_symbols_and_floating_point_probabilities(symbols, counts).unwrap()
    }
}

struct Lookup<Encoder, const PRECISION: usize>(PhantomData<Encoder>);

impl<Encoder, const PRECISION: usize> DecoderModelBuilder<Encoder, PRECISION>
    for Lookup<Encoder, PRECISION>
where
    Encoder: Encode<PRECISION>
        + Into<Vec<<Encoder as Code>::Word>>
        + Default
        + HasBitrate
        + IntoDecoder<PRECISION>,
    <Encoder as Code>::Word: AsPrimitive<usize>
        + AsPrimitive<f64>
        + std::convert::Into<f64>
        + AsPrimitive<<Encoder as Code>::Word>,
    f64: AsPrimitive<<Encoder as Code>::Word>,
    usize: AsPrimitive<<Encoder as Code>::Word>,
    <Encoder as Code>::Word: Into<usize>,
{
    #[allow(clippy::type_complexity)]
    type DecoderModel = LookupDecoderModel<
        i16,
        <Encoder as Code>::Word,
        NonContiguousSymbolTable<Vec<(<Encoder as Code>::Word, i16)>>,
        Box<[<Encoder as Code>::Word]>,
        PRECISION,
    >;

    fn build(symbols: &[i16], counts: &[f64]) -> Self::DecoderModel {
        Self::DecoderModel::from_symbols_and_floating_point_probabilities(symbols, counts).unwrap()
    }
}

fn run_constriction<Encoder, Builder, const PRECISION: usize>(
    label: &str,
    data: &[Vec<i16>],
    stats: &[TimestepStats],
    measure_bitrates: bool,
    measure_runtimes: bool,
    pause: Duration,
) where
    Encoder: Encode<PRECISION>
        + Into<Vec<<Encoder as Code>::Word>>
        + Default
        + HasBitrate
        + IntoDecoder<PRECISION>
        + Clone,
    <Encoder as Code>::Word: AsPrimitive<usize>
        + AsPrimitive<f64>
        + std::convert::Into<f64>
        + AsPrimitive<<Encoder as Code>::Word>,
    Builder: DecoderModelBuilder<Encoder, PRECISION>,
    f64: AsPrimitive<<Encoder as Code>::Word>,
    usize: AsPrimitive<<Encoder as Code>::Word>,
{
    std::thread::sleep(pause);

    let mut output = String::new();
    writeln!(output).expect("Writing to String can't fail.");
    if measure_bitrates {
        writeln!(
            output,
            "bitrates['{}'] = np.empty(({}, 3))",
            label,
            data.len()
        )
        .expect("Writing to String can't fail.");
    }
    if measure_runtimes {
        writeln!(output, "if '{}' not in runtimes:", label).expect("Writing to String can't fail.");
        writeln!(
            output,
            "    runtimes['{}'] = [[] for _ in range({})]",
            label,
            data.len()
        )
        .expect("Writing to String can't fail.");
    }

    let mut all_timesteps = (0..data.len()).collect::<Vec<_>>();
    if measure_runtimes {
        all_timesteps.shuffle(&mut rand::thread_rng());
    }

    let num_runs = if measure_runtimes { 2 } else { 1 };

    for t in all_timesteps {
        let data = &data[t][..];
        let stats = &stats[t];

        let encoder_model = NonContiguousCategoricalEncoderModel::<
            i16,
            <Encoder as Code>::Word,
            PRECISION,
        >::from_symbols_and_floating_point_probabilities(
            stats.symbols.iter().cloned(),
            &stats.counts_f64,
        )
        .unwrap();

        let decoder_model = Builder::build(&stats.symbols, &stats.counts_f64);

        let cross_entropy = cross_entropy(
            &stats.symbols,
            &stats.counts_f64,
            data.len(),
            &encoder_model,
        );

        let mut i = 0;
        let (bitrate, duration_encode, encoder) = loop {
            let mut encoder = Encoder::default();
            let start_time = Instant::now();
            encoder.encode_iid_symbols(data, &encoder_model).unwrap();
            let end_time = Instant::now();
            let duration_encode = (end_time - start_time).as_nanos();
            let bitrate = encoder.bitrate();
            i += 1;
            if i == num_runs {
                break (bitrate, duration_encode, encoder);
            }
        };

        let mut checksum = 0;
        let mut duration_decode = 0;
        if measure_runtimes {
            for i in 0..2 {
                let mut decoder = encoder.clone().into_decoder();
                let start_time = Instant::now();
                for _ in 0..data.len() {
                    checksum ^= decoder.decode_symbol(&decoder_model).unwrap();
                }
                let end_time = Instant::now();
                duration_decode = (end_time - start_time).as_nanos();
                checksum = checksum.wrapping_add(17 + i).wrapping_mul(checksum);
            }
        }

        if measure_bitrates {
            writeln!(
                output,
                "bitrates['{}'][{}] = np.array(({}, {}, {}))",
                label,
                t,
                t,
                cross_entropy,
                bitrate as f64 / data.len() as f64
            )
            .expect("Writing to String can't fail.");
        }

        if measure_runtimes {
            writeln!(
                output,
                "runtimes['{}'][{}].append(({}, {}, {}))",
                label,
                t,
                duration_encode as f64 / data.len() as f64,
                duration_decode as f64 / data.len() as f64,
                checksum
            )
            .expect("Writing to String can't fail.");
        }
    }

    print!("{}", output);
}

macro_rules! run {
    ($encoder:ty, $precision:expr, $builder:ident, $data:expr, $stats:expr, $report_bitrates:expr, $measure_runtimes:expr, $pause:expr) => {
        run_constriction::<$encoder, $builder<$encoder, $precision>, $precision>(
            stringify!($encoder, $precision, $builder),
            $data,
            $stats,
            $report_bitrates,
            $measure_runtimes,
            $pause,
        )
    };
}

fn run_arcode(
    precision: u64,
    data: &[Vec<i16>],
    stats: &[TimestepStats],
    measure_bitrates: bool,
    measure_runtimes: bool,
    pause: Duration,
) {
    std::thread::sleep(pause);

    let mut output = String::new();
    writeln!(output).expect("Writing to String can't fail.");
    if measure_bitrates {
        writeln!(
            output,
            "bitrates['arcode {}'] = np.empty(({}, 3))",
            precision,
            data.len()
        )
        .expect("Writing to String can't fail.");
    }
    if measure_runtimes {
        writeln!(output, "if 'arcode {}' not in runtimes:", precision)
            .expect("Writing to String can't fail.");
        writeln!(
            output,
            "    runtimes['arcode {}'] = [[] for _ in range({})]",
            precision,
            data.len()
        )
        .expect("Writing to String can't fail.");
    }

    let mut all_timesteps = (0..data.len()).collect::<Vec<_>>();
    if measure_runtimes {
        all_timesteps.shuffle(&mut rand::thread_rng());
    }

    let num_runs = if measure_runtimes { 2 } else { 1 };

    for t in all_timesteps {
        let data = &data[t][..];
        let stats = &stats[t];

        let symbol_table = stats
            .symbols
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as u32))
            .collect::<HashMap<_, _>>();

        let model = SourceModelBuilder::new()
            .counts(stats.counts.clone())
            .eof(EOFKind::None)
            .build();

        let mut checksum = 0;
        let mut duration_encode = 0;
        let mut duration_decode = 0;
        let mut bitrate = 0;

        for i in 0..num_runs {
            let compressed = Cursor::new(vec![]);
            let mut compressed_writer = BitWriter::new(compressed);
            let mut encoder = ArithmeticEncoder::new(precision);

            let start_time = Instant::now();
            for &symbol in data {
                encoder
                    .encode(
                        *symbol_table.get(&symbol).unwrap(),
                        &model,
                        &mut compressed_writer,
                    )
                    .unwrap();
            }
            let end_time = Instant::now();
            duration_encode = (end_time - start_time).as_nanos();
            encoder.finish_encode(&mut compressed_writer).unwrap();
            compressed_writer.pad_to_byte().unwrap();
            bitrate = compressed_writer.get_ref().get_ref().len() * 8;

            if measure_runtimes {
                let inverse_symbol_table = stats
                    .symbols
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i as u32, s))
                    .collect::<HashMap<_, _>>();

                let compressed: &[u8] = compressed_writer.get_ref().get_ref().as_ref();
                let mut input_reader = BitReader::<_, MSB>::new(compressed);
                let mut decoder = ArithmeticDecoder::new(precision);
                let start_time = Instant::now();
                for _ in 0..data.len() {
                    checksum ^= inverse_symbol_table
                        .get(&decoder.decode(&model, &mut input_reader).unwrap())
                        .unwrap();
                }
                let end_time = Instant::now();
                duration_decode = (end_time - start_time).as_nanos();
                checksum = checksum.wrapping_add(17 + i).wrapping_mul(checksum);
            }
        }

        if measure_bitrates {
            writeln!(
                output,
                "bitrates['arcode {}'][{}] = np.array(({}, 0, {}))",
                precision,
                t,
                t,
                bitrate as f64 / data.len() as f64
            )
            .expect("Writing to String can't fail.");
        }
        if measure_runtimes {
            writeln!(
                output,
                "runtimes['arcode {}'][{}].append(({}, {}, {}))",
                precision,
                t,
                duration_encode as f64 / data.len() as f64,
                duration_decode as f64 / data.len() as f64,
                checksum
            )
            .expect("Writing to String can't fail.");
        }
    }

    print!("{}", output);
}

#[derive(Debug)]
enum ArgsError {
    NothingToDo,
}

fn main() -> Result<(), ArgsError> {
    let args = Args::from_args();

    #[cfg(debug_assertions)]
    {
        println!(
            "WARNING: this seems to be an unoptimized build. You can use it for debugging but\n\
            any reported runtimes are meaningless. To build or run and optimized (\"release\")\n\
            version, run `cargo build --release` or `cargo run --release`, respectively.\n"
        );
    }

    println!("import numpy as np");
    println!();
    println!("args = '{:?}'", args);
    println!("start_time = '{}'", chrono::Local::now());
    const GIT_VERSION: &str = git_version::git_version!();
    println!("git_commit = '{}'", GIT_VERSION);

    println!();
    println!("entropies_headers = ('time step', 'entropy per symbol')");
    if args.bitrates {
        println!("bitrates_headers = ('time step', 'cross entropy per symbol', 'bits per symbol')");
        println!("bitrates = {{}}");
    }
    if args.runtimes {
        println!(
            "runtimes_headers = ('encoding time per symbol (ns)', 'decoding time per symbol (ns)', 'checksum')"
        );
        println!("runtimes = {{}}");
    }

    let data = load_tlfc();

    println!("entropies = np.array([");
    let stats = data
        .iter()
        .enumerate()
        .map(|(t, data)| {
            let (counts, entropy) = histogram(data);
            let (symbols, counts_f64) = counts.iter().map(|&(s, c)| (s, c as f64)).unzip();
            let counts = counts.iter().map(|&(_, c)| c).collect();
            println!("    ({}, {}),", t, entropy);

            TimestepStats {
                symbols,
                counts,
                counts_f64,
                entropy,
            }
        })
        .collect::<Vec<_>>();
    println!(
        "]) # total_entropy_per_symbol = {}",
        stats.iter().map(|stats| stats.entropy).sum::<f64>() / stats.len() as f64
    );

    let p = Duration::from_secs(args.pause as u64);

    if args.runtimes {
        let b = args.bitrates;
        let mut tasks: Vec<Box<dyn Fn()>> = vec![
            Box::new(|| run!(AnsCoder<u32, u64>, 32, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u32, u64>, 32, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u32, u64>, 24, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u32, u64>, 24, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u32, u64>, 16, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u32, u64>, 16, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u64>, 16, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u64>, 16, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u64>, 12, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u64>, 12, Lookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u64>, 12, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u64>, 12, Lookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u32>, 16, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u32>, 16, Lookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u32>, 16, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u32>, 16, Lookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u32>, 12, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(AnsCoder<u16, u32>, 12, Lookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u32>, 12, NonLookup, &data, &stats, b, true, p)),
            Box::new(|| run!(RangeEncoder<u16, u32>, 12, Lookup, &data, &stats, b, true, p)),
            Box::new(|| run_arcode(63, &data, &stats, b, true, p)),
            Box::new(|| run_arcode(48, &data, &stats, b, true, p)),
            Box::new(|| run_arcode(32, &data, &stats, b, true, p)),
            Box::new(|| run_arcode(24, &data, &stats, b, true, p)),
        ];

        for i in 0..args.runtimes_count {
            eprintln!("# Round {} ...", i + 1);
            println!("\n# Round {} ...", i + 1);
            tasks.shuffle(&mut rand::thread_rng());
            for task in &tasks {
                (task)();
            }
        }
    } else if args.bitrates {
        rayon::scope(|s| {
            s.spawn(|_| run!(AnsCoder<u32, u64>, 32, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u32, u64>, 32, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(AnsCoder<u32, u64>, 24, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u32, u64>, 24, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(AnsCoder<u32, u64>, 16, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u32, u64>, 16, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(AnsCoder<u16, u64>, 16, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u16, u64>, 16, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(AnsCoder<u16, u64>, 12, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u16, u64>, 12, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(AnsCoder<u16, u32>, 16, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u16, u32>, 16, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(AnsCoder<u16, u32>, 12, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run!(RangeEncoder<u16, u32>, 12, NonLookup, &data, &stats, true, false, p));
            s.spawn(|_| run_arcode(63, &data, &stats, true, false, p));
            s.spawn(|_| run_arcode(48, &data, &stats, true, false, p));
            s.spawn(|_| run_arcode(32, &data, &stats, true, false, p));
            s.spawn(|_| run_arcode(24, &data, &stats, true, false, p));
        });
    } else {
        eprintln!("Nothing to do. Neither --bitrates nor --runtimes requested.");
        return Err(ArgsError::NothingToDo);
    }

    println!();
    println!("end_time = '{}'", chrono::Local::now());

    Ok(())
}
