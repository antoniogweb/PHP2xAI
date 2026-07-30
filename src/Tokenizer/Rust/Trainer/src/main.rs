use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
use tokenizers::tokenizer::{
    Decoder, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerImpl,
};
use tokenizers::{AddedToken, TokenizerBuilder};

const DEFAULT_VOCAB_SIZE: usize = 30_000;
const DEFAULT_MIN_FREQUENCY: u64 = 2;

const PAD_TOKEN: &str = "[PAD]";
const UNK_TOKEN: &str = "[UNK]";
const BOS_TOKEN: &str = "[BOS]";
const EOS_TOKEN: &str = "[EOS]";

#[derive(Debug)]
struct Config {
    input_path: PathBuf,
    output_path: PathBuf,
    vocab_size: usize,
    min_frequency: u64,
    pretty: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Tokenizer training failed: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = parse_arguments()?;

    validate_input_file(&config.input_path)?;
    create_output_directory(&config.output_path)?;

    println!("PHP2xAI tokenizer trainer");
    println!("Input: {}", config.input_path.display());
    println!("Output: {}", config.output_path.display());
    println!("Vocabulary size: {}", config.vocab_size);
    println!("Minimum frequency: {}", config.min_frequency);

    /*
     * The order of special_tokens determines their IDs:
     *
     * 0 -> [PAD]
     * 1 -> [UNK]
     * 2 -> [BOS]
     * 3 -> [EOS]
     */
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(config.vocab_size)
        .min_frequency(config.min_frequency)
        .special_tokens(vec![
            AddedToken::from(PAD_TOKEN.to_string(), true),
            AddedToken::from(UNK_TOKEN.to_string(), true),
            AddedToken::from(BOS_TOKEN.to_string(), true),
            AddedToken::from(EOS_TOKEN.to_string(), true),
        ])
        .build();

    let model = BPE::builder().unk_token(UNK_TOKEN.to_string()).build()?;

    /*
     * ByteLevel makes the tokenizer robust with arbitrary UTF-8 text:
     * unknown characters can be represented through byte-level tokens.
     */
    let byte_level_pre_tokenizer = ByteLevelPreTokenizer::default();
    let byte_level_processor = ByteLevelProcessor::default();
    let byte_level_decoder = ByteLevelDecoder::default();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(model)
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(byte_level_pre_tokenizer))
        .with_post_processor(Some(byte_level_processor))
        .with_decoder(Some(byte_level_decoder))
        .build()?;

    tokenizer.train_from_files(
        &mut trainer,
        vec![config.input_path.to_string_lossy().into_owned()],
    )?;

    tokenizer.save(&config.output_path, config.pretty)?;
    let metadata_path = write_tokenizer_metadata(
        &tokenizer,
        &config.input_path,
        &config.output_path,
        config.pretty,
    )?;

    println!("Special token IDs:");

    for token in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] {
        let id = tokenizer
            .token_to_id(token)
            .ok_or_else(|| format!("Special token was not created: {token}"))?;

        println!("  {token} = {id}");
    }

    println!(
        "Tokenizer successfully written to {}",
        config.output_path.display()
    );
    println!(
        "Tokenizer metadata successfully written to {}",
        metadata_path.display()
    );

    Ok(())
}

fn write_tokenizer_metadata<M, N, PT, PP, D>(
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    corpus_path: &Path,
    tokenizer_path: &Path,
    pretty: bool,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>>
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    let file = File::open(corpus_path)?;
    let reader = BufReader::new(file);
    let mut lengths = Vec::new();

    for line in reader.lines() {
        let text = line?;

        if text.trim().is_empty() {
            continue;
        }

        let encoding = tokenizer.encode(text, true)?;
        lengths.push(encoding.len());
    }

    lengths.sort_unstable();

    let samples = lengths.len();
    let max_length = lengths.last().copied().unwrap_or(0);
    let average_length = if samples == 0 {
        0.0
    } else {
        lengths.iter().sum::<usize>() as f64 / samples as f64
    };

    let metadata = serde_json::json!({
        "samples": samples,
        "max_length": max_length,
        "average_length": average_length,
        "p50": percentile(&lengths, 50.0),
        "p95": percentile(&lengths, 95.0),
        "p99": percentile(&lengths, 99.0),
    });

    let metadata_path = tokenizer_path.with_file_name("tokenizer.meta.json");
    let mut output = File::create(&metadata_path)?;

    if pretty {
        serde_json::to_writer_pretty(&mut output, &metadata)?;
    } else {
        serde_json::to_writer(&mut output, &metadata)?;
    }

    output.write_all(b"\n")?;

    Ok(metadata_path)
}

fn percentile(sorted_lengths: &[usize], percentile: f64) -> usize {
    if sorted_lengths.is_empty() {
        return 0;
    }

    let rank = (percentile / 100.0 * sorted_lengths.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted_lengths.len() - 1);

    sorted_lengths[index]
}

fn parse_arguments() -> Result<Config, String> {
    let mut args = env::args().skip(1);

    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut vocab_size = DEFAULT_VOCAB_SIZE;
    let mut min_frequency = DEFAULT_MIN_FREQUENCY;
    let mut pretty = false;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--input" | "-i" => {
                input_path = Some(PathBuf::from(next_argument(&mut args, "--input")?));
            }

            "--output" | "-o" => {
                output_path = Some(PathBuf::from(next_argument(&mut args, "--output")?));
            }

            "--vocab-size" | "-v" => {
                let value = next_argument(&mut args, "--vocab-size")?;

                vocab_size = value
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid vocabulary size: {value}"))?;
            }

            "--min-frequency" | "-f" => {
                let value = next_argument(&mut args, "--min-frequency")?;

                min_frequency = value
                    .parse::<u64>()
                    .map_err(|_| format!("Invalid minimum frequency: {value}"))?;
            }

            "--pretty" => {
                pretty = true;
            }

            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }

            unknown => {
                return Err(format!(
                    "Unknown argument: {unknown}\n\
                     Run with --help to see the available arguments."
                ));
            }
        }
    }

    let input_path = input_path.ok_or_else(|| "Missing required argument: --input".to_string())?;

    let output_path =
        output_path.ok_or_else(|| "Missing required argument: --output".to_string())?;

    if vocab_size < 4 {
        return Err("Vocabulary size must be at least 4 for the special tokens.".to_string());
    }

    Ok(Config {
        input_path,
        output_path,
        vocab_size,
        min_frequency,
        pretty,
    })
}

fn next_argument(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("Missing value after {option}"))
}

fn validate_input_file(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("Input corpus does not exist: {}", path.display()));
    }

    if !path.is_file() {
        return Err(format!(
            "Input corpus is not a regular file: {}",
            path.display()
        ));
    }

    let metadata = fs::metadata(path).map_err(|error| {
        format!(
            "Cannot read input corpus metadata {}: {error}",
            path.display()
        )
    })?;

    if metadata.len() == 0 {
        return Err(format!("Input corpus is empty: {}", path.display()));
    }

    Ok(())
}

fn create_output_directory(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    if parent.as_os_str().is_empty() {
        return Ok(());
    }

    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Cannot create output directory {}: {error}",
            parent.display()
        )
    })
}

fn print_help() {
    println!(
        "\
php2xai-tokenizer-trainer

Trains a Hugging Face ByteLevel BPE tokenizer from a text corpus.

USAGE:
    php2xai-tokenizer-trainer [OPTIONS]

REQUIRED OPTIONS:
    -i, --input <FILE>
        UTF-8 corpus file. Each line should contain one training sample.

    -o, --output <FILE>
        Destination tokenizer.json file.

OPTIONAL:
    -v, --vocab-size <NUMBER>
        Maximum vocabulary size.
        Default: {DEFAULT_VOCAB_SIZE}

    -f, --min-frequency <NUMBER>
        Minimum token-pair frequency.
        Default: {DEFAULT_MIN_FREQUENCY}

    --pretty
        Save formatted JSON instead of compact JSON.

    -h, --help
        Show this help.
"
    );
}
