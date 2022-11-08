mod logic;
use crate::logic::{get_tsv_paths, process_file};
use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde_json::{from_reader, to_writer};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

const PATTERNS: [&str; 2] = [
    "../data/NSDUH_Versions/*/*/*Tab.tsv",
    "../data/NSDUH_Versions/*/*/*/*Tab.tsv",
];

fn get_lengths(tsv_paths: Vec<PathBuf>) -> Vec<usize> {
    let m = MultiProgress::new();
    let pbs: Vec<_> = tsv_paths
        .iter()
        .map(|path| {
            let filename = path.file_name().unwrap().to_str().unwrap();
            let msg = format!("Counting lines in {}", filename);
            m.add(ProgressBar::new_spinner().with_message(msg))
        })
        .collect();

    let lengths = tsv_paths
        .par_iter()
        .zip(pbs)
        .map(|(path, pb)| {
            BufReader::new(File::open(path.clone()).unwrap())
                .lines()
                .progress_with(pb)
                .count()
        })
        .collect();
    m.clear().unwrap();
    lengths
}

fn main() -> Result<()> {
    let tsv_paths: Vec<_> = get_tsv_paths(&PATTERNS)?;
    assert_eq!(tsv_paths.len(), 5);

    let lengths = match File::open("lengths.json") {
        Ok(rdr) => from_reader(rdr)?,
        Err(_) => {
            let lengths = get_lengths(tsv_paths.clone());
            to_writer(File::create("lengths.json")?, &lengths)?;
            lengths
        }
    };

    let m = MultiProgress::new();
    let sty = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")?
        .progress_chars("##-");
    let pbs: Vec<_> = lengths
        .iter()
        .map(|&n| m.add(ProgressBar::new(n as u64).with_style(sty.clone())))
        .collect();

    (tsv_paths, pbs)
        .into_par_iter()
        .map(process_file)
        .collect::<Result<_>>()?;

    m.clear()?;
    Ok(())
}
