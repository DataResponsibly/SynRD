mod col_type;
use col_type::ColType;
use itertools::Itertools;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};

use anyhow::Result;
use glob::glob;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use rayon::iter::{ParallelBridge, ParallelIterator};

const PATTERNS: [&str; 2] = [
    "../data/NSDUH_Versions/*/*/*.tsv",
    "../data/NSDUH_Versions/*/*/*/*.tsv",
];

fn main() -> Result<()> {
    let tsv_paths: Vec<_> = PATTERNS
        .iter()
        .flat_map(|pattern| glob(pattern).unwrap())
        .map(|path| path.unwrap().display().to_string())
        .collect();
    assert_eq!(tsv_paths.len(), 5);

    for path in tsv_paths.into_iter().progress() {
        let n = BufReader::new(File::open(path.clone())?).lines().count();
        let buf = BufReader::new(File::open(path.clone())?);
        let mut lines = buf.lines();
        let header: Vec<String> = lines.next().unwrap()?.split('\t').map_into().collect();

        // let mut writers = Vec::new();
        // let mut senders =

        // let year_ind = header.iter().position(|name| name == "YEAR");
        let dtypes: Vec<ColType> = lines
            .par_bridge()
            .progress_count(n as u64)
            .fold(
                || vec![ColType::Int; n],
                |mut a, line| {
                    a.iter_mut()
                        .zip(line.unwrap().split('\t').map_into())
                        .for_each(|(a, b)| *a = a.combine(b));
                    a
                },
            )
            .reduce(
                || vec![ColType::Int; n],
                |mut a, b| {
                    a.iter_mut()
                        .zip(b.into_iter())
                        .for_each(|(a, b)| *a = a.combine(b));
                    a
                },
            );

        let dtypes_path = path.replace(".tsv", ".txt");
        let dtypes_dict = dtypes
            .iter()
            .zip(header.iter())
            .map(|(v, h)| format!("\"{h}\": {v},\n"))
            .collect::<String>();
        fs::write(dtypes_path, format!("{{\n{dtypes_dict}}}\n"))?;
    }
    Ok(())
}
