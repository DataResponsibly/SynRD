mod col_type;
use glob::{glob, PatternError};
use indicatif::{ProgressBar, ProgressIterator};
use itertools::Itertools;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use self::col_type::ColType;

pub fn open_file_with_header(year: &str, path: &Path, header: &[String]) -> BufWriter<File> {
    let path = path.parent().unwrap().join(&format!("{year}.tsv"));
    let mut f = BufWriter::new(File::create(path).unwrap());
    writeln!(f, "{}", header.join("\t")).unwrap();
    f
}

pub fn get_tsv_paths(patterns: &[&str]) -> Result<Vec<PathBuf>, PatternError> {
    patterns
        .iter()
        .copied()
        .map(glob)
        .flatten_ok()
        .flatten_ok()
        .collect()
}

pub fn process_file((path, pb): (PathBuf, ProgressBar)) -> anyhow::Result<()> {
    let mut lines = BufReader::new(File::open(path.clone())?).lines();
    let header: Vec<String> = lines.next().unwrap()?.split('\t').map_into().collect();

    let mut datasets_per_year = HashMap::new();

    let year_ind = header.iter().position(|name| name == "YEAR").unwrap();

    pb.reset();
    let dtypes: Vec<ColType> = lines
        .progress_with(pb)
        .map(|line| line.unwrap())
        .enumerate()
        .map(|(n, line)| {
            let token: Vec<_> = line.split('\t').collect();
            let year = token[year_ind];
            if ColType::from(year) != ColType::Int {
                println!("Line {n} has {year} value for YEAR, which cannot be parsed");
            }
            let token: Vec<ColType> = token.into_iter().map_into().collect();

            let f = datasets_per_year
                .entry(year.to_string())
                .or_insert_with(|| open_file_with_header(year, &path, &header));
            writeln!(f, "{}", line).unwrap();
            token
        })
        .fold(vec![ColType::Int; header.len()], |mut acc, v| {
            acc.iter_mut().zip(v).for_each(|(a, b)| *a = a.combine(&b));
            acc
        });

    let dtypes_path = path.parent().unwrap().join("schema.txt");
    if header.len() != dtypes.len() {
        eprintln!(
            "Header len != dtypes len for {}: {} vs {}",
            path.display(),
            header.len(),
            dtypes.len()
        );
    }
    let dtypes_dict: String = header
        .iter()
        .zip(dtypes.iter())
        .map(|(header, dtype)| format!("\"{header}\": {dtype},\n"))
        .collect();
    fs::write(dtypes_path, format!("{{\n{dtypes_dict}}}\n"))?;
    Ok(())
}
