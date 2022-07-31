mod col_type;
use anyhow::Result;
use col_type::ColType;
use glob::glob;
use indicatif::{MultiProgress, ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::thread;

const PATTERNS: [&str; 2] = [
    "../data/NSDUH_Versions/*/*/*Tab.tsv",
    "../data/NSDUH_Versions/*/*/*/*Tab.tsv",
];

fn open_default_split_file(year: &str, parent_dir: &Path, header: &[String]) -> BufWriter<File> {
    let path = parent_dir.join(&format!("{year}.tsv"));
    let mut f = BufWriter::new(File::create(path).unwrap());
    writeln!(f, "{}", header.join("\t")).unwrap();
    f
}

fn main() -> Result<()> {
    let tsv_paths: Vec<_> = PATTERNS
        .iter()
        .flat_map(|pattern| glob(pattern).unwrap())
        .map(|path| path.unwrap())
        .collect();
    assert_eq!(tsv_paths.len(), 5);

    let m = MultiProgress::new();
    let sty = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .progress_chars("##-");

    let ns = tsv_paths
        .iter()
        .map(|path| Ok(BufReader::new(File::open(path.clone())?).lines().count()))
        .collect::<Result<Vec<_>>>()?;

    for (path, n) in tsv_paths.into_iter().zip(ns.into_iter()) {
        let pb = m.add(ProgressBar::new(n as u64).with_style(sty.clone()));

        let _ = thread::spawn(move || -> Result<()> {
            let parent_dir = path.parent().unwrap();
            let buf = BufReader::new(File::open(path.clone())?);
            let mut lines = buf.lines();
            let header: Vec<String> = lines.next().unwrap()?.split('\t').map_into().collect();

            let mut datasets_per_year = HashMap::new();

            let year_ind = header.iter().position(|name| name == "YEAR").unwrap();

            let dtypes: Vec<ColType> = lines
                .progress_with(pb)
                .map(|line| line.unwrap())
                .map(|line| {
                    let token = line.split('\t').collect::<Vec<_>>();
                    let year = token[year_ind];
                    let token: Vec<_> = token.into_iter().map_into().collect();

                    let f = datasets_per_year
                        .entry(year.to_string())
                        .or_insert_with(|| open_default_split_file(year, parent_dir, &header));
                    writeln!(f, "{}", line).unwrap();
                    token
                })
                .fold(vec![ColType::Int; n], |mut v, token| {
                    v.iter_mut().zip(token).for_each(|(a, b)| *a = a.combine(b));
                    v
                });

            let dtypes_path = parent_dir.join("schema.txt");
            let dtypes_dict = dtypes
                .iter()
                .zip(header.iter())
                .map(|(v, h)| format!("\"{h}\": {v},\n"))
                .collect::<String>();
            fs::write(dtypes_path, format!("{{\n{dtypes_dict}}}\n"))?;
            Ok(())
        });
    }

    m.join_and_clear()?;
    Ok(())
}
