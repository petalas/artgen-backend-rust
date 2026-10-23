//! Main-thread GPU worker. Prefer the supervised scripts/evolution-bench.py CLI.
use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::benchmark::{
    fixtures,
    trial::{self, TimestampMode, TrialSpec},
};
use std::{collections::BTreeMap, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .context("expected freeze or trial; use scripts/evolution-bench.py --help")?;
    let flags: Vec<_> = args.collect();
    let mut values = parse_flags(&flags)?;
    match command.as_str() {
        "freeze" => {
            ensure!(
                values.len() == 3,
                "freeze expects projects-dir, fixtures, resolution"
            );
            let manifest = fixtures::freeze(
                &path(&values, "projects-dir")?,
                &path(&values, "fixtures")?,
                value(&values, "resolution")?.parse()?,
            )?;
            println!("{}", serde_json::to_string(&manifest)?);
        }
        "trial" => {
            let timestamp_mode = match values.remove("profile-mode").as_deref() {
                None | Some("stages") => TimestampMode::Stages,
                Some("batch-envelope") => TimestampMode::BatchEnvelope,
                Some(mode) => bail!("unknown profiling mode {mode}"),
            };
            ensure!(
                values.len() == 7,
                "trial expects manifest, case, seed, batches, warmup, diagnostic, output"
            );
            let spec = TrialSpec {
                case_id: value(&values, "case")?.into(),
                seed: value(&values, "seed")?.parse()?,
                batches: value(&values, "batches")?.parse()?,
                warmup_batches: value(&values, "warmup")?.parse()?,
                diagnostic: value(&values, "diagnostic")?.parse()?,
                timestamp_mode,
            };
            let record = trial::run(
                &path(&values, "manifest")?,
                &spec,
                &path(&values, "output")?,
            )?;
            println!("{}", serde_json::to_string(&record)?);
            ensure!(
                record.correctness,
                "trial failed numerical or completion gates"
            );
        }
        _ => bail!("unknown command {command}"),
    }
    Ok(())
}

fn parse_flags(args: &[String]) -> Result<BTreeMap<String, String>> {
    ensure!(
        args.len() <= 32 && args.len() % 2 == 0,
        "expected bounded --name value pairs"
    );
    let mut values = BTreeMap::new();
    for pair in args.as_chunks::<2>().0 {
        let key = pair[0]
            .strip_prefix("--")
            .context("flags must start with --")?;
        ensure!(
            values.insert(key.into(), pair[1].clone()).is_none(),
            "duplicate flag {key}"
        );
    }
    Ok(values)
}
fn value<'a>(values: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str> {
    values
        .get(name)
        .map(String::as_str)
        .with_context(|| format!("missing --{name}"))
}
fn path(values: &BTreeMap<String, String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(value(values, name)?))
}
