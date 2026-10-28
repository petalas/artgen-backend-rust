//! Main-thread GPU worker. Prefer the supervised scripts/evolution-bench.py CLI,
//! or scripts/hillclimb.py, which drives the resident `serve` mode.
use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::benchmark::{
    fixtures,
    trial::{self, ResidentCase, TimestampMode, TrialSpec},
};
use std::{
    collections::BTreeMap,
    io::{BufRead, Write},
    path::{Path, PathBuf},
};

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
            let manifest = path(&values, "manifest")?;
            values.remove("manifest");
            let (spec, output) = trial_spec(values)?;
            let record = trial::run(&manifest, &spec, &output)?;
            println!("{}", serde_json::to_string(&record)?);
            ensure!(
                record.correctness,
                "trial failed numerical or completion gates"
            );
        }
        "serve" => {
            ensure!(values.len() == 1, "serve expects only --manifest");
            serve(&path(&values, "manifest")?)?;
        }
        _ => bail!("unknown command {command}"),
    }
    Ok(())
}

/// Read one trial request per stdin line, using the `trial` flags without
/// `--manifest`. Each case's engine is built once on first use, so driver
/// shader compilation is paid once per case instead of once per trial. Answers
/// are `@result {json}` lines; other stdout lines are engine logs. Requests that
/// fail before producing a record end the process, because the engine state is
/// then unknown.
fn serve(manifest: &Path) -> Result<()> {
    let mut cases: BTreeMap<String, ResidentCase> = BTreeMap::new();
    let mut stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let words: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        let (spec, output) = trial_spec(parse_flags(&words)?)?;
        ensure!(
            fixtures::PROJECTS.contains(&spec.case_id.as_str()),
            "unknown case {}",
            spec.case_id
        );
        if !cases.contains_key(&spec.case_id) {
            let case = ResidentCase::open(manifest, &spec.case_id)?;
            cases.insert(spec.case_id.clone(), case);
        }
        let case = cases.get_mut(&spec.case_id).context("resident case")?;
        let record = case.run(&spec, &output)?;
        writeln!(stdout, "@result {}", serde_json::to_string(&record)?)?;
        stdout.flush()?;
    }
    Ok(())
}

fn trial_spec(mut values: BTreeMap<String, String>) -> Result<(TrialSpec, PathBuf)> {
    let timestamp_mode = match values.remove("profile-mode").as_deref() {
        None | Some("stages") => TimestampMode::Stages,
        Some("batch-envelope") => TimestampMode::BatchEnvelope,
        Some(mode) => bail!("unknown profiling mode {mode}"),
    };
    ensure!(
        values.len() == 6,
        "trial expects case, seed, batches, warmup, diagnostic, output"
    );
    let spec = TrialSpec {
        case_id: value(&values, "case")?.into(),
        seed: value(&values, "seed")?.parse()?,
        batches: value(&values, "batches")?.parse()?,
        warmup_batches: value(&values, "warmup")?.parse()?,
        diagnostic: value(&values, "diagnostic")?.parse()?,
        timestamp_mode,
    };
    Ok((spec, path(&values, "output")?))
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
