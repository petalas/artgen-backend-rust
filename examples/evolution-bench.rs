//! Main-thread GPU worker. Prefer the supervised scripts/evolution-bench.py CLI,
//! or scripts/hillclimb.py, which drives the resident `serve` mode.
use anyhow::{bail, ensure, Context, Result};
use artgen_backend_rust::{
    benchmark::{
        fixtures,
        quality::{self, QualitySpec},
        trial::{self, ResidentCase, TimestampMode, TrialSpec},
    },
    gpu_evolver::EngineCapacity,
};
use std::{
    collections::BTreeMap,
    io::{BufRead, Write},
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().context(
        "expected freeze, trial, quality or serve; use scripts/evolution-bench.py --help",
    )?;
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
        "quality" => quality(values)?,
        "serve" => {
            ensure!(values.len() == 1, "serve expects only --manifest");
            serve(&path(&values, "manifest")?)?;
        }
        _ => bail!("unknown command {command}"),
    }
    Ok(())
}

/// One blank-start quality run in its own process (the supervisor uses `serve`).
fn quality(mut values: BTreeMap<String, String>) -> Result<()> {
    let manifest = path(&values, "manifest")?;
    values.remove("manifest");
    let (spec, output) = quality_spec(values)?;
    let capacity = quality_capacity(&spec)?;
    let mut case = ResidentCase::open_for(&manifest, &spec.case_id, Some(capacity))?;
    let record = case.run_quality(&spec, &output)?;
    println!("{}", serde_json::to_string(&record)?);
    ensure!(
        record.errors.is_empty(),
        "quality run reported device errors"
    );
    Ok(())
}

/// Read one request per stdin line: the `trial` flags without `--manifest`, or
/// `quality` followed by the `quality` flags without `--manifest`. Each case's
/// engine is built once on first use, so driver shader compilation is paid once
/// per case instead of once per request. Answers are `@result {json}` lines;
/// other stdout lines are engine logs. Requests that fail before producing a
/// record end the process, because the engine state is then unknown. A case's
/// engine is sized for its first request: a trial's fixture parameters or a
/// quality run's parameters. A later request needing more chains or offspring
/// fails (scripts/quality.py gives each side its own server).
fn serve(manifest: &Path) -> Result<()> {
    let mut cases: BTreeMap<String, ResidentCase> = BTreeMap::new();
    let mut stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let mut words: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        let quality = words.first().is_some_and(|word| word == "quality");
        if quality {
            words.remove(0);
        }
        let values = parse_flags(&words)?;
        let case_id = value(&values, "case")?.to_owned();
        ensure!(
            fixtures::PROJECTS.contains(&case_id.as_str()),
            "unknown case {case_id}"
        );
        let request = if quality {
            Request::Quality(quality_spec(values)?)
        } else {
            Request::Trial(trial_spec(values)?)
        };
        if !cases.contains_key(&case_id) {
            let capacity = match &request {
                Request::Quality((spec, _)) => Some(quality_capacity(spec)?),
                Request::Trial(_) => None,
            };
            let case = ResidentCase::open_for(manifest, &case_id, capacity)?;
            cases.insert(case_id.clone(), case);
        }
        let case = cases.get_mut(&case_id).context("resident case")?;
        let reply = match request {
            Request::Quality((spec, output)) => {
                serde_json::to_string(&case.run_quality(&spec, &output)?)?
            }
            Request::Trial((spec, output)) => serde_json::to_string(&case.run(&spec, &output)?)?,
        };
        writeln!(stdout, "@result {reply}")?;
        stdout.flush()?;
    }
    Ok(())
}

enum Request {
    Trial((TrialSpec, PathBuf)),
    Quality((QualitySpec, PathBuf)),
}

/// The engine capacity a quality run's parameters need.
fn quality_capacity(spec: &QualitySpec) -> Result<EngineCapacity> {
    Ok(EngineCapacity::for_params(&quality::params_with(&spec.overrides)?))
}

/// `--params` is a compact JSON object of camelCase `MutationParams` fields
/// merged onto the defaults (no whitespace, since requests split on it).
fn quality_spec(mut values: BTreeMap<String, String>) -> Result<(QualitySpec, PathBuf)> {
    let overrides = match values.remove("params") {
        None => serde_json::Map::new(),
        Some(json) => serde_json::from_str(&json).context("--params must be a JSON object")?,
    };
    ensure!(
        values.len() == 5,
        "quality expects case, seed, seconds, warmup, output and optional params"
    );
    let spec = QualitySpec {
        case_id: value(&values, "case")?.into(),
        seed: value(&values, "seed")?.parse()?,
        seconds: value(&values, "seconds")?.parse()?,
        warmup_batches: value(&values, "warmup")?.parse()?,
        overrides,
    };
    Ok((spec, path(&values, "output")?))
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
