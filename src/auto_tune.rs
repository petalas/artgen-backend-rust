use artgen_shared::auto_tune::*;
use artgen_shared::benchmark::BenchmarkRequest;
use artgen_shared::mutation_params::MutationParams;
use rand::{Rng, RngExt};

/// Autonomous parameter optimizer using EDA (Estimation of Distribution Algorithm).
pub struct AutoTuner {
    pub config: AutoTuneConfig,
    pub trials: Vec<TrialRecord>,
    next_trial_number: u32,
    /// Pre-generated LHS samples for exploration phase (each is a Vec<f32> of normalized values).
    lhs_samples: Vec<Vec<f32>>,
    /// EDA distribution: mean per enabled param in normalized space.
    param_means: Vec<f32>,
    /// EDA distribution: std per enabled param in normalized space.
    param_stds: Vec<f32>,
    /// Normalized params of the currently pending trial (stored after next_trial()).
    pending_normalized: Vec<f32>,
}

impl AutoTuner {
    pub fn new(config: AutoTuneConfig) -> Self {
        let enabled_count = config.param_specs.iter().filter(|p| p.enabled).count();
        let mut rng = rand::rng();
        let lhs_samples = latin_hypercube(config.exploration_trials as usize, enabled_count, &mut rng);
        let param_means = vec![0.5; enabled_count];
        let param_stds = vec![0.25; enabled_count];
        Self {
            config,
            trials: Vec::new(),
            next_trial_number: 1,
            lhs_samples,
            param_means,
            param_stds,
            pending_normalized: Vec::new(),
        }
    }

    /// Restore from persisted state.
    pub fn from_state(state: AutoTuneState) -> Self {
        let enabled_count = state.config.param_specs.iter().filter(|p| p.enabled).count();
        let mut rng = rand::rng();
        // Regenerate any remaining LHS samples needed
        let exploration_done = state.trials.len() as u32 >= state.config.exploration_trials;
        let lhs_samples = if exploration_done {
            vec![]
        } else {
            let remaining = state.config.exploration_trials as usize - state.trials.len();
            latin_hypercube(remaining, enabled_count, &mut rng)
        };
        Self {
            config: state.config,
            trials: state.trials,
            next_trial_number: state.next_trial_number,
            lhs_samples,
            param_means: state.param_means,
            param_stds: state.param_stds,
            pending_normalized: Vec::new(),
        }
    }

    /// Generate the next trial as a BenchmarkRequest.
    pub fn next_trial(&mut self) -> BenchmarkRequest {
        let trial_num = self.next_trial_number;
        self.next_trial_number += 1;

        let enabled_count = self.config.param_specs.iter().filter(|p| p.enabled).count();
        let in_exploration = (self.trials.len() as u32) < self.config.exploration_trials
            && !self.lhs_samples.is_empty();

        let (normalized, phase) = if in_exploration {
            // Use pre-generated LHS sample
            let sample = self.lhs_samples.remove(0);
            (sample, "explore")
        } else {
            let mut rng = rand::rng();
            // ~exploration_rate chance of pure random
            if rng.random::<f32>() < self.config.exploration_rate {
                let sample: Vec<f32> = (0..enabled_count).map(|_| rng.random::<f32>()).collect();
                (sample, "exploit-random")
            } else {
                // EDA: sample from N(mean, std), clamped to [0,1]
                let sample: Vec<f32> = self
                    .param_means
                    .iter()
                    .zip(self.param_stds.iter())
                    .map(|(&mean, &std)| {
                        let v = mean + std * sample_standard_normal(&mut rng);
                        v.clamp(0.0, 1.0)
                    })
                    .collect();
                (sample, "exploit")
            }
        };

        let params = normalized_to_params(&normalized, &self.config.param_specs, &self.config.base_params);
        let label = format!("auto-{:03} ({})", trial_num, phase);
        self.pending_normalized = normalized;

        BenchmarkRequest {
            drawing_json: self.config.drawing_json.clone(),
            params,
            duration_secs: self.config.duration_secs,
            label,
            resolution: self.config.resolution,
            snapshot_id: self.config.snapshot_id.clone(),
        }
    }

    /// Record a completed trial result, using the pending_normalized params
    /// stored from the most recent next_trial() call.
    pub fn record_result(&mut self, result_id: &str, final_fitness: f32, improvements_per_sec: f64, label: &str) {
        let trial_num = self.trials.len() as u32 + 1;
        let phase = if label.contains("explore") {
            "explore"
        } else {
            "exploit"
        };

        let normalized = std::mem::take(&mut self.pending_normalized);

        self.trials.push(TrialRecord {
            trial_number: trial_num,
            label: label.to_string(),
            normalized_params: normalized,
            final_fitness,
            improvements_per_sec,
            result_id: result_id.to_string(),
            phase: phase.to_string(),
        });

        self.update_distribution();
    }

    /// Recompute EDA distribution from elite trials.
    fn update_distribution(&mut self) {
        let n = self.trials.len();
        if n < 3 {
            return; // Not enough data
        }

        let enabled_count = self.config.param_specs.iter().filter(|p| p.enabled).count();
        if enabled_count == 0 {
            return;
        }

        // Filter trials that have normalized params
        let valid_trials: Vec<&TrialRecord> = self
            .trials
            .iter()
            .filter(|t| t.normalized_params.len() == enabled_count)
            .collect();

        if valid_trials.len() < 3 {
            return;
        }

        // Sort by fitness descending, take top elite_fraction
        let mut sorted: Vec<&TrialRecord> = valid_trials;
        sorted.sort_by(|a, b| b.final_fitness.partial_cmp(&a.final_fitness).unwrap_or(std::cmp::Ordering::Equal));

        let elite_count = ((sorted.len() as f32 * self.config.elite_fraction).ceil() as usize).max(2);
        let elites = &sorted[..elite_count.min(sorted.len())];

        // Compute mean and std for each param dimension
        let mut means = vec![0.0f32; enabled_count];
        let mut stds = vec![0.0f32; enabled_count];

        for i in 0..enabled_count {
            let values: Vec<f32> = elites.iter().map(|t| t.normalized_params[i]).collect();
            let n = values.len() as f32;
            let mean = values.iter().sum::<f32>() / n;
            let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let std = variance.sqrt().max(0.02); // Floor to prevent premature convergence
            means[i] = mean;
            stds[i] = std;
        }

        self.param_means = means;
        self.param_stds = stds;
    }

    /// Compute Spearman rank correlation for parameter importance.
    pub fn compute_importance(&self) -> Vec<(String, f32)> {
        let enabled_specs: Vec<&ParamSpec> = self.config.param_specs.iter().filter(|p| p.enabled).collect();
        let enabled_count = enabled_specs.len();
        if enabled_count == 0 {
            return vec![];
        }

        // Filter trials with valid normalized params
        let valid_trials: Vec<&TrialRecord> = self
            .trials
            .iter()
            .filter(|t| t.normalized_params.len() == enabled_count)
            .collect();

        let n = valid_trials.len();
        if n < 3 {
            return enabled_specs.iter().map(|s| (s.name.clone(), 0.0)).collect();
        }

        // Rank fitness values
        let fitness_ranks = rank_values(&valid_trials.iter().map(|t| t.final_fitness).collect::<Vec<_>>());

        let mut importance = Vec::with_capacity(enabled_count);
        for i in 0..enabled_count {
            let param_values: Vec<f32> = valid_trials.iter().map(|t| t.normalized_params[i]).collect();
            let param_ranks = rank_values(&param_values);
            let corr = spearman_correlation(&param_ranks, &fitness_ranks);
            importance.push((enabled_specs[i].name.clone(), corr));
        }

        // Sort by absolute correlation descending
        importance.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        importance
    }

    /// Build status for WS broadcast.
    pub fn status(&self, running: bool) -> AutoTuneStatus {
        let best = self
            .trials
            .iter()
            .max_by(|a, b| a.final_fitness.partial_cmp(&b.final_fitness).unwrap_or(std::cmp::Ordering::Equal));

        let (best_fitness, best_trial) = best
            .map(|t| (t.final_fitness, t.trial_number))
            .unwrap_or((0.0, 0));

        let phase = if (self.trials.len() as u32) < self.config.exploration_trials {
            "exploration".to_string()
        } else {
            "exploitation".to_string()
        };

        AutoTuneStatus {
            running,
            trial_number: self.trials.len() as u32,
            total_trials: self.config.exploration_trials,
            phase,
            best_fitness,
            best_trial,
            snapshot_id: self.config.snapshot_id.clone(),
            param_importance: self.compute_importance(),
            config: self.config.clone(),
        }
    }

    /// Serialize to persistable state.
    pub fn to_state(&self) -> AutoTuneState {
        AutoTuneState {
            config: self.config.clone(),
            trials: self.trials.clone(),
            next_trial_number: self.next_trial_number,
            param_means: self.param_means.clone(),
            param_stds: self.param_stds.clone(),
        }
    }

    /// Save to disk.
    pub fn save(&self, project: &str) {
        let path = auto_tune_path(project);
        let state = self.to_state();
        match serde_json::to_string_pretty(&state) {
            Ok(json) => {
                let tmp = path.with_extension("json.tmp");
                if let Err(e) = std::fs::write(&tmp, &json) {
                    eprintln!("[AutoTune] Failed to write tmp: {}", e);
                    return;
                }
                if let Err(e) = std::fs::rename(&tmp, &path) {
                    eprintln!("[AutoTune] Failed to rename: {}", e);
                }
            }
            Err(e) => eprintln!("[AutoTune] Failed to serialize: {}", e),
        }
    }

    /// Load from disk.
    pub fn load(project: &str) -> Option<Self> {
        let path = auto_tune_path(project);
        let data = std::fs::read_to_string(&path).ok()?;
        let state: AutoTuneState = serde_json::from_str(&data).ok()?;
        Some(Self::from_state(state))
    }

}

/// File path for auto-tune state.
fn auto_tune_path(project: &str) -> std::path::PathBuf {
    std::path::Path::new("projects").join(project).join("auto_tune.json")
}

/// Latin Hypercube Sampling: generate `n` samples in `d` dimensions, each in [0,1].
fn latin_hypercube(n: usize, d: usize, rng: &mut impl Rng) -> Vec<Vec<f32>> {
    if n == 0 || d == 0 {
        return vec![];
    }

    let mut samples = vec![vec![0.0f32; d]; n];

    for dim in 0..d {
        // Create permutation of 0..n
        let mut perm: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }

        for i in 0..n {
            // Each sample gets a random point within its stratum
            let stratum = perm[i] as f32;
            let u: f32 = rng.random();
            samples[i][dim] = (stratum + u) / n as f32;
        }
    }

    samples
}

/// Convert normalized [0,1] vector to MutationParams.
pub fn normalized_to_params(
    normalized: &[f32],
    specs: &[ParamSpec],
    base: &MutationParams,
) -> MutationParams {
    let mut params = base.clone();
    let mut norm_idx = 0;

    for spec in specs {
        if !spec.enabled {
            continue;
        }
        if norm_idx >= normalized.len() {
            break;
        }
        let v = normalized[norm_idx];
        let native = denormalize(v, &spec.kind);
        set_param_by_name(&mut params, &spec.name, native);
        norm_idx += 1;
    }

    params.sanitize();
    params
}

/// Convert MutationParams to normalized [0,1] vector (only enabled params).
pub fn params_to_normalized(params: &MutationParams, specs: &[ParamSpec]) -> Vec<f32> {
    let mut result = Vec::new();
    for spec in specs {
        if !spec.enabled {
            continue;
        }
        let native = get_param_by_name(params, &spec.name);
        result.push(normalize(native, &spec.kind));
    }
    result
}

/// Map a native value to [0,1] based on ParamKind.
fn normalize(native: f32, kind: &ParamKind) -> f32 {
    match kind {
        ParamKind::LogScale { min, max } => {
            let log_min = min.max(1e-6).ln();
            let log_max = max.ln();
            let log_val = native.max(1e-6).ln();
            ((log_val - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
        }
        ParamKind::Linear { min, max } => {
            if (max - min).abs() < 1e-9 {
                0.5
            } else {
                ((native - min) / (max - min)).clamp(0.0, 1.0)
            }
        }
        ParamKind::Pow2 { min_exp, max_exp } => {
            if max_exp == min_exp {
                0.5
            } else {
                // native is a power of 2, find its exponent
                let exp = if native > 0.0 { (native.log2().round() as u32).clamp(*min_exp, *max_exp) } else { *min_exp };
                (exp - min_exp) as f32 / (max_exp - min_exp) as f32
            }
        }
        ParamKind::Boolean => {
            if native >= 0.5 { 1.0 } else { 0.0 }
        }
    }
}

/// Map [0,1] to native value based on ParamKind.
fn denormalize(v: f32, kind: &ParamKind) -> f32 {
    match kind {
        ParamKind::LogScale { min, max } => {
            let log_min = min.max(1e-6).ln();
            let log_max = max.ln();
            (log_min + v * (log_max - log_min)).exp()
        }
        ParamKind::Linear { min, max } => {
            min + v * (max - min)
        }
        ParamKind::Pow2 { min_exp, max_exp } => {
            let range = (max_exp - min_exp) as f32;
            let exp = (*min_exp as f32 + v * range).round() as u32;
            let exp = exp.clamp(*min_exp, *max_exp);
            2u32.pow(exp) as f32
        }
        ParamKind::Boolean => {
            if v >= 0.5 { 1.0 } else { 0.0 }
        }
    }
}

/// Get a parameter value by field name.
fn get_param_by_name(params: &MutationParams, name: &str) -> f32 {
    match name {
        "add_polygon_prob" => params.add_polygon_prob,
        "remove_polygon_prob" => params.remove_polygon_prob,
        "reorder_polygon_prob" => params.reorder_polygon_prob,
        "offset_polygon_prob" => params.offset_polygon_prob,
        "move_point_prob" => params.move_point_prob,
        "remove_point_prob" => params.remove_point_prob,
        "micro_adjust_prob" => params.micro_adjust_prob,
        "change_color_prob" => params.change_color_prob,
        "adjust_brightness_prob" => params.adjust_brightness_prob,
        "adjust_saturation_prob" => params.adjust_saturation_prob,
        "move_point_max_delta" => params.move_point_max_delta,
        "micro_adjust_delta" => params.micro_adjust_delta,
        "new_point_max_distance" => params.new_point_max_distance,
        "offset_polygon_magnitude" => params.offset_polygon_magnitude,
        "crossover_prob" => params.crossover_prob,
        "spatial_crossover_weight" => params.spatial_crossover_weight,
        "lambda" => params.lambda as f32,
        "single_mutation_mode" => if params.single_mutation_mode { 1.0 } else { 0.0 },
        "adaptive_mutation" => if params.adaptive_mutation { 1.0 } else { 0.0 },
        _ => 0.0,
    }
}

/// Set a parameter value by field name.
fn set_param_by_name(params: &mut MutationParams, name: &str, value: f32) {
    match name {
        "add_polygon_prob" => params.add_polygon_prob = value,
        "remove_polygon_prob" => params.remove_polygon_prob = value,
        "reorder_polygon_prob" => params.reorder_polygon_prob = value,
        "offset_polygon_prob" => params.offset_polygon_prob = value,
        "move_point_prob" => params.move_point_prob = value,
        "remove_point_prob" => params.remove_point_prob = value,
        "micro_adjust_prob" => params.micro_adjust_prob = value,
        "change_color_prob" => params.change_color_prob = value,
        "adjust_brightness_prob" => params.adjust_brightness_prob = value,
        "adjust_saturation_prob" => params.adjust_saturation_prob = value,
        "move_point_max_delta" => params.move_point_max_delta = value,
        "micro_adjust_delta" => params.micro_adjust_delta = value,
        "new_point_max_distance" => params.new_point_max_distance = value,
        "offset_polygon_magnitude" => params.offset_polygon_magnitude = value,
        "crossover_prob" => params.crossover_prob = value,
        "spatial_crossover_weight" => params.spatial_crossover_weight = value,
        "lambda" => params.lambda = value as u32,
        "single_mutation_mode" => params.single_mutation_mode = value >= 0.5,
        "adaptive_mutation" => params.adaptive_mutation = value >= 0.5,
        _ => {}
    }
}

/// Build the default parameter search space.
pub fn default_param_specs() -> Vec<ParamSpec> {
    vec![
        // Probabilities (LogScale)
        ParamSpec { name: "add_polygon_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.5 }, enabled: true },
        ParamSpec { name: "remove_polygon_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        ParamSpec { name: "reorder_polygon_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        ParamSpec { name: "offset_polygon_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        ParamSpec { name: "move_point_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.5 }, enabled: true },
        ParamSpec { name: "remove_point_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        ParamSpec { name: "micro_adjust_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.5 }, enabled: true },
        ParamSpec { name: "change_color_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        ParamSpec { name: "adjust_brightness_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        ParamSpec { name: "adjust_saturation_prob".into(), kind: ParamKind::LogScale { min: 1e-4, max: 0.1 }, enabled: true },
        // Deltas (Linear)
        ParamSpec { name: "move_point_max_delta".into(), kind: ParamKind::Linear { min: 0.01, max: 0.5 }, enabled: true },
        ParamSpec { name: "micro_adjust_delta".into(), kind: ParamKind::Linear { min: 0.001, max: 0.1 }, enabled: true },
        ParamSpec { name: "new_point_max_distance".into(), kind: ParamKind::Linear { min: 0.005, max: 0.2 }, enabled: true },
        ParamSpec { name: "offset_polygon_magnitude".into(), kind: ParamKind::Linear { min: 0.01, max: 0.5 }, enabled: true },
        // Crossover (Linear)
        ParamSpec { name: "crossover_prob".into(), kind: ParamKind::Linear { min: 0.0, max: 1.0 }, enabled: true },
        ParamSpec { name: "spatial_crossover_weight".into(), kind: ParamKind::Linear { min: 0.0, max: 1.0 }, enabled: true },
        // Lambda (Pow2)
        ParamSpec { name: "lambda".into(), kind: ParamKind::Pow2 { min_exp: 0, max_exp: 6 }, enabled: true },
        // Booleans
        ParamSpec { name: "single_mutation_mode".into(), kind: ParamKind::Boolean, enabled: true },
        ParamSpec { name: "adaptive_mutation".into(), kind: ParamKind::Boolean, enabled: true },
    ]
}

/// Rank values (average rank for ties). Higher values get higher ranks.
fn rank_values(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j - 1) as f32 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation coefficient.
fn spearman_correlation(x_ranks: &[f32], y_ranks: &[f32]) -> f32 {
    let n = x_ranks.len() as f32;
    if n < 3.0 {
        return 0.0;
    }

    let x_mean = x_ranks.iter().sum::<f32>() / n;
    let y_mean = y_ranks.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..x_ranks.len() {
        let dx = x_ranks[i] - x_mean;
        let dy = y_ranks[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

/// Sample from standard normal distribution using Box-Muller transform.
fn sample_standard_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
