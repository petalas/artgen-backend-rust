use artgen_shared::auto_tune::*;
use artgen_shared::benchmark::BenchmarkRequest;
use artgen_shared::mutation_params::MutationParams;

/// Coordinate descent parameter optimizer.
/// Optimizes one parameter at a time: for each enabled param, probe base ± step_size
/// in normalized [0,1] space, pick the best, then move to the next param.
/// After all params, shrink step and repeat until step_size < min_step_size.
pub struct AutoTuner {
    pub config: AutoTuneConfig,
    pub trials: Vec<TrialRecord>,
    next_trial_number: u32,
    pub progress: CoordDescentProgress,
    /// Normalized params of the currently pending trial.
    pending_normalized: Vec<f32>,
}

impl AutoTuner {
    pub fn new(config: AutoTuneConfig) -> Self {
        let base_normalized = params_to_normalized(&config.base_params, &config.param_specs);
        let progress = CoordDescentProgress {
            pass: 1,
            param_index: 0,
            phase: CoordPhase::Baseline,
            step_size: config.initial_step_size,
            base_normalized,
            base_fitness: 0.0,
            replicate_fitnesses: Vec::new(),
            high_fitness: None,
            low_fitness: None,
            param_results: Vec::new(),
        };
        Self {
            config,
            trials: Vec::new(),
            next_trial_number: 1,
            progress,
            pending_normalized: Vec::new(),
        }
    }

    /// Restore from persisted state.
    pub fn from_state(state: AutoTuneState) -> Self {
        Self {
            config: state.config,
            trials: state.trials,
            next_trial_number: state.next_trial_number,
            progress: state.progress,
            pending_normalized: Vec::new(),
        }
    }

    /// Whether the optimizer has finished (step_size below minimum).
    pub fn is_done(&self) -> bool {
        self.progress.phase == CoordPhase::Done
    }

    /// Generate the next trial as a BenchmarkRequest.
    pub fn next_trial(&mut self) -> BenchmarkRequest {
        let trial_num = self.next_trial_number;
        self.next_trial_number += 1;

        let (normalized, label) = match self.progress.phase {
            CoordPhase::Baseline => {
                let normalized = self.progress.base_normalized.clone();
                let label = format!("cd-{:03} baseline", trial_num);
                (normalized, label)
            }
            CoordPhase::ProbeHigh => {
                let mut normalized = self.progress.base_normalized.clone();
                let enabled_idx = self.progress.param_index;
                let spec = self.enabled_spec(enabled_idx);
                let base_val = normalized[enabled_idx];
                normalized[enabled_idx] = (base_val + self.progress.step_size).min(1.0);
                let native = denormalize(normalized[enabled_idx], &spec.kind);
                let label = format!(
                    "cd-{:03} {} +{:.0}% ({:.4})",
                    trial_num,
                    spec.name,
                    self.progress.step_size * 100.0,
                    native
                );
                (normalized, label)
            }
            CoordPhase::ProbeLow => {
                let mut normalized = self.progress.base_normalized.clone();
                let enabled_idx = self.progress.param_index;
                let spec = self.enabled_spec(enabled_idx);
                let base_val = normalized[enabled_idx];
                normalized[enabled_idx] = (base_val - self.progress.step_size).max(0.0);
                let native = denormalize(normalized[enabled_idx], &spec.kind);
                let label = format!(
                    "cd-{:03} {} -{:.0}% ({:.4})",
                    trial_num,
                    spec.name,
                    self.progress.step_size * 100.0,
                    native
                );
                (normalized, label)
            }
            CoordPhase::Done => {
                // Shouldn't be called when done, but return baseline as safety
                let normalized = self.progress.base_normalized.clone();
                let label = format!("cd-{:03} done", trial_num);
                (normalized, label)
            }
        };

        let params = normalized_to_params(&normalized, &self.config.param_specs, &self.config.base_params);
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

    /// Record a completed trial result.
    pub fn record_result(&mut self, result_id: &str, final_fitness: f32, improvements_per_sec: f64, label: &str) {
        let trial_num = self.trials.len() as u32 + 1;
        let phase_str = format!("{:?}", self.progress.phase);
        let normalized = std::mem::take(&mut self.pending_normalized);

        self.trials.push(TrialRecord {
            trial_number: trial_num,
            label: label.to_string(),
            normalized_params: normalized,
            final_fitness,
            improvements_per_sec,
            result_id: result_id.to_string(),
            phase: phase_str,
        });

        self.progress.replicate_fitnesses.push(final_fitness);

        // Check if we have enough replicates
        if self.progress.replicate_fitnesses.len() >= self.config.replicates as usize {
            let median = median_fitness(&self.progress.replicate_fitnesses);
            self.progress.replicate_fitnesses.clear();
            self.decide_phase(median);
        }
    }

    /// Process the median fitness for the current phase and advance.
    fn decide_phase(&mut self, fitness: f32) {
        match self.progress.phase {
            CoordPhase::Baseline => {
                self.progress.base_fitness = fitness;
                // Start probing the first enabled param
                self.advance_to_first_probeable_param();
            }
            CoordPhase::ProbeHigh => {
                self.progress.high_fitness = Some(fitness);
                // Now try low probe — but check if it would clamp to same as base
                let enabled_idx = self.progress.param_index;
                let base_val = self.progress.base_normalized[enabled_idx];
                let low_val = (base_val - self.progress.step_size).max(0.0);
                let spec = self.enabled_spec(enabled_idx);
                if values_same_after_denorm(base_val, low_val, &spec.kind) {
                    // Skip low probe, decide with just high
                    self.progress.low_fitness = None;
                    self.decide_and_advance();
                } else {
                    self.progress.phase = CoordPhase::ProbeLow;
                }
            }
            CoordPhase::ProbeLow => {
                self.progress.low_fitness = Some(fitness);
                self.decide_and_advance();
            }
            CoordPhase::Done => {}
        }
    }

    /// Pick best of {base, high, low}, update base if improved, advance to next param.
    fn decide_and_advance(&mut self) {
        let enabled_idx = self.progress.param_index;
        let spec = self.enabled_spec(enabled_idx);
        let param_name = spec.name.clone();
        let base_fitness = self.progress.base_fitness;

        let high_fitness = self.progress.high_fitness;
        let low_fitness = self.progress.low_fitness;

        // Find best
        let mut best_fitness = base_fitness;
        let mut best_choice = "base";
        if let Some(hf) = high_fitness {
            if hf > best_fitness {
                best_fitness = hf;
                best_choice = "high";
            }
        }
        if let Some(lf) = low_fitness {
            if lf > best_fitness {
                best_fitness = lf;
                best_choice = "low";
            }
        }

        let improvement = best_fitness - base_fitness;

        // Update base params if improved
        match best_choice {
            "high" => {
                let base_val = self.progress.base_normalized[enabled_idx];
                self.progress.base_normalized[enabled_idx] = (base_val + self.progress.step_size).min(1.0);
                self.progress.base_fitness = best_fitness;
            }
            "low" => {
                let base_val = self.progress.base_normalized[enabled_idx];
                self.progress.base_normalized[enabled_idx] = (base_val - self.progress.step_size).max(0.0);
                self.progress.base_fitness = best_fitness;
            }
            _ => {} // keep base
        }

        self.progress.param_results.push(ParamProbeResult {
            param_name,
            base_fitness,
            high_fitness,
            low_fitness,
            chosen: best_choice.to_string(),
            improvement,
        });

        // Clear probe fitnesses for next param
        self.progress.high_fitness = None;
        self.progress.low_fitness = None;

        // Advance to next param
        self.advance_to_next_param();
    }

    /// Move to the next enabled param, or start a new pass if all done.
    fn advance_to_next_param(&mut self) {
        let enabled_count = self.enabled_count();
        let mut next_idx = self.progress.param_index + 1;

        // Skip params that can't be probed at current step_size
        while next_idx < enabled_count {
            if !self.should_skip_param(next_idx) {
                break;
            }
            next_idx += 1;
        }

        if next_idx < enabled_count {
            self.progress.param_index = next_idx;
            self.start_probing_current_param();
        } else {
            // All params done for this pass — shrink step and start new pass
            self.progress.step_size *= self.config.step_decay;
            if self.progress.step_size < self.config.min_step_size {
                self.progress.phase = CoordPhase::Done;
                println!(
                    "[AutoTune] Done after {} passes, step_size {:.4} < min {:.4}",
                    self.progress.pass, self.progress.step_size, self.config.min_step_size
                );
            } else {
                self.progress.pass += 1;
                self.progress.param_index = 0;
                self.progress.param_results.clear();
                // Re-baseline at start of each pass
                self.progress.phase = CoordPhase::Baseline;
                println!(
                    "[AutoTune] Pass {} starting, step_size={:.4}",
                    self.progress.pass, self.progress.step_size
                );
            }
        }
    }

    /// Find the first enabled param that can be probed, starting from index 0.
    fn advance_to_first_probeable_param(&mut self) {
        let enabled_count = self.enabled_count();
        let mut idx = 0;
        while idx < enabled_count {
            if !self.should_skip_param(idx) {
                break;
            }
            idx += 1;
        }
        if idx < enabled_count {
            self.progress.param_index = idx;
            self.start_probing_current_param();
        } else {
            // No params can be probed at this step size — shrink and try again
            self.progress.step_size *= self.config.step_decay;
            if self.progress.step_size < self.config.min_step_size {
                self.progress.phase = CoordPhase::Done;
            } else {
                self.progress.pass += 1;
                self.progress.param_results.clear();
                self.advance_to_first_probeable_param();
            }
        }
    }

    /// Begin probing the current param — start with high, skip if clamped to same.
    fn start_probing_current_param(&mut self) {
        let enabled_idx = self.progress.param_index;
        let base_val = self.progress.base_normalized[enabled_idx];
        let high_val = (base_val + self.progress.step_size).min(1.0);
        let spec = self.enabled_spec(enabled_idx);

        if values_same_after_denorm(base_val, high_val, &spec.kind) {
            // High clamped to same — try low directly
            let low_val = (base_val - self.progress.step_size).max(0.0);
            if values_same_after_denorm(base_val, low_val, &spec.kind) {
                // Both directions clamp to same — skip this param entirely
                self.progress.param_results.push(ParamProbeResult {
                    param_name: spec.name.clone(),
                    base_fitness: self.progress.base_fitness,
                    high_fitness: None,
                    low_fitness: None,
                    chosen: "skip".to_string(),
                    improvement: 0.0,
                });
                self.advance_to_next_param();
            } else {
                // Only low is viable
                self.progress.high_fitness = None;
                self.progress.phase = CoordPhase::ProbeLow;
            }
        } else {
            self.progress.phase = CoordPhase::ProbeHigh;
        }
    }

    /// Whether a param should be skipped at the current step size.
    fn should_skip_param(&self, enabled_idx: usize) -> bool {
        let spec = self.enabled_spec(enabled_idx);
        let base_val = self.progress.base_normalized[enabled_idx];

        // For Pow2 params, skip if step_size can't change the exponent
        if let ParamKind::Pow2 { min_exp, max_exp } = &spec.kind {
            if max_exp > min_exp {
                let min_step = 0.5 / (max_exp - min_exp) as f32;
                if self.progress.step_size < min_step {
                    return true;
                }
            }
        }

        // Skip if both high and low clamp to same value as base
        let high_val = (base_val + self.progress.step_size).min(1.0);
        let low_val = (base_val - self.progress.step_size).max(0.0);
        values_same_after_denorm(base_val, high_val, &spec.kind)
            && values_same_after_denorm(base_val, low_val, &spec.kind)
    }

    /// Get the enabled ParamSpec at the given enabled-index.
    fn enabled_spec(&self, enabled_idx: usize) -> &ParamSpec {
        self.config
            .param_specs
            .iter()
            .filter(|p| p.enabled)
            .nth(enabled_idx)
            .expect("enabled_idx out of bounds")
    }

    /// Count of enabled params.
    fn enabled_count(&self) -> usize {
        self.config.param_specs.iter().filter(|p| p.enabled).count()
    }

    /// Estimate total trials for the current configuration.
    fn estimate_total_trials(&self) -> u32 {
        let enabled = self.enabled_count() as u32;
        let reps = self.config.replicates;
        // Each pass: 1 baseline + up to enabled*2 probes, each repeated `reps` times
        let trials_per_pass = (1 + enabled * 2) * reps;
        // Estimate number of passes
        let mut step = self.config.initial_step_size;
        let mut passes = 0u32;
        while step >= self.config.min_step_size {
            passes += 1;
            step *= self.config.step_decay;
        }
        trials_per_pass * passes.max(1)
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

        let phase_str = match self.progress.phase {
            CoordPhase::Baseline => "baseline".to_string(),
            CoordPhase::ProbeHigh => "probe-high".to_string(),
            CoordPhase::ProbeLow => "probe-low".to_string(),
            CoordPhase::Done => "done".to_string(),
        };

        let current_param = if self.progress.phase != CoordPhase::Baseline && self.progress.phase != CoordPhase::Done {
            let spec = self.enabled_spec(self.progress.param_index);
            Some(spec.name.clone())
        } else {
            None
        };

        let current_direction = match self.progress.phase {
            CoordPhase::ProbeHigh => Some("high".to_string()),
            CoordPhase::ProbeLow => Some("low".to_string()),
            _ => None,
        };

        AutoTuneStatus {
            running,
            trial_number: self.trials.len() as u32,
            total_trials: self.estimate_total_trials(),
            phase: phase_str,
            best_fitness,
            best_trial,
            snapshot_id: self.config.snapshot_id.clone(),
            current_param,
            current_direction,
            pass: self.progress.pass,
            step_size: self.progress.step_size,
            base_fitness: self.progress.base_fitness,
            param_results: self.progress.param_results.clone(),
            config: self.config.clone(),
        }
    }

    /// Serialize to persistable state.
    pub fn to_state(&self) -> AutoTuneState {
        AutoTuneState {
            config: self.config.clone(),
            trials: self.trials.clone(),
            next_trial_number: self.next_trial_number,
            progress: self.progress.clone(),
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

/// Compute median of a slice of f32 values.
fn median_fitness(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Check if two normalized values produce the same denormalized result.
fn values_same_after_denorm(a: f32, b: f32, kind: &ParamKind) -> bool {
    let da = denormalize(a, kind);
    let db = denormalize(b, kind);
    (da - db).abs() < 1e-7
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
        // Booleans — disabled by default (single_mutation_mode always on, adaptive always off)
        ParamSpec { name: "single_mutation_mode".into(), kind: ParamKind::Boolean, enabled: false },
        ParamSpec { name: "adaptive_mutation".into(), kind: ParamKind::Boolean, enabled: false },
    ]
}
