pub use artgen_shared::mutation_params::MutationParams;

#[cfg(test)]
mod tests {
    use super::MutationParams;
    use crate::settings;

    #[test]
    fn sanitize_caps_max_polygons_at_storage_capacity() {
        let capacity = u32::try_from(settings::POLYGON_CAPACITY).unwrap();
        let mut params = MutationParams { max_polygons: capacity, ..MutationParams::default() };
        params.sanitize();
        assert_eq!(params.max_polygons, capacity);
        params.max_polygons = capacity + 1;
        params.sanitize();
        assert_eq!(params.max_polygons, capacity);
        assert_eq!(MutationParams::default().max_polygons, settings::DEFAULT_MAX_POLYGONS);
    }

    #[test]
    fn sanitize_bounds_the_per_point_penalty() {
        for (given, expected) in [
            (settings::PER_POINT_PENALTY, settings::PER_POINT_PENALTY),
            (0.0, 0.0),
            (-1.0, 0.0),
            (1.0, settings::MAX_PER_POINT_PENALTY),
            (f32::NAN, 0.0),
        ] {
            let mut params = MutationParams { per_point_penalty: given, ..MutationParams::default() };
            params.sanitize();
            assert_eq!(params.per_point_penalty.to_bits(), expected.to_bits(), "{given}");
        }
    }

    /// Records and fixtures written before the field keep their exact maps.
    #[test]
    fn the_default_penalty_is_omitted_and_restored() {
        let json = serde_json::to_value(MutationParams::default()).unwrap();
        assert!(json.get("perPointPenalty").is_none());
        let parsed: MutationParams = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, MutationParams::default());
        let strong = MutationParams { per_point_penalty: 3.2e-6, ..MutationParams::default() };
        let json = serde_json::to_value(&strong).unwrap();
        assert_eq!(json["perPointPenalty"].as_f64(), Some(f64::from(3.2e-6_f32)));
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), strong);
    }

    #[test]
    fn sanitize_bounds_the_step_spread_and_keeps_min_at_most_max() {
        let limit = settings::STEP_SPREAD_LIMIT;
        for ((min, max), expected) in [
            ((1.0, 1.0), (1.0, 1.0)),
            ((0.25, 4.0), (0.25, 4.0)),
            ((0.0, 100.0), (1.0 / limit, limit)),
            ((f32::NAN, f32::NAN), (1.0, 1.0)),
            // A min above max is lowered to max.
            ((4.0, 0.5), (0.5, 0.5)),
            ((-1.0, f32::INFINITY), (1.0 / limit, limit)),
        ] {
            let mut params =
                MutationParams { step_spread_min: min, step_spread_max: max, ..MutationParams::default() };
            params.sanitize();
            assert_eq!((params.step_spread_min, params.step_spread_max), expected, "{min} {max}");
        }
    }

    #[test]
    fn step_multipliers_are_log_spaced_from_min_to_max() {
        let ones = MutationParams::default().step_multipliers();
        assert!(ones.iter().all(|m| m.to_bits() == 1.0f32.to_bits()), "{ones:?}");
        let spread = MutationParams { step_spread_min: 0.25, step_spread_max: 4.0, ..MutationParams::default() };
        let rungs = spread.step_multipliers();
        assert_eq!(rungs.len(), settings::STEP_RUNGS);
        assert_eq!(rungs[0], 0.25);
        assert!((rungs[settings::STEP_RUNGS - 1] - 4.0).abs() < 1e-5, "{rungs:?}");
        // A constant ratio between neighbours: 16^(1/7).
        let ratio = 16f32.powf(1.0 / 7.0);
        for pair in rungs.windows(2) {
            assert!((pair[1] / pair[0] - ratio).abs() < 1e-5, "{rungs:?}");
        }
        // Equal ends give that value on every rung; out-of-range ends are
        // taken as sanitize leaves them.
        let flat = MutationParams { step_spread_min: 2.0, step_spread_max: 2.0, ..MutationParams::default() };
        assert_eq!(flat.step_multipliers(), [2.0; settings::STEP_RUNGS]);
        let wild = MutationParams { step_spread_min: 0.0, step_spread_max: 1e9, ..MutationParams::default() };
        let rungs = wild.step_multipliers();
        assert_eq!(rungs[0], 1.0 / settings::STEP_SPREAD_LIMIT);
        assert!((rungs[settings::STEP_RUNGS - 1] - settings::STEP_SPREAD_LIMIT).abs() < 1e-4);
    }

    #[test]
    fn the_default_step_spread_is_omitted_and_restored() {
        let json = serde_json::to_value(MutationParams::default()).unwrap();
        assert!(json.get("stepSpreadMin").is_none() && json.get("stepSpreadMax").is_none());
        let spread = MutationParams { step_spread_min: 0.5, step_spread_max: 2.0, ..MutationParams::default() };
        let json = serde_json::to_value(&spread).unwrap();
        assert_eq!((json["stepSpreadMin"].as_f64(), json["stepSpreadMax"].as_f64()), (Some(0.5), Some(2.0)));
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), spread);
    }
}
