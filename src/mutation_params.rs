pub use artgen_shared::mutation_params::{MutationParams, PointPenalty};

#[cfg(test)]
mod tests {
    use super::{MutationParams, PointPenalty};
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

    /// Every field is written, read back unchanged, and required.
    #[test]
    fn params_round_trip_and_every_field_is_required() {
        let default = MutationParams::default();
        let json = serde_json::to_value(&default).unwrap();
        assert_eq!(serde_json::from_value::<MutationParams>(json.clone()).unwrap(), default);
        for key in json.as_object().unwrap().keys() {
            let mut missing = json.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert!(serde_json::from_value::<MutationParams>(missing).is_err(), "{key}");
        }
    }

    /// Keys of removed settings are rejected rather than silently dropped.
    #[test]
    fn removed_fields_are_rejected() {
        for key in ["removePolygonProb", "removePointProb", "penaltyExponent", "mergePolygonProb"] {
            let mut json = serde_json::to_value(MutationParams::default()).unwrap();
            json[key] = 0.0.into();
            assert!(serde_json::from_value::<MutationParams>(json).is_err(), "{key}");
        }
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
        let none = MutationParams { step_spread_min: 1.0, step_spread_max: 1.0, ..MutationParams::default() };
        let ones = none.step_multipliers();
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

    /// select.wgsl's compute_fitness expression.
    fn shader_score(fitness: f32, per_point: f32, triangles: u32) -> f32 {
        let mut fitness = fitness;
        let num_points = triangles * 3;
        fitness -= fitness * per_point * num_points as f32;
        fitness
    }

    #[test]
    fn the_penalty_keeps_every_bit_of_the_shader_score() {
        for per_point in [settings::PER_POINT_PENALTY, 3.2e-6, settings::MAX_PER_POINT_PENALTY, 0.0] {
            let penalty = PointPenalty { per_point };
            for triangles in [0, 1, 7, 999, 1000, 1337, 2048] {
                for fitness in [99.123_45_f32, 87.5, 12.0] {
                    let expected = shader_score(fitness, per_point, triangles);
                    let actual = penalty.apply(fitness, triangles);
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{per_point} {triangles}");
                }
                let fraction = f64::from(per_point) * f64::from(triangles * 3);
                assert_eq!(penalty.fraction(triangles).to_bits(), fraction.to_bits());
            }
        }
    }

    /// Fitness stays positive at the largest penalty `sanitize` accepts.
    #[test]
    fn the_fraction_at_capacity_stays_below_one() {
        let capacity = u32::try_from(settings::POLYGON_CAPACITY).unwrap();
        let penalty = PointPenalty { per_point: settings::MAX_PER_POINT_PENALTY };
        let fraction = penalty.fraction(capacity);
        assert!(fraction < 0.9, "{fraction}");
        assert!(penalty.apply(100.0, capacity) > 0.0);
    }
}
