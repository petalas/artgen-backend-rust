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

    #[test]
    fn no_spread_is_omitted_and_a_missing_end_reads_as_no_spread() {
        let none = MutationParams { step_spread_min: 1.0, step_spread_max: 1.0, ..MutationParams::default() };
        let json = serde_json::to_value(&none).unwrap();
        assert!(json.get("stepSpreadMin").is_none() && json.get("stepSpreadMax").is_none());
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), none);
        // The default spread is written out, so it survives a round trip.
        let default = MutationParams::default();
        let json = serde_json::to_value(&default).unwrap();
        assert_eq!((json["stepSpreadMin"].as_f64(), json["stepSpreadMax"].as_f64()), (Some(0.25), Some(4.0)));
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), default);
    }

    #[test]
    fn legacy_micro_color_weight_is_omitted_and_a_missing_one_reads_as_legacy() {
        let legacy = MutationParams { micro_color_prob: 0.002786, ..MutationParams::default() };
        let json = serde_json::to_value(&legacy).unwrap();
        assert!(json.get("microColorProb").is_none());
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), legacy);
        let default = MutationParams::default();
        let json = serde_json::to_value(&default).unwrap();
        assert_eq!(json["microColorProb"].as_f64(), Some(f64::from(0.000697f32)));
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), default);
    }

    /// Records and fixtures written before absorb keep their exact maps; settings
    /// saved with the removed merge thresholds still load.
    #[test]
    fn absorb_is_omitted_at_zero_and_removed_merge_thresholds_are_ignored() {
        let default = MutationParams::default();
        assert_eq!(default.absorb_polygon_prob, 0.0);
        let mut json = serde_json::to_value(&default).unwrap();
        assert!(json.get("absorbPolygonProb").is_none());
        json["mergeCentroidThreshold"] = 0.15.into();
        json["mergeColorThreshold"] = 0.2.into();
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), default);
        let absorbing = MutationParams { absorb_polygon_prob: 0.25, ..default };
        let json = serde_json::to_value(&absorbing).unwrap();
        assert_eq!(json["absorbPolygonProb"].as_f64(), Some(0.25));
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), absorbing);
        for (given, expected) in [(-1.0, 0.0), (0.5, 0.5), (2.0, 1.0)] {
            let mut params = MutationParams { absorb_polygon_prob: given, ..MutationParams::default() };
            params.sanitize();
            assert_eq!(params.absorb_polygon_prob, expected, "{given}");
        }
    }

    #[test]
    fn sanitize_bounds_the_penalty_exponent() {
        for (given, expected) in [
            (0.0, 0.0),
            (1.5, 1.5),
            (-1.0, 0.0),
            (settings::MAX_PENALTY_EXPONENT, settings::MAX_PENALTY_EXPONENT),
            (7.0, settings::MAX_PENALTY_EXPONENT),
            (f32::INFINITY, settings::MAX_PENALTY_EXPONENT),
            (f32::NAN, 0.0),
        ] {
            let mut params = MutationParams { penalty_exponent: given, ..MutationParams::default() };
            params.sanitize();
            assert_eq!(params.penalty_exponent.to_bits(), expected.to_bits(), "{given}");
        }
    }

    /// Records and fixtures written before the field keep their exact maps.
    #[test]
    fn the_flat_penalty_exponent_is_omitted_and_a_missing_one_reads_as_flat() {
        let json = serde_json::to_value(MutationParams::default()).unwrap();
        assert!(json.get("penaltyExponent").is_none());
        let parsed: MutationParams = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.penalty_exponent.to_bits(), 0.0f32.to_bits());
        let scaled = MutationParams { penalty_exponent: 1.5, ..MutationParams::default() };
        let json = serde_json::to_value(&scaled).unwrap();
        assert_eq!(json["penaltyExponent"].as_f64(), Some(1.5));
        assert_eq!(serde_json::from_value::<MutationParams>(json).unwrap(), scaled);
    }

    /// The expression select.wgsl's compute_fitness used before the exponent.
    fn historical(fitness: f32, per_point: f32, triangles: u32) -> f32 {
        let mut fitness = fitness;
        let num_points = triangles * 3;
        fitness -= fitness * per_point * num_points as f32;
        fitness
    }

    #[test]
    fn the_flat_penalty_keeps_every_bit_of_the_historical_score() {
        for per_point in [settings::PER_POINT_PENALTY, 3.2e-6, settings::MAX_PER_POINT_PENALTY, 0.0] {
            let penalty = PointPenalty { per_point, exponent: 0.0 };
            for triangles in [0, 1, 7, 999, 1000, 1337, 2048] {
                for fitness in [99.123_45_f32, 87.5, 12.0] {
                    let expected = historical(fitness, per_point, triangles);
                    let actual = penalty.apply(fitness, triangles);
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{per_point} {triangles}");
                }
                let fraction = f64::from(per_point) * f64::from(triangles * 3);
                assert_eq!(penalty.fraction(triangles).to_bits(), fraction.to_bits());
            }
        }
    }

    #[test]
    fn a_scaled_penalty_follows_the_closed_form() {
        let n0 = f64::from(settings::PENALTY_REFERENCE_TRIANGLES);
        for per_point in [settings::PER_POINT_PENALTY, settings::MAX_PER_POINT_PENALTY] {
            for exponent in [0.5_f32, 1.0, 1.5, 2.0] {
                let penalty = PointPenalty { per_point, exponent };
                let (p, k) = (f64::from(per_point), f64::from(exponent));
                assert_eq!(penalty.fraction(0), 0.0);
                for triangles in [1_u32, 10, 500, 1000, 2048] {
                    let n = f64::from(triangles);
                    let closed = 3.0 * p * n0 / (k + 1.0) * (n / n0).powf(k + 1.0);
                    let fraction = penalty.fraction(triangles);
                    assert!((fraction - closed).abs() <= 1e-6 * closed, "{k} {n}: {fraction} {closed}");
                    let applied = f64::from(penalty.apply(90.0, triangles));
                    assert!((applied - 90.0 * (1.0 - closed)).abs() < 1e-4, "{k} {n}: {applied}");
                }
            }
        }
    }

    /// At N0 triangles the next one costs 3p, as under the flat penalty.
    #[test]
    fn the_next_triangle_at_the_reference_count_costs_three_points() {
        let p = settings::MAX_PER_POINT_PENALTY;
        let n0 = settings::PENALTY_REFERENCE_TRIANGLES;
        for exponent in [1.0_f32, 1.5] {
            let penalty = PointPenalty { per_point: p, exponent };
            // Central difference of P around N0, one triangle each side.
            let cost = (penalty.fraction(n0 + 1) - penalty.fraction(n0 - 1)) / 2.0;
            let flat = 3.0 * f64::from(p);
            assert!((cost - flat).abs() < 1e-3 * flat, "{exponent}: {cost} vs {flat}");
        }
    }

    /// Fitness stays positive at the extremes `sanitize` accepts.
    #[test]
    fn the_fraction_at_capacity_stays_below_one() {
        let capacity = u32::try_from(settings::POLYGON_CAPACITY).unwrap();
        for exponent in [0.0, 0.4, 1.0, 1.5, settings::MAX_PENALTY_EXPONENT] {
            let penalty = PointPenalty { per_point: settings::MAX_PER_POINT_PENALTY, exponent };
            let fraction = penalty.fraction(capacity);
            assert!(fraction < 0.9, "{exponent}: {fraction}");
            assert!(penalty.apply(100.0, capacity) > 0.0);
        }
    }
}
