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
}
