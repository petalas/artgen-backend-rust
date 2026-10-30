//! Operator stats: per search operator, how many offspring it produced
//! (offered) and how many of them select accepted as a strict improvement
//! (improved) or as an equal-fitness move (neutral).
//!
//! mutate writes each offspring's executed operator to `offspring_ops`, one
//! word per offspring slot, outside any drawing state, so canonical state and
//! replay identity do not see it. select counts its chain's λ codes in
//! workgroup memory and adds them, and the accepted winner's code, to the
//! cumulative u32 counters of `operator_stats`. The batch readback copies those
//! counters with the control flags; [`OperatorTally`] turns the copies into u64
//! totals. The codes are WGSL constants generated from [`Operator`].
//!
//! Counters start at zero when chains are (re)initialized and are not part of
//! the search, so shader variants that never write them leave them meaningless.

use std::fmt::Write as _;

use serde::Serialize;

/// The operator that produced an offspring: the branch of mutate that ran, or
/// crossover. The discriminant is the shader code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Operator {
    Add,
    Remove,
    Reorder,
    Scale,
    Rotate,
    AdjacentSwap,
    Merge,
    Clone,
    SwapColors,
    Offset,
    MovePoint,
    MediumMove,
    /// `micro_adjust`'s vertex branch.
    MicroMove,
    ChangeColor,
    /// `micro_adjust`'s color branch.
    MicroColor,
    Brightness,
    /// Also the roulette's fallback branch.
    Saturation,
    Crossover,
    /// No branch could run (an empty parent whose roulette pick failed its
    /// guard): the offspring is the parent, evaluated over the whole image.
    Unchanged,
}

pub const OPERATOR_COUNT: usize = 19;

/// Words of `operator_stats`: offered, then improved, then neutral, each
/// indexed by operator code.
pub const OPERATOR_STATS_WORDS: usize = 3 * OPERATOR_COUNT;

impl Operator {
    /// In code order: `ALL[i] as usize == i`.
    pub const ALL: [Self; OPERATOR_COUNT] = [
        Self::Add,
        Self::Remove,
        Self::Reorder,
        Self::Scale,
        Self::Rotate,
        Self::AdjacentSwap,
        Self::Merge,
        Self::Clone,
        Self::SwapColors,
        Self::Offset,
        Self::MovePoint,
        Self::MediumMove,
        Self::MicroMove,
        Self::ChangeColor,
        Self::MicroColor,
        Self::Brightness,
        Self::Saturation,
        Self::Crossover,
        Self::Unchanged,
    ];

    /// In `snake_case`, as records and scripts name it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Remove => "remove",
            Self::Reorder => "reorder",
            Self::Scale => "scale",
            Self::Rotate => "rotate",
            Self::AdjacentSwap => "adjacent_swap",
            Self::Merge => "merge",
            Self::Clone => "clone",
            Self::SwapColors => "swap_colors",
            Self::Offset => "offset",
            Self::MovePoint => "move_point",
            Self::MediumMove => "medium_move",
            Self::MicroMove => "micro_move",
            Self::ChangeColor => "change_color",
            Self::MicroColor => "micro_color",
            Self::Brightness => "brightness",
            Self::Saturation => "saturation",
            Self::Crossover => "crossover",
            Self::Unchanged => "unchanged",
        }
    }

    /// Every name, in code order.
    #[must_use]
    pub fn names() -> [&'static str; OPERATOR_COUNT] {
        Self::ALL.map(Self::name)
    }

    /// The shader constant holding this code, e.g. `OP_ADJACENT_SWAP`.
    fn wgsl_constant(self) -> String {
        format!("OP_{}", self.name().to_ascii_uppercase())
    }
}

/// `OPERATOR_COUNT` and one `OP_*` constant per operator, for the generated
/// prelude of the bundled shaders.
pub(super) fn wgsl_constants() -> String {
    let mut source = format!(
        "const OPERATOR_COUNT: u32 = {OPERATOR_COUNT}u; // operators::OPERATOR_COUNT\n"
    );
    for op in Operator::ALL {
        writeln!(source, "const {}: u32 = {}u;", op.wgsl_constant(), op as u32)
            .expect("writing to a String cannot fail");
    }
    source
}

/// Cumulative counts per operator code since the chains were initialized.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct OperatorStats {
    pub offered: [u64; OPERATOR_COUNT],
    pub improved: [u64; OPERATOR_COUNT],
    pub neutral: [u64; OPERATOR_COUNT],
}

/// Host totals of the GPU's wrapping u32 counters. Each collected batch folds
/// in its snapshot as the wrapping difference from the previous one; a batch
/// adds at most chains x λ x iterations (about 2^24) per counter, far below
/// 2^32, so the difference is exact even after a counter wraps.
#[derive(Clone, Debug)]
pub(super) struct OperatorTally {
    last: [u32; OPERATOR_STATS_WORDS],
    totals: OperatorStats,
}

impl Default for OperatorTally {
    /// The state of freshly zeroed GPU counters.
    fn default() -> Self {
        Self { last: [0; OPERATOR_STATS_WORDS], totals: OperatorStats::default() }
    }
}

impl OperatorTally {
    /// Fold in the counters as one batch left them.
    pub fn fold(&mut self, words: &[u32; OPERATOR_STATS_WORDS]) {
        let totals = [&mut self.totals.offered, &mut self.totals.improved, &mut self.totals.neutral];
        for (kind, counts) in totals.into_iter().enumerate() {
            for (code, total) in counts.iter_mut().enumerate() {
                let word = kind * OPERATOR_COUNT + code;
                *total += u64::from(words[word].wrapping_sub(self.last[word]));
            }
        }
        self.last = *words;
    }

    pub fn totals(&self) -> &OperatorStats {
        &self.totals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_follow_the_table_order_and_names_are_distinct() {
        for (code, op) in Operator::ALL.into_iter().enumerate() {
            assert_eq!(op as usize, code, "{op:?}");
        }
        let names = Operator::names();
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), OPERATOR_COUNT);
        assert!(wgsl_constants().contains("const OP_ADJACENT_SWAP: u32 = 5u;\n"));
        assert!(wgsl_constants().contains("const OP_UNCHANGED: u32 = 18u;\n"));
    }

    /// Every code has a writer: mutate stores each mutation code and crossover;
    /// select only reads them.
    #[test]
    fn mutate_writes_every_operator_code() {
        let mutate = include_str!("../shaders/mutate.wgsl");
        for op in Operator::ALL {
            let constant = op.wgsl_constant();
            assert!(
                mutate.contains(&format!("= {constant};")),
                "mutate.wgsl never stores {constant}"
            );
        }
    }

    fn words(offered: u32, improved: u32, neutral: u32) -> [u32; OPERATOR_STATS_WORDS] {
        let mut words = [0; OPERATOR_STATS_WORDS];
        words[0] = offered;
        words[OPERATOR_COUNT] = improved;
        words[2 * OPERATOR_COUNT + OPERATOR_COUNT - 1] = neutral;
        words
    }

    #[test]
    fn tally_adds_the_difference_from_the_previous_batch() {
        let mut tally = OperatorTally::default();
        tally.fold(&words(100, 3, 1));
        tally.fold(&words(250, 3, 4));
        let totals = tally.totals();
        assert_eq!(totals.offered[0], 250);
        assert_eq!(totals.improved[0], 3);
        assert_eq!(totals.neutral[OPERATOR_COUNT - 1], 4);
        assert_eq!(totals.offered[1..], [0; OPERATOR_COUNT - 1]);
    }

    #[test]
    fn tally_is_exact_across_a_counter_wrap() {
        let mut tally = OperatorTally::default();
        let near = u32::MAX - 10;
        tally.fold(&words(near, 0, 0));
        // 21 more offspring: the GPU word wrapped to 10.
        tally.fold(&words(10, 0, 0));
        tally.fold(&words(20, 0, 0));
        assert_eq!(tally.totals().offered[0], u64::from(near) + 21 + 10);
    }
}
