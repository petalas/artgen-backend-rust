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
//! Step rungs: with a step spread (`MutationParams::step_spread_min`/`max`),
//! offspring `i` of a chain scales its step sizes by rung `i % STEP_RUNGS`'s
//! multiplier. For an operator that uses the step ([`Operator::scales_with_step`])
//! mutate also stores `rung + 1` above the code (`OP_RUNG_SHIFT`), and select
//! counts it in per-rung offered, improved and neutral blocks after the
//! operator blocks. Other offspring carry no rung and are not counted there.
//!
//! Counters start at zero when chains are (re)initialized and are not part of
//! the search, so shader variants that never write them leave them meaningless.

use std::fmt::Write as _;

use serde::Serialize;

use crate::settings::STEP_RUNGS;

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
/// indexed by operator code; then the same three blocks indexed by step rung.
pub const OPERATOR_STATS_WORDS: usize = 3 * OPERATOR_COUNT + 3 * STEP_RUNGS;

/// An `offspring_ops` word holds the operator code below this bit and, for an
/// operator that scales with the step, its rung + 1 from this bit up (0: none).
const OP_RUNG_SHIFT: u32 = 8;
const _: () = assert!(OPERATOR_COUNT < 1 << OP_RUNG_SHIFT);

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

    /// Whether the operator's size is multiplied by the offspring's step (the
    /// adaptive scale times its rung's multiplier): offset and the three vertex
    /// moves. Only these count in the rung blocks.
    #[must_use]
    pub const fn scales_with_step(self) -> bool {
        matches!(self, Self::Offset | Self::MovePoint | Self::MediumMove | Self::MicroMove)
    }

    /// Names of the operators that scale with the step, in code order.
    #[must_use]
    pub fn step_scaled_names() -> Vec<&'static str> {
        Self::ALL.into_iter().filter(|op| op.scales_with_step()).map(Self::name).collect()
    }

    /// The shader constant holding this code, e.g. `OP_ADJACENT_SWAP`.
    fn wgsl_constant(self) -> String {
        format!("OP_{}", self.name().to_ascii_uppercase())
    }
}

/// `OPERATOR_COUNT`, one `OP_*` constant per operator, the step rung
/// constants and `op_scales_with_step`, for the generated prelude of the
/// bundled shaders.
pub(super) fn wgsl_constants() -> String {
    let mut source = format!(
        "const OPERATOR_COUNT: u32 = {OPERATOR_COUNT}u; // operators::OPERATOR_COUNT\n\
         const STEP_RUNGS: u32 = {STEP_RUNGS}u; // settings::STEP_RUNGS\n\
         // offspring_ops: operator code below OP_RUNG_SHIFT, rung + 1 above it (operators.rs)\n\
         const OP_RUNG_SHIFT: u32 = {OP_RUNG_SHIFT}u;\n\
         const OP_CODE_MASK: u32 = {}u;\n",
        (1u32 << OP_RUNG_SHIFT) - 1
    );
    for op in Operator::ALL {
        writeln!(source, "const {}: u32 = {}u;", op.wgsl_constant(), op as u32)
            .expect("writing to a String cannot fail");
    }
    let scaled: Vec<String> = Operator::ALL
        .into_iter()
        .filter(|op| op.scales_with_step())
        .map(|op| format!("op == {}", op.wgsl_constant()))
        .collect();
    writeln!(
        source,
        "fn op_scales_with_step(op: u32) -> bool {{ return {}; }} // Operator::scales_with_step",
        scaled.join(" || ")
    )
    .expect("writing to a String cannot fail");
    source
}

/// Cumulative counts per operator code, and per step rung for the operators
/// that scale with the step, since the chains were initialized.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct OperatorStats {
    pub offered: [u64; OPERATOR_COUNT],
    pub improved: [u64; OPERATOR_COUNT],
    pub neutral: [u64; OPERATOR_COUNT],
    pub rung_offered: [u64; STEP_RUNGS],
    pub rung_improved: [u64; STEP_RUNGS],
    pub rung_neutral: [u64; STEP_RUNGS],
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
        let t = &mut self.totals;
        // In buffer order, so each block starts where the previous one ended.
        let blocks: [&mut [u64]; 6] = [
            &mut t.offered,
            &mut t.improved,
            &mut t.neutral,
            &mut t.rung_offered,
            &mut t.rung_improved,
            &mut t.rung_neutral,
        ];
        let mut word = 0;
        for block in blocks {
            for total in block.iter_mut() {
                *total += u64::from(words[word].wrapping_sub(self.last[word]));
                word += 1;
            }
        }
        debug_assert_eq!(word, OPERATOR_STATS_WORDS);
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
        assert!(wgsl_constants().contains(
            "return op == OP_OFFSET || op == OP_MOVE_POINT || op == OP_MEDIUM_MOVE || op == OP_MICRO_MOVE;"
        ));
        assert_eq!(Operator::step_scaled_names(), ["offset", "move_point", "medium_move", "micro_move"]);
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
    fn tally_reads_the_rung_blocks_after_the_operator_blocks() {
        let rungs = 3 * OPERATOR_COUNT;
        let mut words = [0; OPERATOR_STATS_WORDS];
        words[rungs] = 7; // offered, rung 0
        words[rungs + STEP_RUNGS - 1] = 9; // offered, last rung
        words[rungs + STEP_RUNGS + 2] = 2; // improved, rung 2
        words[rungs + 2 * STEP_RUNGS + 3] = 1; // neutral, rung 3
        let mut tally = OperatorTally::default();
        tally.fold(&words);
        let totals = tally.totals();
        assert_eq!(totals.rung_offered, [7, 0, 0, 0, 0, 0, 0, 9]);
        assert_eq!(totals.rung_improved, [0, 0, 2, 0, 0, 0, 0, 0]);
        assert_eq!(totals.rung_neutral, [0, 0, 0, 1, 0, 0, 0, 0]);
        assert_eq!((totals.offered, totals.improved, totals.neutral), Default::default());
        // The rung counters wrap like the operator counters.
        words[rungs] = 3;
        tally.fold(&words);
        assert_eq!(tally.totals().rung_offered[0], 7 + u64::from(3u32.wrapping_sub(7)));
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
