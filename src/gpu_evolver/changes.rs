//! Change records: which polygons an evaluated drawing changed relative to the
//! drawing whose pixels are cached. Binning, raster and the framebuffer update
//! skip pixels that no changed polygon can cover; an unknown record turns those
//! shortcuts off. `src/shaders/changes.wgsl` holds the shader encoding and
//! `offspring_changes.wgsl` its readers; CPU tests compare their constants and
//! the [`GpuChainChange`] layout with this module.
//!
//! The offspring record lives in otherwise unused words of a working
//! [`GpuDrawingState`]. mutate writes it; select, binning and raster read it.
//! Historical shader variants read and write these words in place, so their
//! location is fixed:
//! - `fitness_bits`, `stagnation_counter`: the dirty rectangle's min and max
//!   corners, `x | y << 16` pixels, max exclusive;
//! - `rng_state[1]`, `rng_state[2]`: the changed-index words.
//!
//! The chain record is a [`GpuChainChange`], one per chain in its own buffer.
//! select writes it when it accepts a winner; `update_framebuffers` reads it in
//! the same iteration. Parent states carry no change record.

use std::mem::{offset_of, size_of};

use bytemuck::{Pod, Zeroable};

use super::buffers::{GpuDrawingState, GpuPolygon};

/// Polygon indices one changed-index record can list: two words of two halves.
pub const CHANGED_INDICES: usize = 4;
const _: () = assert!(CHANGED_INDICES == 2 * 2);

/// Byte offset of the dirty rectangle's min corner in a working state.
pub const DIRTY_MIN_OFFSET: u64 = offset_of!(GpuDrawingState, fitness_bits) as u64;
/// Byte offset of the dirty rectangle's max corner in a working state.
pub const DIRTY_MAX_OFFSET: u64 = offset_of!(GpuDrawingState, stagnation_counter) as u64;
/// Byte offset of the two changed-index words in a working state.
pub const CHANGED_WORDS_OFFSET: u64 =
    (offset_of!(GpuDrawingState, rng_state) + size_of::<u32>()) as u64;

/// Pack `[min_x, min_y, max_x, max_y]` pixels (max exclusive) into the two
/// dirty-rectangle words, as the shaders' `pack_dirty_box`.
///
/// # Panics
/// Panics if a coordinate exceeds 16 bits.
#[must_use]
pub fn pack_dirty_box(corners: [u32; 4]) -> [u32; 2] {
    assert!(corners.iter().all(|&v| v <= 0xFFFF), "dirty box {corners:?} exceeds 16 bits");
    [corners[0] | (corners[1] << 16), corners[2] | (corners[3] << 16)]
}

/// Changed-index words. Each holds two halves, `index + 1` in its low and high
/// 16 bits; a zero half lists nothing. A zero first word means unknown, and the
/// second word is meaningful only when the first is non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChangedIndices(pub [u32; 2]);

impl ChangedIndices {
    /// Every shortcut off: every pixel blends.
    pub const UNKNOWN: Self = Self([0, 0]);
}

impl GpuDrawingState {
    /// The offspring dirty rectangle as `[min_x, min_y, max_x, max_y]` pixels.
    #[must_use]
    pub fn dirty_box(&self) -> [u32; 4] {
        let [lo, hi] = [self.fitness_bits, self.stagnation_counter];
        [lo & 0xFFFF, lo >> 16, hi & 0xFFFF, hi >> 16]
    }

    /// Set the offspring dirty rectangle; see [`pack_dirty_box`].
    pub fn set_dirty_box(&mut self, corners: [u32; 4]) {
        [self.fitness_bits, self.stagnation_counter] = pack_dirty_box(corners);
    }

    /// The offspring changed-index words.
    #[must_use]
    pub fn changed_indices(&self) -> ChangedIndices {
        ChangedIndices([self.rng_state[1], self.rng_state[2]])
    }

    pub fn set_changed_indices(&mut self, changed: ChangedIndices) {
        [self.rng_state[1], self.rng_state[2]] = changed.0;
    }
}

/// One chain's record of its last accepted winner (WGSL `ChainChange`): the
/// winner's changed-index words, and the parent versions of those indices for
/// `update_framebuffers`. All zeros is unknown.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuChainChange {
    pub changed: [u32; 2],
    /// Bit k: `parents[k]` holds the parent version of index half k.
    pub parent_mask: u32,
    pub(crate) _pad: u32,
    pub parents: [GpuPolygon; CHANGED_INDICES],
}

pub const GPU_CHAIN_CHANGE_SIZE: usize = size_of::<GpuChainChange>();
const _: () = assert!(GPU_CHAIN_CHANGE_SIZE == 16 + 16 * CHANGED_INDICES);

#[cfg(test)]
mod tests {
    use super::{pack_dirty_box, ChangedIndices, CHANGED_INDICES};
    use crate::gpu_evolver::buffers::GpuDrawingState;
    use crate::settings::{POLYGON_CAPACITY, TILE_MAX_POLYS};
    use bytemuck::Zeroable;

    // CPU reference of changes.wgsl, which has no interpreter here. Wrapping
    // adds and shifts match WGSL u32 arithmetic.
    const NO_CHANGED_INDEX: u32 = 0xFFFF;
    const NO_CHANGES: u32 = u32::MAX;

    fn changed_indices(first: u32, second: u32) -> u32 {
        first.wrapping_add(1) | (second.wrapping_add(1) << 16)
    }

    /// mutate.wgsl's encoding of a list of changed indices: unknown when more
    /// than `CHANGED_INDICES`, `NO_CHANGES` when empty.
    fn encode(indices: &[u32]) -> ChangedIndices {
        if indices.len() > CHANGED_INDICES {
            return ChangedIndices::UNKNOWN;
        }
        if indices.is_empty() {
            return ChangedIndices([NO_CHANGES, 0]);
        }
        let mut padded = [NO_CHANGED_INDEX; CHANGED_INDICES];
        padded[..indices.len()].copy_from_slice(indices);
        let second = if indices.len() > 2 { changed_indices(padded[2], padded[3]) } else { 0 };
        ChangedIndices([changed_indices(padded[0], padded[1]), second])
    }

    fn indices() -> u32 {
        u32::try_from(CHANGED_INDICES).unwrap()
    }

    fn changed_half(words: [u32; 2], k: usize) -> u32 {
        (words[k / 2] >> (16 * (k % 2))) & 0xFFFF
    }

    /// The readers' view: listed indices below `count`, or None when unknown.
    fn decode(changed: ChangedIndices, count: u32) -> Option<Vec<u32>> {
        let [first, second] = changed.0;
        let words = [first, if first == 0 { 0 } else { second }];
        (first != 0).then(|| {
            (0..CHANGED_INDICES)
                .map(|k| changed_half(words, k))
                .filter(|&encoded| encoded != 0 && encoded - 1 < count)
                .map(|encoded| encoded - 1)
                .collect()
        })
    }

    /// changes.wgsl's `changes_shortcut`: whether a tile list of `raw_count`
    /// polygons keeps the shortcut on.
    fn shortcut(changed: ChangedIndices, raw_count: u32) -> bool {
        let listed = if changed.0[1] != 0 { indices() } else { 2 };
        changed.0[0] != 0 && raw_count + listed <= TILE_MAX_POLYS
    }

    #[test]
    fn changed_indices_encode_as_mutate_writes_them() {
        assert_eq!(encode(&[7]), ChangedIndices([8, 0]));
        assert_eq!(encode(&[2, 5]), ChangedIndices([3 | 6 << 16, 0]));
        assert_eq!(encode(&[1, 2, 3]), ChangedIndices([2 | 3 << 16, 4]));
        assert_eq!(encode(&[0, 1, 2, 3]), ChangedIndices([1 | 2 << 16, 3 | 4 << 16]));
        assert_eq!(encode(&[0, 1, 2, 3, 4]), ChangedIndices::UNKNOWN);
        assert_eq!(encode(&[]), ChangedIndices([NO_CHANGES, 0]));
    }

    #[test]
    fn readers_decode_every_listed_index_below_the_drawing_count() {
        let count = u32::try_from(POLYGON_CAPACITY).unwrap();
        for indices in [&[0][..], &[999], &[3, 998], &[4, 0, 999], &[10, 20, 30, 40], &[1998, 1999]] {
            assert_eq!(decode(encode(indices), count).as_deref(), Some(indices));
        }
        // NO_CHANGES names 0xFFFE twice, beyond every drawing.
        assert_eq!(decode(encode(&[]), count), Some(vec![]));
        assert_eq!(decode(ChangedIndices::UNKNOWN, count), None);
        // A stale second word is ignored once the first is unknown.
        assert_eq!(decode(ChangedIndices([0, 5]), count), None);
        // An index the drawing no longer has (a removed tail) is empty.
        assert_eq!(decode(encode(&[3, 9]), 9), Some(vec![3]));
    }

    #[test]
    fn tile_lists_need_room_for_one_polygon_per_listed_index() {
        let two = encode(&[1, 2]);
        assert!(shortcut(two, TILE_MAX_POLYS - 2));
        assert!(!shortcut(two, TILE_MAX_POLYS - 1));
        let four = encode(&[1, 2, 3]);
        assert!(shortcut(four, TILE_MAX_POLYS - 4));
        assert!(!shortcut(four, TILE_MAX_POLYS - 3));
        assert!(!shortcut(ChangedIndices::UNKNOWN, 0));
    }

    #[test]
    fn offspring_accessors_use_the_shader_words() {
        let mut state = GpuDrawingState::zeroed();
        state.set_dirty_box([3, 4, 383, 384]);
        assert_eq!(pack_dirty_box([3, 4, 383, 384]), [0x0004_0003, 0x0180_017F]);
        assert_eq!([state.fitness_bits, state.stagnation_counter], [0x0004_0003, 0x0180_017F]);
        assert_eq!(state.dirty_box(), [3, 4, 383, 384]);
        state.rng_state = [7, 0, 0, 9];
        state.set_changed_indices(encode(&[1, 2, 3]));
        assert_eq!(state.rng_state, [7, 2 | 3 << 16, 4, 9]);
        assert_eq!(state.changed_indices(), encode(&[1, 2, 3]));
    }

    /// The shader constants equal this reference's.
    #[test]
    fn shader_constants_match_the_reference() {
        use crate::gpu_evolver::programs::{bundled_shader_source, BundledShader};
        use wgpu::naga::{front::wgsl, Expression, Literal};
        let module = wgsl::parse_str(&bundled_shader_source(BundledShader::Mutate)).unwrap();
        let value = |name: &str| {
            let (_, constant) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("{name} is not declared"));
            match module.global_expressions[constant.init] {
                Expression::Literal(Literal::U32(value)) => value,
                ref other => panic!("{name} = {other:?}"),
            }
        };
        assert_eq!(value("CHANGED_INDICES"), indices());
        assert_eq!(value("CHANGED_SLOTS"), 2 * indices());
        assert_eq!(value("NO_CHANGED_INDEX"), NO_CHANGED_INDEX);
        assert_eq!(value("NO_CHANGES"), NO_CHANGES);
        assert_eq!(value("TILE_MAX_POLYS"), TILE_MAX_POLYS);
    }
}
