// Declarations and tile constants come from the checked ordinary C source.
@group(0) @binding(0) var<storage, read> working_states: array<DrawingState>;
@group(0) @binding(1) var<storage, read_write> compact_prefix: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect_args: array<u32, 3>;
@group(1) @binding(0) var<uniform> params: Params;

fn consumed_axis(lo: u32, hi: u32, extent: u32, tile: u32) -> u32 {
    if lo >= extent {
        return 0u;
    }
    let start = lo / tile;
    let end = min((hi + tile - 1u) / tile, (extent + tile - 1u) / tile);
    // Interior zero/inverted boxes can consume a tile. Do not compare lo/hi.
    if end <= start {
        return 0u;
    }
    return end - start;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let count = params.chain_count_param * params.lambda;
    var total = 0u;
    compact_prefix[0] = 0u;
    // Host preflight bounds count, every sum and the padded 2D dispatch in u32.
    for (var i = 0u; i < count; i++) {
        let lo = working_states[i].fitness_bits;
        let hi = working_states[i].stagnation_counter;
        let nx = consumed_axis(lo & 65535u, hi & 65535u, params.image_width, WG_X);
        let ny = consumed_axis(lo >> 16u, hi >> 16u, params.image_height, WG_Y);
        total += nx * ny;
        compact_prefix[i + 1u] = total;
    }
    var x = 0u;
    var y = 1u;
    if total != 0u {
        y = total / 65535u + select(0u, 1u, total % 65535u != 0u);
        x = total / y + select(0u, 1u, total % y != 0u);
    }
    indirect_args[0] = x;
    indirect_args[1] = y;
    indirect_args[2] = 1u;
}
