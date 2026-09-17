#!/usr/bin/env python3
"""
hex_to_prism.py
================

Convert an all-hexahedral Exodus II mesh into an all-wedge (triangular-prism)
mesh, splitting every HEX8 into two 6-node prisms (WEDGE6) along the
"extrusion" axis of the hex -- and correctly carrying side sets, node sets,
and block IDs/names through the conversion.

This version talks directly to the underlying netCDF file (Exodus II is
just netCDF with a fixed schema) so that face-based side sets, which
general-purpose mesh libraries like meshio don't round-trip reliably, come
through intact.

ASSUMPTION (true for essentially every boundary-layer / swept hex mesh):
    The mesh was built by extruding a quad surface mesh along its outward
    normal. Under the standard HEX8 node ordering that such tools produce,
    local nodes 0-3 form one cap of the hex and local nodes 4-7 form the
    other cap, with node (i+4) lying "above" node i along the extrusion
    direction. Splitting every hex with the SAME local diagonal (0-2 on the
    bottom cap, 4-6 on the top cap) is then automatically conforming
    mesh-wide: the 4 lateral faces are left as full, unsplit quads, and
    along the extrusion direction layer L's top cap is literally the same
    4 global nodes (same local order) as layer L+1's bottom cap.

If that assumption doesn't hold for your mesh, pass --auto-detect together
with --direction dx dy dz so each hex picks whichever pair of opposite
faces best aligns with that direction before splitting.

Requirements:
    pip install netCDF4 numpy

Usage:
    python hex_to_prism.py input.exo output.exo
    python hex_to_prism.py input.exo output.exo --auto-detect --direction 0 0 1

Limitations (called out again inline where relevant):
  * Only HEX8 element blocks are split; other block types are copied through
    unchanged (and their elements still get correctly renumbered wherever
    global element numbering is used, e.g. in side sets).
  * Side sets are assumed to reference only hex element faces (true for an
    all-hex mesh). Distribution factors, if present, are carried through by
    matching node identity, not by a hard-coded ordering, so this is exact
    even when a face's node order is reversed between the two prisms.
  * Transient results (nodal/element time-history variables) are NOT
    duplicated/remapped; if your file has them, re-map or drop them
    separately. elem_map / elem_num_map are regenerated as a simple
    sequential 1..N map rather than preserving custom numbering.
"""

import argparse
import re
import sys

import numpy as np
import netCDF4 as nc


# --------------------------------------------------------------------------
# Topology tables (all node indices are LOCAL and 0-indexed)
# --------------------------------------------------------------------------

# Exodus standard HEX8 side -> local node ids (order matters for dist. factors)
HEX_FACE_TO_NODES = {
    1: (0, 1, 5, 4),
    2: (1, 2, 6, 5),
    3: (2, 3, 7, 6),
    4: (0, 3, 7, 4),
    5: (0, 1, 2, 3),   # bottom cap
    6: (4, 5, 6, 7),   # top cap
}

# Exodus standard WEDGE6 side -> local node ids, given our fixed wedge
# connectivity order (w0..w5) == (n0,n1,n2,n4,n5,n6) for prism A
#                              == (n0,n2,n3,n4,n6,n7) for prism B
WEDGE_SIDE_NODES = {
    1: (0, 1, 4, 3),
    2: (1, 2, 5, 4),
    3: (0, 3, 5, 2),
    4: (0, 2, 1),   # bottom triangle
    5: (3, 4, 5),   # top triangle
}

# old hex side -> list of (which prism, new wedge side id) it maps onto.
# Sides 5/6 (the caps) split across both prisms; the 4 lateral sides map
# onto exactly one prism each.
SIDE_MAP = {
    1: [('A', 1)],
    2: [('A', 2)],
    3: [('B', 2)],
    4: [('B', 3)],
    5: [('A', 4), ('B', 4)],
    6: [('A', 5), ('B', 5)],
}

# The 3 possible opposite-face pairings of a hex, used only by --auto-detect
FACE_PAIRS = [
    ((0, 1, 2, 3), (4, 5, 6, 7)),
    ((0, 1, 5, 4), (3, 2, 6, 7)),
    ((1, 2, 6, 5), (0, 3, 7, 4)),
]


def split_hex(conn0):
    """conn0: 8 global node ids (0-indexed), canonical local order.
    Returns (prism_A, prism_B) as 6-tuples of global node ids."""
    n0, n1, n2, n3, n4, n5, n6, n7 = conn0
    return (n0, n1, n2, n4, n5, n6), (n0, n2, n3, n4, n6, n7)


def reorder_for_axis(conn0, coords, direction):
    """Reorder a hex's 8 node ids so the chosen extrusion axis (whichever of
    the 3 opposite-face pairs best aligns with `direction`) occupies the
    canonical bottom(0-3)/top(4-7) slots."""
    pts = coords[list(conn0)]
    best, best_score = None, -1.0
    for bot, top in FACE_PAIRS:
        bc, tc = pts[list(bot)].mean(0), pts[list(top)].mean(0)
        axis = tc - bc
        norm = np.linalg.norm(axis)
        if norm < 1e-14:
            continue
        score = abs(np.dot(axis, direction) / norm)
        if score > best_score:
            best_score = score
            b, t = bot, top
            if np.dot(axis, direction) < 0:
                b, t = top, bot
            best = tuple(conn0[i] for i in b) + tuple(conn0[i] for i in t)
    return best


# --------------------------------------------------------------------------
# Exodus I/O helpers
# --------------------------------------------------------------------------

def read_coords(src):
    n = len(src.dimensions['num_nodes'])
    ndim = len(src.dimensions['num_dim'])
    if 'coord' in src.variables:
        c = src.variables['coord'][:]  # (num_dim, num_nodes)
        return np.asarray(c).T
    coords = np.zeros((n, ndim))
    for k, name in enumerate(['coordx', 'coordy', 'coordz'][:ndim]):
        coords[:, k] = src.variables[name][:]
    return coords


def discover_indices(src, pattern):
    """Find all `elem_ss3`-style variables and return their numeric
    suffixes, sorted. Used instead of trusting dimension counts, which can
    silently disagree with which variables actually exist in real files."""
    rx = re.compile(pattern)
    found = set()
    for name in src.variables:
        m = rx.fullmatch(name)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def get_block_info(src):
    """Return list of dicts, one per element block, in file order."""
    blocks = []
    block_indices = discover_indices(src, r'connect(\d+)')
    if 'num_el_blk' in src.dimensions and len(src.dimensions['num_el_blk']) != len(block_indices):
        print(f"WARNING: num_el_blk dimension says {len(src.dimensions['num_el_blk'])} blocks "
              f"but found {len(block_indices)} connect* variables ({block_indices}). "
              f"Using the variables actually present.")
    for i in block_indices:
        var = src.variables[f'connect{i}']
        elem_type = var.elem_type.upper()
        conn = np.asarray(var[:]) - 1  # -> 0-indexed
        elem_dim, node_dim = var.dimensions
        blocks.append(dict(
            idx=i, elem_type=elem_type, conn=conn,
            elem_dim_name=elem_dim, node_dim_name=node_dim,
            is_hex=elem_type.startswith('HEX'),
        ))
    return blocks


def build_global_id_map(blocks):
    """For every OLD global (1-indexed) element id, compute its NEW global
    id(s), and for hex blocks the split connectivity. Also returns the total
    new element count and per-block new element counts."""
    id_map = {}          # old_global_id -> tuple of new_global_id(s)
    hex_conn0 = {}        # old_global_id -> original 0-indexed local conn (hex only)

    old_offset = 0
    new_offset = 0
    new_counts = []
    for b in blocks:
        n_el = b['conn'].shape[0]
        if b['is_hex']:
            for j in range(n_el):
                old_id = old_offset + j + 1
                new_a = new_offset + 2 * j + 1
                new_b = new_offset + 2 * j + 2
                id_map[old_id] = (new_a, new_b)
                hex_conn0[old_id] = tuple(int(x) for x in b['conn'][j])
            new_counts.append(2 * n_el)
            new_offset += 2 * n_el
        else:
            for j in range(n_el):
                old_id = old_offset + j + 1
                id_map[old_id] = (new_offset + j + 1,)
            new_counts.append(n_el)
            new_offset += n_el
        old_offset += n_el

    return id_map, hex_conn0, new_counts, new_offset


def convert_side_set(src, ss_idx, id_map, hex_conn0, new_conn):
    """Remap one side set. Returns (new_elem_list, new_side_list, new_df_list_or_None)."""
    elem_var = src.variables[f'elem_ss{ss_idx}']
    side_var = src.variables[f'side_ss{ss_idx}']
    elems = np.asarray(elem_var[:])
    sides = np.asarray(side_var[:])

    missing = [int(e) for e in elems if int(e) not in hex_conn0]
    if missing:
        raise RuntimeError(
            f"Side set index {ss_idx} references {len(missing)} element(s) that are not "
            f"HEX8 elements (e.g. old element id {missing[0]}). This script only knows how "
            f"to split hex faces; if this mesh isn't purely hex, this side set needs "
            f"different handling."
        )

    df_name = f'dist_fact_ss{ss_idx}'
    has_df = df_name in src.variables
    df = np.asarray(src.variables[df_name][:]) if has_df else None

    new_elems, new_sides = [], []
    new_dfs = [] if has_df else None

    df_pos = 0  # running offset into the flat df array (4 values per hex face)
    for e, s in zip(elems, sides):
        e = int(e)
        s = int(s)
        n_df_here = len(HEX_FACE_TO_NODES[s])  # always 4 for a hex face
        old_df = df[df_pos:df_pos + n_df_here] if has_df else None
        df_pos += n_df_here

        old_node_order = [hex_conn0[e][p] for p in HEX_FACE_TO_NODES[s]]

        for prism, new_side_id in SIDE_MAP[s]:
            new_elem_id = id_map[e][0] if prism == 'A' else id_map[e][1]
            conn = new_conn[e][0] if prism == 'A' else new_conn[e][1]
            new_node_order = [conn[p] for p in WEDGE_SIDE_NODES[new_side_id]]

            new_elems.append(new_elem_id)
            new_sides.append(new_side_id)

            if has_df:
                # match by node identity so this is correct even when a
                # face's node order is reversed between the two prisms, and
                # even for the split (triangular) sides which only keep 3
                # of the original 4 node values.
                perm = [old_node_order.index(nid) for nid in new_node_order]
                new_dfs.extend(old_df[k] for k in perm)

    # Hard validation: never let an out-of-range side or element id reach
    # the output file silently (this is what caused the ParaView "invalid
    # face index 6" error -- catch it here instead).
    bad_sides = [s for s in new_sides if not (1 <= s <= 5)]
    if bad_sides:
        raise RuntimeError(f"Side set index {ss_idx}: produced invalid wedge side id(s) {set(bad_sides)}; "
                            f"this indicates a bug -- please report it.")

    return new_elems, new_sides, new_dfs


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------

def convert(input_path, output_path, auto_detect=False, direction=None):
    src = nc.Dataset(input_path, 'r')
    fmt = src.data_model
    dst = nc.Dataset(output_path, 'w', format=fmt)

    blocks = get_block_info(src)
    if not any(b['is_hex'] for b in blocks):
        sys.exit("No HEX8 element blocks found in the input mesh.")

    coords = read_coords(src) if auto_detect else None
    direction_arr = np.asarray(direction, dtype=float) if direction else None

    # --- split hex connectivity -------------------------------------------------
    id_map, hex_conn0, new_counts, total_new_elem = build_global_id_map(blocks)

    new_conn = {}  # old_global_id -> (prismA tuple, prismB tuple), 0-indexed
    for old_id, conn0 in hex_conn0.items():
        if auto_detect:
            ordered = reorder_for_axis(conn0, coords, direction_arr)
            if ordered is None:
                raise RuntimeError(f"Degenerate hex (old elem id {old_id}) during auto-detect.")
        else:
            ordered = conn0
        new_conn[old_id] = split_hex(ordered)

    n_hex_total = len(hex_conn0)
    print(f"Splitting {n_hex_total} hexahedra into {2 * n_hex_total} triangular prisms...")

    # --- copy plain dimensions (everything except the ones we override) --------
    override_dims = {'num_elem'}
    for b in blocks:
        override_dims.add(b['elem_dim_name'])
        if b['is_hex']:
            override_dims.add(b['node_dim_name'])
    ss_indices = discover_indices(src, r'elem_ss(\d+)')
    if 'num_side_sets' in src.dimensions and len(src.dimensions['num_side_sets']) != len(ss_indices):
        print(f"WARNING: num_side_sets dimension says {len(src.dimensions['num_side_sets'])} side sets "
              f"but found {len(ss_indices)} elem_ss* variables ({ss_indices}). "
              f"Using the variables actually present.")
    for i in ss_indices:
        if f'side_ss{i}' in src.variables:
            override_dims.add(src.variables[f'side_ss{i}'].dimensions[0])
        if f'dist_fact_ss{i}' in src.variables:
            override_dims.add(src.variables[f'dist_fact_ss{i}'].dimensions[0])

    for name, dim in src.dimensions.items():
        if name in override_dims:
            continue
        dst.createDimension(name, (None if dim.isunlimited() else len(dim)))

    dst.createDimension('num_elem', total_new_elem)
    for b, new_n in zip(blocks, new_counts):
        dst.createDimension(b['elem_dim_name'], new_n)
        if b['is_hex']:
            dst.createDimension(b['node_dim_name'], 6)

    # --- copy global attributes --------------------------------------------------
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

    # --- variables to skip in the generic copy pass (handled specially) --------
    skip_vars = set()
    for b in blocks:
        skip_vars.add(f"connect{b['idx']}")
    skip_vars |= {'elem_map', 'elem_num_map'}
    for i in ss_indices:
        skip_vars |= {f'elem_ss{i}', f'side_ss{i}', f'dist_fact_ss{i}'}

    # --- generic copy of all untouched variables --------------------------------
    for name, var in src.variables.items():
        if name in skip_vars:
            continue
        new_var = dst.createVariable(name, var.dtype, var.dimensions)
        new_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        new_var[:] = var[:]

    # --- regenerate elem_num_map / elem_map as simple sequential maps ----------
    if 'elem_num_map' in src.variables:
        v = dst.createVariable('elem_num_map', src.variables['elem_num_map'].dtype, ('num_elem',))
        v[:] = np.arange(1, total_new_elem + 1)
    if 'elem_map' in src.variables:
        v = dst.createVariable('elem_map', src.variables['elem_map'].dtype, ('num_elem',))
        v[:] = np.arange(1, total_new_elem + 1)

    # --- write split/pass-through connectivity ----------------------------------
    old_offset = 0
    for b in blocks:
        n_el = b['conn'].shape[0]
        if b['is_hex']:
            new_data = np.empty((2 * n_el, 6), dtype=b['conn'].dtype)
            for j in range(n_el):
                old_id = old_offset + j + 1
                a, bb = new_conn[old_id]
                new_data[2 * j] = a
                new_data[2 * j + 1] = bb
            var = dst.createVariable(f"connect{b['idx']}", b['conn'].dtype,
                                      (b['elem_dim_name'], b['node_dim_name']))
            var.elem_type = "WEDGE6"
            var[:] = new_data + 1  # back to 1-indexed
        else:
            var = dst.createVariable(f"connect{b['idx']}", b['conn'].dtype,
                                      (b['elem_dim_name'], b['node_dim_name']))
            var.elem_type = src.variables[f"connect{b['idx']}"].elem_type
            var[:] = b['conn'] + 1
        old_offset += n_el

    # --- side sets ---------------------------------------------------------------
    for i in ss_indices:
        new_elems, new_sides, new_dfs = convert_side_set(src, i, id_map, hex_conn0, new_conn)

        elem_dim = f'num_side_ss{i}'
        dst.createDimension(elem_dim, len(new_elems))
        v = dst.createVariable(f'elem_ss{i}', src.variables[f'elem_ss{i}'].dtype, (elem_dim,))
        v[:] = new_elems
        v2 = dst.createVariable(f'side_ss{i}', src.variables[f'side_ss{i}'].dtype, (elem_dim,))
        v2[:] = new_sides

        if new_dfs is not None:
            df_dim = f'num_df_ss{i}'
            dst.createDimension(df_dim, len(new_dfs))
            v3 = dst.createVariable(f'dist_fact_ss{i}', src.variables[f'dist_fact_ss{i}'].dtype, (df_dim,))
            v3[:] = new_dfs

    src.close()
    dst.close()
    print(f"Wrote {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input Exodus file (all-hex mesh)")
    ap.add_argument("output", help="output Exodus file (all-wedge mesh)")
    ap.add_argument("--auto-detect", action="store_true",
                     help="pick the extrusion axis per-hex instead of assuming local nodes 0-3/4-7")
    ap.add_argument("--direction", nargs=3, type=float, default=None,
                     help="reference direction (dx dy dz) used with --auto-detect")
    args = ap.parse_args()

    if args.auto_detect and args.direction is None:
        ap.error("--auto-detect requires --direction dx dy dz")

    convert(args.input, args.output, auto_detect=args.auto_detect, direction=args.direction)


if __name__ == "__main__":
    main()
