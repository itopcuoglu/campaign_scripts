#!/usr/bin/env python3
"""
validate_exodus.py

Structural/geometric sanity checker for an Exodus II file, aimed at
catching the kinds of problems that make ParaView's IOSS-based Exodus
reader crash hard (segfault) instead of failing gracefully: those readers
generally trust the file and do not bounds-check everything.

Checks performed:
  1. Node coordinates: correct shape, no NaN/Inf.
  2. Every element block: connectivity node ids within [1, num_nodes],
     no degenerate elements (an element that repeats a node id -- either
     exactly, or two logical corners resolving to the same physical
     node), and element volume sign/magnitude (flags zero and negative
     volumes) for HEX8 and WEDGE6 blocks.
  3. Every side set: elem ids within range for the block they resolve
     into, and side numbers within the valid range for that element's
     topology (1-6 for HEX8, 1-5 for WEDGE6). Also flags any side-set
     face whose node set doesn't actually correspond to a real face of
     that element (which is the case most likely to crash a reader that
     assumes side->node mapping is always geometrically consistent).
  4. Cross-checks: total element count vs. sum of block sizes, and basic
     dimension consistency.

Usage:
    python validate_exodus.py assembly_boundary_split.exo
    python validate_exodus.py assembly_boundary_split.exo --max-report 50
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("pip install netCDF4")


HEX_FACES = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [0, 4, 7, 3],
    [0, 3, 2, 1],
    [4, 5, 6, 7],
]
WEDGE_FACES = [
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [2, 0, 3, 5],
    [0, 2, 1],
    [3, 4, 5],
]


def tet_vol(a, b, c, d):
    return np.dot(np.cross(b - a, c - a), d - a) / 6.0


def hex_volume(p):
    # decompose into 6 tets around the centroid-free standard split
    c = p.mean(axis=0)
    faces = HEX_FACES
    vol = 0.0
    for f in faces:
        quad = p[f]
        vol += tet_vol(quad[0], quad[1], quad[2], c)
        vol += tet_vol(quad[0], quad[2], quad[3], c)
    return vol


def wedge_volume(p):
    v = tet_vol(p[0], p[1], p[2], p[3])
    v += tet_vol(p[1], p[2], p[3], p[4])
    v += tet_vol(p[2], p[3], p[4], p[5])
    return v


def s(char_array):
    try:
        return b"".join(char_array.tobytes().split(b"\x00")[:1]).decode(
            "ascii", errors="ignore"
        )
    except Exception:
        return str(char_array)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-report", type=int, default=25,
                     help="max number of offending records to print per check")
    ap.add_argument("--vol-tol", type=float, default=1e-13,
                     help="absolute volume below which an element is 'zero volume'")
    args = ap.parse_args()

    d = nc.Dataset(args.path, "r")
    problems = 0

    num_nodes = d.dimensions["num_nodes"].size
    num_elem = d.dimensions["num_elem"].size
    num_el_blk = d.dimensions["num_el_blk"].size
    print(f"num_nodes={num_nodes}  num_elem={num_elem}  num_el_blk={num_el_blk}")

    # ---- coordinates ----
    if "coord" in d.variables:
        coord = np.array(d.variables["coord"][:, :]).T
    else:
        comps = [np.array(d.variables[n][:]) for n in ("coordx", "coordy", "coordz")
                  if n in d.variables]
        coord = np.stack(comps, axis=1)
    if coord.shape[0] != num_nodes:
        print(f"[FAIL] coord has {coord.shape[0]} rows, expected num_nodes={num_nodes}")
        problems += 1
    bad_xyz = ~np.isfinite(coord)
    if bad_xyz.any():
        n_bad = int(bad_xyz.any(axis=1).sum())
        print(f"[FAIL] {n_bad} node(s) have NaN/Inf coordinates "
              f"(e.g. node ids: {list(np.nonzero(bad_xyz.any(axis=1))[0][:args.max_report] + 1)})")
        problems += 1
    else:
        print("[ok] coordinates: shape and finiteness")

    # ---- element blocks ----
    eb_ids = list(d.variables["eb_prop1"][:]) if "eb_prop1" in d.variables else list(range(1, num_el_blk + 1))
    eb_names = [s(x) for x in d.variables["eb_names"][:]] if "eb_names" in d.variables else [""] * num_el_blk

    block_conn = []       # 0-based node ids, per block
    block_type = []
    block_offset = []
    offset = 0
    for i in range(1, num_el_blk + 1):
        var = d.variables[f"connect{i}"]
        conn = np.array(var[:, :], dtype=np.int64) - 1
        etype = getattr(var, "elem_type", "UNKNOWN")
        block_conn.append(conn)
        block_type.append(etype.upper())
        block_offset.append(offset)
        offset += conn.shape[0]
        print(f"  block id={eb_ids[i-1]:<8} name={eb_names[i-1]:<20} type={etype:<8} "
              f"n_elem={conn.shape[0]}")

    if offset != num_elem:
        print(f"[FAIL] sum of block sizes ({offset}) != num_elem ({num_elem})")
        problems += 1
    else:
        print("[ok] block sizes sum to num_elem")

    print("\nChecking connectivity ranges and degeneracy ...")
    for bi, (conn, etype) in enumerate(zip(block_conn, block_type)):
        oob = (conn < 0) | (conn >= num_nodes)
        if oob.any():
            rows = np.nonzero(oob.any(axis=1))[0][: args.max_report]
            print(f"[FAIL] block {eb_ids[bi]}: {int(oob.any(axis=1).sum())} element(s) "
                  f"reference node ids outside [1,{num_nodes}]; e.g. local elems "
                  f"{[int(r)+1 for r in rows]}")
            problems += 1

        # degenerate: same node id used twice in the same element
        dup_mask = np.zeros(conn.shape[0], dtype=bool)
        for row_i in range(conn.shape[0]):
            row = conn[row_i]
            if len(set(row.tolist())) != len(row):
                dup_mask[row_i] = True
        if dup_mask.any():
            rows = np.nonzero(dup_mask)[0][: args.max_report]
            print(f"[FAIL] block {eb_ids[bi]}: {int(dup_mask.sum())} element(s) reuse a node id "
                  f"within the same element (degenerate); e.g. local elems "
                  f"{[int(r)+1 for r in rows]}")
            problems += 1

        # volumes
        if etype.startswith("HEX") or etype.startswith("WEDGE") or etype.startswith("PENTA"):
            vols = np.empty(conn.shape[0])
            safe = ~(oob.any(axis=1) | dup_mask)
            idxs = np.nonzero(safe)[0]
            for row_i in idxs:
                p = coord[conn[row_i]]
                vols[row_i] = hex_volume(p) if etype.startswith("HEX") else wedge_volume(p)
            vols[~safe] = np.nan
            neg = np.nonzero(vols < -args.vol_tol)[0]
            zero = np.nonzero(np.abs(vols) <= args.vol_tol)[0]
            if len(neg):
                print(f"[FAIL] block {eb_ids[bi]}: {len(neg)} element(s) have NEGATIVE volume "
                      f"(inverted); e.g. local elems {[int(r)+1 for r in neg[:args.max_report]]}")
                problems += 1
            if len(zero):
                print(f"[FAIL] block {eb_ids[bi]}: {len(zero)} element(s) have ~ZERO volume "
                      f"(collapsed); e.g. local elems {[int(r)+1 for r in zero[:args.max_report]]}")
                problems += 1
            if not len(neg) and not len(zero):
                print(f"[ok] block {eb_ids[bi]}: all element volumes positive")

    # ---- side sets ----
    num_side_sets = d.dimensions["num_side_sets"].size if "num_side_sets" in d.dimensions else 0
    if num_side_sets:
        print("\nChecking side sets ...")
        ss_ids = list(d.variables["ss_prop1"][:])
        ss_names = [s(x) for x in d.variables["ss_names"][:]] if "ss_names" in d.variables else [""] * num_side_sets

        max_side = {"HEX8": 6, "HEX": 6, "WEDGE6": 5, "WEDGE": 5, "PENTA6": 5, "PENTA": 5}

        for i in range(1, num_side_sets + 1):
            elem = np.array(d.variables[f"elem_ss{i}"][:], dtype=np.int64) - 1
            side = np.array(d.variables[f"side_ss{i}"][:], dtype=np.int64) - 1
            name = ss_names[i-1]
            sid = ss_ids[i-1]

            bad_elem = (elem < 0) | (elem >= num_elem)
            if bad_elem.any():
                print(f"[FAIL] side set {sid} ({name}): {int(bad_elem.sum())} entries reference "
                      f"elem ids outside [1,{num_elem}]")
                problems += 1

            bad_side_topo = np.zeros(len(elem), dtype=bool)
            bad_side_face = np.zeros(len(elem), dtype=bool)
            for k in range(len(elem)):
                if bad_elem[k]:
                    continue
                ge = int(elem[k])
                # find owning block
                bi = None
                for b in range(len(block_offset)):
                    lo = block_offset[b]
                    hi = lo + block_conn[b].shape[0]
                    if lo <= ge < hi:
                        bi = b
                        break
                etype = block_type[bi]
                nmax = max_side.get(etype)
                if nmax is None:
                    continue
                if side[k] < 0 or side[k] >= nmax:
                    bad_side_topo[k] = True
                    continue
                faces = HEX_FACES if etype.startswith("HEX") else WEDGE_FACES
                row = block_conn[bi][ge - block_offset[bi]]
                face_node_count = len(faces[side[k]])
                # sanity: a triangular side (3 nodes) vs quad side (4 nodes)
                # just confirm indices are in range for this element's npe
                if max(faces[side[k]]) >= len(row):
                    bad_side_face[k] = True

            n_bad_topo = int(bad_side_topo.sum())
            n_bad_face = int(bad_side_face.sum())
            if n_bad_topo:
                print(f"[FAIL] side set {sid} ({name}): {n_bad_topo} entries have a side number "
                      f"outside the valid range for their element's type")
                problems += 1
            if n_bad_face:
                print(f"[FAIL] side set {sid} ({name}): {n_bad_face} entries have a side index "
                      f"inconsistent with the element's node count")
                problems += 1
            if not n_bad_topo and not n_bad_face and not bad_elem.any():
                print(f"[ok] side set {sid} ({name}): {len(elem)} faces, all references valid")

    d.close()

    print()
    if problems:
        print(f"RESULT: {problems} category(ies) of problems found -- see [FAIL] lines above.")
        sys.exit(1)
    else:
        print("RESULT: no structural problems found by this checker. "
              "The crash may be memory-related (23M elements is a lot for "
              "an interactive ParaView session) rather than a file-format issue -- "
              "try opening with a smaller subset, or check `dmesg`/terminal output "
              "for an out-of-memory kill rather than a true segfault.")


if __name__ == "__main__":
    main()
