#!/usr/bin/env python3
"""
diagnose_exodus.py

Read-only inspection of an Exodus file's element blocks and side sets,
independent of hex_to_prism.py, to pin down exactly what's stored.

Usage:
    python diagnose_exodus.py input.exo
    python diagnose_exodus.py output.exo --find-elem 3267041
"""
import argparse
import re
import numpy as np
import netCDF4 as nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--find-elem", type=int, default=None,
                     help="print every side-set entry referencing this global element id")
    args = ap.parse_args()

    ds = nc.Dataset(args.path, 'r')

    print(f"=== {args.path} ===")
    print(f"data_model: {ds.data_model}")

    # Element blocks
    block_ids = sorted(int(m.group(1)) for m in
                        (re.fullmatch(r'connect(\d+)', n) for n in ds.variables) if m)
    print(f"\nElement blocks found (connect* variables): {block_ids}")
    if 'num_el_blk' in ds.dimensions:
        print(f"  num_el_blk dimension says: {len(ds.dimensions['num_el_blk'])}")
    for i in block_ids:
        var = ds.variables[f'connect{i}']
        print(f"  connect{i}: elem_type={var.elem_type!r}, shape={var.shape}")

    # Side set name -> id mapping, if available
    id_to_name = {}
    if 'ss_prop1' in ds.variables and 'ss_names' in ds.variables:
        ids = np.asarray(ds.variables['ss_prop1'][:])
        names_raw = ds.variables['ss_names'][:]
        for k, row in enumerate(names_raw):
            name = b''.join(c for c in row.tobytes().split(b'\x00')[:1]).decode(errors='ignore') \
                if hasattr(row, 'tobytes') else ''.join(row).strip('\x00')
            id_to_name[int(ids[k])] = name

    ss_ids = sorted(int(m.group(1)) for m in
                     (re.fullmatch(r'elem_ss(\d+)', n) for n in ds.variables) if m)
    print(f"\nSide sets found (elem_ss* variables): {ss_ids}")
    if 'num_side_sets' in ds.dimensions:
        print(f"  num_side_sets dimension says: {len(ds.dimensions['num_side_sets'])}")

    ss_prop = np.asarray(ds.variables['ss_prop1'][:]) if 'ss_prop1' in ds.variables else None
    total_elem = len(ds.dimensions['num_elem']) if 'num_elem' in ds.dimensions else None

    for pos, i in enumerate(ss_ids):
        elems = np.asarray(ds.variables[f'elem_ss{i}'][:])
        sides = np.asarray(ds.variables[f'side_ss{i}'][:])
        ss_id = int(ss_prop[pos]) if ss_prop is not None and pos < len(ss_prop) else None
        name = id_to_name.get(ss_id, '?')
        print(f"  ss index {i} (id={ss_id}, name={name!r}): "
              f"{len(elems)} entries, elem id range [{elems.min()},{elems.max()}], "
              f"unique side values = {sorted(set(sides.tolist()))}")

        # --- sanity checks that could explain a crash in the viewer ---
        problems = []
        if total_elem is not None and (elems.min() < 1 or elems.max() > total_elem):
            problems.append(f"element id(s) out of range 1..{total_elem}")
        if len(set(zip(elems.tolist(), sides.tolist()))) != len(elems):
            problems.append("duplicate (elem, side) entries")
        elem_types = {ds.variables[n].elem_type.upper() for n in ds.variables if re.fullmatch(r'connect\d+', n)}
        max_valid_side = 5 if any(t.startswith('WEDGE') for t in elem_types) else 6
        if sides.min() < 1 or sides.max() > max_valid_side:
            problems.append(f"side id(s) outside expected range 1..{max_valid_side} "
                             f"(found min={sides.min()}, max={sides.max()})")

        df_name = f'dist_fact_ss{i}'
        if df_name in ds.variables:
            df = np.asarray(ds.variables[df_name][:])
            # expected count: 4 per quad side (1,2,3) + 3 per tri side (4,5) for a wedge-side set
            side_node_counts = {1: 4, 2: 4, 3: 4, 4: 3, 5: 3, 6: 4}  # covers hex(1-6) and wedge(1-5)
            expected = sum(side_node_counts.get(int(s), 0) for s in sides)
            print(f"    dist_fact_ss{i}: length={len(df)}, expected(sum of per-side node counts)={expected}"
                  + ("  <-- MISMATCH" if len(df) != expected else ""))
            if len(df) != expected:
                problems.append("dist_fact length does not match expected node-count sum")
            if np.isnan(df).any() or np.isinf(df).any():
                problems.append("dist_fact contains NaN/Inf")

        if problems:
            print(f"    !! POTENTIAL ISSUES: {problems}")

        if args.find_elem is not None:
            hits = np.where(elems == args.find_elem)[0]
            for h in hits:
                print(f"    -> elem {elems[h]} side {sides[h]} (entry #{h}) in ss index {i} ({name})")

    ds.close()


if __name__ == "__main__":
    main()
