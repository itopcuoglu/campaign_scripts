#!/usr/bin/env python3
"""
hex_to_prism_boundary_layer.py

Convert the hexahedra of an Exodus II mesh into triangular prisms (wedges),
splitting each hex along the diagonal of its wall-normal ("cap") faces, for
every hex found in an unstructured wall-normal "column" that starts at a
user-specified wall side set and is walked, element by element, out to the
outer/far-field boundary.

WHY A COLUMN WALK IS NEEDED
----------------------------
Because the mesh is *not* known to be laid out in a structured i/j/k sense,
a hex touching the wall cannot simply be split in isolation: the split must
propagate consistently, hex to hex, along the (possibly curved / element-
by-element varying) line that starts normal to the wall and travels out to
the domain boundary. Two neighbouring hexes that share a quad face must
agree on how that shared face is diagonalised, otherwise the resulting
wedges will not be conformal. This script:

  1. Builds hex-to-hex face adjacency from the raw connectivity (no
     assumption of IJK structure).
  2. Starting at each element/local-face pair in the given wall side
     set(s), walks the *opposite* face of every hex in turn (opposite
     meaning: the standard Exodus HEX8 face pairs 1-3, 2-4, 5-6) until a
     face is reached that has no neighbour (i.e. it lies on some other,
     outer, boundary of the domain). That whole chain of hexes is one
     "column".
  3. Splits every hex in the column into two triangular prisms (WEDGE6),
     propagating the *same physical diagonal* (identified by global node
     IDs, not local numbering) from the wall face, through every
     intermediate face, to the outer boundary face. This guarantees the
     new wedge faces match up exactly between consecutive elements in the
     column, and leaves every hex face that is NOT part of the column's
     extrusion direction untouched (still a plain quad, shared normally
     with whatever is next to it).
  4. Elements that are never reached by any wall column (e.g. hexes in a
     genuinely structured core far from any tracked wall, if you choose
     not to track every wall face) are left as hexahedra, untouched.
  5. Rebuilds element blocks (hex blocks lose their converted elements,
     new WEDGE6 blocks are appended -- one per source hex block, so
     material/attribute association is preserved), remaps every side set
     (wall sets AND any other side set, e.g. far-field/symmetry/blade
     tip caps) so that old (element, local-face) references now point at
     the correct new hex or pair-of-wedges, and writes a new Exodus file.

REQUIREMENTS
------------
    pip install netCDF4 numpy

This script talks to the Exodus II file as plain netCDF (Exodus II *is* a
netCDF file with a fixed schema), so it does not require SEACAS/exodus.py
to be installed. It only requires that the mesh contains one or more all-
HEX8 element blocks. Non-HEX8 blocks are copied through unmodified.

LIMITATIONS / CAVEATS (please read before trusting production results)
------------------------------------------------------------------------
  * Side-set distribution factors (dist_fact_ss#) are dropped in the
    output (num_df_ss = 0 for every side set). Re-splitting them correctly
    for a face that becomes two triangles is solver-specific; most CFD
    codes recompute df's from geometry on read, but check yours.
  * If a wall column forks (a face shared by more than 2 elements -- non-
    conformal hanging nodes) or loops back on an already-processed
    element, the walk stops at that point and prints a warning; you will
    need to inspect that region by hand. A clean hex mesh (even if
    "unstructured"/unstructured-looking) should not hit this.
  * Only HEX8 -> two WEDGE6 splitting is implemented (the classic
    diagonal split of a hex into two triangular prisms). No new nodes are
    created -- the split only reuses existing hex corner nodes, so
    node coordinates, node sets, and nodal fields need no remapping.
  * Element/side-set "ID properties" beyond the first (ss_prop1/eb_prop1)
    and QA records are not copied. Global/nodal/element *result* variables
    (if this is a results file rather than a plain mesh) are not carried
    over -- run this on the mesh-only Exodus file.

USAGE
-----
    # 1) See what's in the file first (block ids, side set ids/names):
    python hex_to_prism_boundary_layer.py mesh.exo --list

    # 2) Convert, tracking columns from wall side sets 10, 11, 12, 13
    #    (three blades + tower), writing prism block ids offset by +1000:
    python hex_to_prism_boundary_layer.py mesh.exo mesh_prism.exo \\
        --wall-sideset-ids 10 11 12 13

    # or refer to them by name if the file has ss_names:
    python hex_to_prism_boundary_layer.py mesh.exo mesh_prism.exo \\
        --wall-sideset-names blade1_wall blade2_wall blade3_wall tower_wall
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("This script requires the 'netCDF4' python package: pip install netCDF4")


# --------------------------------------------------------------------------
# HEX8 topology (0-based local node indices), standard Exodus/SEACAS order:
#   bottom quad: 0,1,2,3   top quad: 4,5,6,7   vertical edges i -> i+4
# --------------------------------------------------------------------------
HEX_FACES = [
    [0, 1, 5, 4],   # face 1
    [1, 2, 6, 5],   # face 2
    [2, 3, 7, 6],   # face 3
    [0, 4, 7, 3],   # face 4
    [0, 3, 2, 1],   # face 5 (bottom)
    [4, 5, 6, 7],   # face 6 (top)
]
OPPOSITE_FACE = {0: 2, 2: 0, 1: 3, 3: 1, 4: 5, 5: 4}

# Node correspondence across each opposite-face pair, i.e. which local node
# on one face is connected (by a hex edge) to which local node on the other.
_CORR_02 = {0: 3, 3: 0, 1: 2, 2: 1, 4: 7, 7: 4, 5: 6, 6: 5}
_CORR_13 = {1: 0, 0: 1, 2: 3, 3: 2, 5: 4, 4: 5, 6: 7, 7: 6}
_CORR_45 = {0: 4, 4: 0, 1: 5, 5: 1, 2: 6, 6: 2, 3: 7, 7: 3}


def face_correspondence(face_idx):
    if face_idx in (0, 2):
        return _CORR_02
    if face_idx in (1, 3):
        return _CORR_13
    return _CORR_45


# WEDGE6 topology (0-based): bottom tri 0,1,2 ; top tri 3,4,5 ; edges i->i+3
WEDGE_FACES = [
    [0, 1, 4, 3],   # quad side 1
    [1, 2, 5, 4],   # quad side 2
    [2, 0, 3, 5],   # quad side 3
    [0, 2, 1],      # bottom triangle
    [3, 4, 5],      # top triangle
]


# ==========================================================================
# Reading
# ==========================================================================

class ExodusMesh:
    def __init__(self, path):
        self.ds = nc.Dataset(path, "r")
        d = self.ds
        self.num_dim = d.dimensions["num_dim"].size
        self.num_nodes = d.dimensions["num_nodes"].size
        self.num_elem = d.dimensions["num_elem"].size
        self.num_el_blk = d.dimensions["num_el_blk"].size

        # ---- element blocks ----
        self.eb_ids = list(d.variables["eb_prop1"][:])
        self.eb_names = None
        if "eb_names" in d.variables:
            self.eb_names = [self._s(x) for x in d.variables["eb_names"][:]]

        self.block_conn = []      # list of (num_el, npe) 0-based node id arrays
        self.block_elem_type = [] # list of strings
        self.block_offset = []    # global elem index (0-based) where block starts
        offset = 0
        for i in range(1, self.num_el_blk + 1):
            var = d.variables[f"connect{i}"]
            conn = np.array(var[:, :], dtype=np.int64) - 1
            etype = getattr(var, "elem_type", "UNKNOWN")
            self.block_conn.append(conn)
            self.block_elem_type.append(etype)
            self.block_offset.append(offset)
            offset += conn.shape[0]
        assert offset == self.num_elem, "block sizes do not sum to num_elem"

        # flattened per-global-element lookup: which block, which local row
        self.elem_block = np.empty(self.num_elem, dtype=np.int32)
        self.elem_local = np.empty(self.num_elem, dtype=np.int64)
        for bi, conn in enumerate(self.block_conn):
            n = conn.shape[0]
            off = self.block_offset[bi]
            self.elem_block[off:off + n] = bi
            self.elem_local[off:off + n] = np.arange(n)

        # ---- coordinates (support both storage conventions) ----
        if "coord" in d.variables:
            self.coord = np.array(d.variables["coord"][:, :]).T  # (num_nodes, num_dim)
        else:
            comps = []
            for name in ("coordx", "coordy", "coordz"):
                if name in d.variables:
                    comps.append(np.array(d.variables[name][:]))
            self.coord = np.stack(comps, axis=1)

        # ---- side sets ----
        self.num_side_sets = d.dimensions["num_side_sets"].size if "num_side_sets" in d.dimensions else 0
        self.ss_ids = list(d.variables["ss_prop1"][:]) if self.num_side_sets else []
        self.ss_names = None
        if self.num_side_sets and "ss_names" in d.variables:
            self.ss_names = [self._s(x) for x in d.variables["ss_names"][:]]
        self.ss_elem = []   # list of 0-based elem-index arrays, per side set
        self.ss_side = []   # list of 0-based local-face arrays, per side set
        for i in range(1, self.num_side_sets + 1):
            e = np.array(d.variables[f"elem_ss{i}"][:], dtype=np.int64) - 1
            s = np.array(d.variables[f"side_ss{i}"][:], dtype=np.int64) - 1
            self.ss_elem.append(e)
            self.ss_side.append(s)

        # ---- node sets (copied through unmodified) ----
        self.num_node_sets = d.dimensions["num_node_sets"].size if "num_node_sets" in d.dimensions else 0

    @staticmethod
    def _s(char_array):
        return b"".join(c for c in char_array.tobytes().split(b"\x00")[:1]).decode(
            "ascii", errors="ignore"
        ) if hasattr(char_array, "tobytes") else str(char_array)

    def global_conn(self, ge):
        """Full (0-based) node-id connectivity row for global element index ge."""
        bi = self.elem_block[ge]
        li = self.elem_local[ge]
        return self.block_conn[bi][li]

    def close(self):
        self.ds.close()


# ==========================================================================
# Adjacency
# ==========================================================================

def build_hex_adjacency(mesh, hex_block_indices):
    """
    frozenset(4 global node ids) -> list of (global_elem_idx, local_face_idx)
    Only built over hex blocks (mixed-topology meshes: non-hex blocks are
    ignored here, since we do not attempt to walk columns through them).
    """
    adjacency = defaultdict(list)
    for bi in hex_block_indices:
        conn = mesh.block_conn[bi]
        off = mesh.block_offset[bi]
        for li in range(conn.shape[0]):
            ge = off + li
            row = conn[li]
            for f, idxs in enumerate(HEX_FACES):
                key = frozenset(row[idxs].tolist())
                adjacency[key].append((ge, f))
    return adjacency


# ==========================================================================
# Column walk + splitting
# ==========================================================================

def signed_prism_volume(coord, node_ids6):
    p = coord[node_ids6]
    # split prism (0,1,2)-(3,4,5) into 3 tets and sum signed volumes
    def tet_vol(a, b, c, d):
        return np.dot(np.cross(b - a, c - a), d - a) / 6.0
    v = tet_vol(p[0], p[1], p[2], p[3])
    v += tet_vol(p[1], p[2], p[3], p[4])
    v += tet_vol(p[2], p[3], p[4], p[5])
    return v


def split_hex(conn_row, entry_face, diag_global):
    """
    conn_row      : (8,) global node ids of the hex (0-based)
    entry_face    : local face index (0..5) that is being treated as a
                    "cap" of the split (e.g. the wall face, or the face
                    inherited from the previous element in the column)
    diag_global   : a 2-tuple/set of global node ids -- the diagonal of
                    entry_face to cut along (for the very first hex in a
                    column this is chosen arbitrarily; afterwards it is
                    dictated by the previous hex so the shared face
                    matches up)

    Returns: prismA(6,), prismB(6,) global-node-id arrays in WEDGE6 order,
             and exit_face, exit_diag_global for continuing the walk.
    """
    p = HEX_FACES[entry_face]
    corr = face_correspondence(entry_face)
    exit_face = OPPOSITE_FACE[entry_face]

    diag_global = set(diag_global)
    idx0 = next(k for k in range(4) if conn_row[p[k]] in diag_global)
    p_rot = p[idx0:] + p[:idx0]
    q_rot = [corr[x] for x in p_rot]

    prismA_local = [p_rot[0], p_rot[1], p_rot[2], q_rot[0], q_rot[1], q_rot[2]]
    prismB_local = [p_rot[0], p_rot[2], p_rot[3], q_rot[0], q_rot[2], q_rot[3]]

    prismA = conn_row[prismA_local]
    prismB = conn_row[prismB_local]

    exit_diag_global = {conn_row[q_rot[0]], conn_row[q_rot[2]]}
    return prismA, prismB, exit_face, exit_diag_global


def walk_and_split(mesh, adjacency, wall_faces, hex_block_set, coord, verbose=True):
    """
    wall_faces: list of (global_elem_idx, local_face_idx) starting points.

    Returns:
      converted        : dict global_elem_idx -> (prismA(6,), prismB(6,))
      n_columns, n_warnings
    """
    converted = {}
    visited = set()
    n_columns = 0
    n_warnings = 0

    for e0, f0 in wall_faces:
        if e0 in visited:
            continue  # a corner element touched by two wall patches
        n_columns += 1
        conn0 = mesh.global_conn(e0)
        p0 = HEX_FACES[f0]
        diag = {conn0[p0[0]], conn0[p0[2]]}

        cur_e, cur_f = e0, f0
        while True:
            if cur_e in visited:
                print(f"  [warn] column re-entered an already-processed element "
                      f"{cur_e}; stopping this column early.")
                n_warnings += 1
                break
            if mesh.elem_block[cur_e] not in hex_block_set:
                print(f"  [warn] column ran into a non-HEX8 element {cur_e}; stopping.")
                n_warnings += 1
                break

            conn = mesh.global_conn(cur_e)
            prismA, prismB, exit_f, exit_diag = split_hex(conn, cur_f, diag)

            # orientation fix so both wedges have positive volume
            if signed_prism_volume(coord, prismA) < 0:
                prismA = prismA[[0, 2, 1, 3, 5, 4]]
            if signed_prism_volume(coord, prismB) < 0:
                prismB = prismB[[0, 2, 1, 3, 5, 4]]

            converted[cur_e] = (prismA, prismB)
            visited.add(cur_e)

            exit_nodes = frozenset(conn[HEX_FACES[exit_f]].tolist())
            others = [(e, f) for (e, f) in adjacency.get(exit_nodes, []) if e != cur_e]
            if len(others) == 0:
                break  # reached the outer boundary -- column done
            if len(others) > 1:
                print(f"  [warn] non-manifold face beyond element {cur_e} "
                      f"({len(others)} neighbours); stopping this column.")
                n_warnings += 1
                break

            cur_e, cur_f = others[0]
            diag = exit_diag

    if verbose:
        print(f"  columns started : {n_columns}")
        print(f"  hexes converted : {len(converted)}")
        print(f"  warnings        : {n_warnings}")
    return converted


# ==========================================================================
# Rebuild blocks + remap side sets
# ==========================================================================

def rebuild(mesh, converted, wedge_id_offset):
    """
    Returns a dict describing the new mesh layout:
      new_blocks: list of dicts {id, name, elem_type, conn (n,npe)}
      old_to_new: dict old_global_elem -> new_global_elem            (untouched)
                  dict old_global_elem -> (new_ge_A, new_ge_B)        (converted)
      new_conn_by_ge: full (num_elem_new, ) lookup -> (nodes tuple, elem_type)
    """
    new_blocks = []
    old_to_new = {}
    new_conn_lookup = {}   # new_ge -> (np.array of node ids, "HEX8"/"WEDGE6")

    next_ge = 0

    # 1) untouched (and non-hex) elements, block by block, preserving block id
    for bi in range(mesh.num_el_blk):
        conn = mesh.block_conn[bi]
        off = mesh.block_offset[bi]
        etype = mesh.block_elem_type[bi]
        is_hex = etype.upper().startswith("HEX")
        keep_rows = []
        for li in range(conn.shape[0]):
            ge = off + li
            if is_hex and ge in converted:
                continue
            keep_rows.append(li)
        if not keep_rows:
            continue
        new_conn = conn[keep_rows]
        block_start = next_ge
        for row_i, li in enumerate(keep_rows):
            ge_old = off + li
            ge_new = block_start + row_i
            old_to_new[ge_old] = ge_new
            new_conn_lookup[ge_new] = (new_conn[row_i], etype)
        next_ge += len(keep_rows)
        new_blocks.append({
            "id": mesh.eb_ids[bi],
            "name": (mesh.eb_names[bi] if mesh.eb_names else f"block_{mesh.eb_ids[bi]}"),
            "elem_type": etype,
            "conn": new_conn,
        })

    # 2) new wedge blocks, one per source hex block that had conversions
    by_block = defaultdict(list)   # source block index -> list of (old_ge, prismA, prismB)
    for ge, (pa, pb) in converted.items():
        bi = mesh.elem_block[ge]
        by_block[bi].append((ge, pa, pb))

    for bi, items in by_block.items():
        items.sort(key=lambda t: t[0])
        rows = []
        block_start = next_ge
        row_i = 0
        for ge_old, pa, pb in items:
            ge_A = block_start + row_i
            ge_B = block_start + row_i + 1
            old_to_new[ge_old] = (ge_A, ge_B)
            new_conn_lookup[ge_A] = (pa, "WEDGE6")
            new_conn_lookup[ge_B] = (pb, "WEDGE6")
            rows.append(pa)
            rows.append(pb)
            row_i += 2
        next_ge += row_i
        new_id = mesh.eb_ids[bi] + wedge_id_offset
        src_name = mesh.eb_names[bi] if mesh.eb_names else f"block_{mesh.eb_ids[bi]}"
        new_blocks.append({
            "id": new_id,
            "name": f"{src_name}_prism",
            "elem_type": "WEDGE6",
            "conn": np.array(rows, dtype=np.int64),
        })

    num_elem_new = next_ge
    return new_blocks, old_to_new, new_conn_lookup, num_elem_new


def remap_side_sets(mesh, old_to_new, new_conn_lookup):
    """
    Returns list of dicts: {id, name, elem (1-based new), side (1-based new)}
    """
    new_ss = []
    for i in range(mesh.num_side_sets):
        old_elems = mesh.ss_elem[i]
        old_sides = mesh.ss_side[i]
        new_e_list = []
        new_s_list = []
        for e0, f0 in zip(old_elems, old_sides):
            e0 = int(e0)
            f0 = int(f0)
            old_conn = mesh.global_conn(e0)
            old_face_nodes = frozenset(old_conn[HEX_FACES[f0]].tolist())

            mapped = old_to_new[e0]
            if isinstance(mapped, tuple):
                candidates = mapped  # (ge_A, ge_B), both WEDGE6
            else:
                candidates = (mapped,)

            found_any = False
            for new_ge in candidates:
                nodes, etype = new_conn_lookup[new_ge]
                faces = HEX_FACES if etype.upper().startswith("HEX") else WEDGE_FACES
                for lf, idxs in enumerate(faces):
                    face_nodes = frozenset(nodes[idxs].tolist())
                    if face_nodes == old_face_nodes or (
                        len(face_nodes) == 3 and face_nodes.issubset(old_face_nodes)
                    ):
                        new_e_list.append(new_ge + 1)
                        new_s_list.append(lf + 1)
                        found_any = True
            if not found_any:
                print(f"  [warn] side set '{mesh.ss_ids[i]}' entry (elem {e0+1}, "
                      f"side {f0+1}) could not be remapped -- dropped.")
        new_ss.append({
            "id": mesh.ss_ids[i],
            "name": mesh.ss_names[i] if mesh.ss_names else f"surface_{mesh.ss_ids[i]}",
            "elem": np.array(new_e_list, dtype=np.int64),
            "side": np.array(new_s_list, dtype=np.int64),
        })
    return new_ss


# ==========================================================================
# Writing
# ==========================================================================

def write_exodus(out_path, mesh, new_blocks, new_ss):
    npe_by_type = {"HEX8": 8, "WEDGE6": 6}
    num_elem_new = sum(b["conn"].shape[0] for b in new_blocks)

    with nc.Dataset(out_path, "w", format="NETCDF3_64BIT_OFFSET") as o:
        # global attrs
        o.setncattr("api_version", np.float32(8.11))
        o.setncattr("version", np.float32(8.11))
        o.setncattr("floating_point_word_size", np.int32(8))
        o.setncattr("file_size", np.int32(1))
        o.setncattr("title", "hex_to_prism_boundary_layer.py output")

        len_string = 33
        len_line = 81
        o.createDimension("len_string", len_string)
        o.createDimension("len_line", len_line)
        o.createDimension("four", 4)
        o.createDimension("time_step", None)
        o.createDimension("num_dim", mesh.num_dim)
        o.createDimension("num_nodes", mesh.num_nodes)
        o.createDimension("num_elem", num_elem_new)
        o.createDimension("num_el_blk", len(new_blocks))
        o.createDimension("num_qa_rec", 1)

        # coordinates
        # Written as the combined 2D "coord" array (num_dim, num_nodes) AND
        # as the legacy separate coordx/coordy/coordz 1D arrays, because
        # different ExodusII/IOSS reader builds look for one or the other
        # (some readers -- as seen with vtkexodusII's ex_get_coord --
        # specifically request "x nodal coordinates" i.e. coordx and do not
        # fall back to "coord"). Writing both is harmless and guarantees
        # compatibility either way.
        # mesh.coord is (num_nodes, num_dim) in C order; np.ascontiguousarray
        # forces a real contiguous copy for the transposed write instead of
        # handing netCDF4 a non-contiguous Fortran-order view of .T (which
        # triggers a deprecated in-place reshape path in recent NumPy/
        # netCDF4 combinations and is worth avoiding rather than trusting).
        coordv = o.createVariable("coord", "f8", ("num_dim", "num_nodes"))
        coordv[:, :] = np.ascontiguousarray(mesh.coord.T)

        names = ["x", "y", "z"][: mesh.num_dim]
        for i, nm in enumerate(names):
            v = o.createVariable(f"coord{nm}", "f8", ("num_nodes",))
            v[:] = np.ascontiguousarray(mesh.coord[:, i])
        coor_names = o.createVariable("coor_names", "S1", ("num_dim", "len_string"))
        for i, nm in enumerate(names):
            arr = np.zeros((len_string,), dtype="S1")
            for j, ch in enumerate(nm):
                arr[j] = ch.encode("ascii")
            coor_names[i, :] = arr

        # element blocks
        eb_prop1 = o.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        eb_status = o.createVariable("eb_status", "i4", ("num_el_blk",))
        eb_names = o.createVariable("eb_names", "S1", ("num_el_blk", "len_string"))

        for bi, blk in enumerate(new_blocks):
            n_el = blk["conn"].shape[0]
            npe = npe_by_type[blk["elem_type"].upper()]
            o.createDimension(f"num_el_in_blk{bi+1}", n_el)
            o.createDimension(f"num_nod_per_el{bi+1}", npe)
            conn_v = o.createVariable(
                f"connect{bi+1}", "i4", (f"num_el_in_blk{bi+1}", f"num_nod_per_el{bi+1}")
            )
            conn_v.setncattr("elem_type", blk["elem_type"])
            conn_v[:, :] = blk["conn"] + 1  # back to 1-based

            eb_prop1[bi] = blk["id"]
            eb_status[bi] = 1
            arr = np.zeros((len_string,), dtype="S1")
            for j, ch in enumerate(blk["name"][: len_string - 1]):
                arr[j] = ch.encode("ascii", errors="ignore")
            eb_names[bi, :] = arr

        # side sets
        n_ss = len(new_ss)
        if n_ss:
            o.createDimension("num_side_sets", n_ss)
            ss_prop1 = o.createVariable("ss_prop1", "i4", ("num_side_sets",))
            ss_prop1.setncattr("name", "ID")
            ss_status = o.createVariable("ss_status", "i4", ("num_side_sets",))
            ss_names = o.createVariable("ss_names", "S1", ("num_side_sets", "len_string"))
            for i, ss in enumerate(new_ss):
                n = len(ss["elem"])
                o.createDimension(f"num_side_ss{i+1}", max(n, 1))
                # NOTE: no num_df_ss{i} dimension/variable is created here.
                # We don't (re)compute per-node distribution factors for the
                # split faces, and in netCDF classic format a dimension
                # created with length 0 is silently treated as an *unlimited*
                # dimension -- since "time_step" already occupies the single
                # unlimited-dimension slot classic format allows, creating a
                # second one (num_df_ss{i}=0) raises
                # "NC_UNLIMITED size already in use". Simply omitting the
                # dist-factor dimension/variable is valid Exodus (num_df=0
                # for that side set) and most readers recompute df's from
                # geometry anyway -- see the module docstring caveat.
                elem_v = o.createVariable(f"elem_ss{i+1}", "i4", (f"num_side_ss{i+1}",))
                side_v = o.createVariable(f"side_ss{i+1}", "i4", (f"num_side_ss{i+1}",))
                if n:
                    elem_v[:] = ss["elem"]
                    side_v[:] = ss["side"]
                ss_prop1[i] = ss["id"]
                ss_status[i] = 1 if n else 0
                arr = np.zeros((len_string,), dtype="S1")
                for j, ch in enumerate(ss["name"][: len_string - 1]):
                    arr[j] = ch.encode("ascii", errors="ignore")
                ss_names[i, :] = arr

        # node sets: copied through unchanged (node numbering is untouched)
        if mesh.num_node_sets:
            src = mesh.ds
            n_ns = mesh.num_node_sets
            o.createDimension("num_node_sets", n_ns)
            ns_prop1 = o.createVariable("ns_prop1", "i4", ("num_node_sets",))
            ns_status = o.createVariable("ns_status", "i4", ("num_node_sets",))
            ns_names = o.createVariable("ns_names", "S1", ("num_node_sets", "len_string"))
            ns_prop1[:] = src.variables["ns_prop1"][:]
            ns_status[:] = (src.variables["ns_status"][:] if "ns_status" in src.variables
                             else np.ones(n_ns, dtype="i4"))
            if "ns_names" in src.variables:
                ns_names[:, :] = src.variables["ns_names"][:]
            for i in range(1, n_ns + 1):
                n = src.dimensions[f"num_nod_ns{i}"].size
                o.createDimension(f"num_nod_ns{i}", n)
                v = o.createVariable(f"node_ns{i}", "i4", (f"num_nod_ns{i}",))
                v[:] = src.variables[f"node_ns{i}"][:]

        # minimal QA record
        qa = o.createVariable("qa_records", "S1", ("num_qa_rec", "four", "len_string"))
        qa_fields = ["hex_to_prism_boundary_layer.py", "1.0", "2026/09/18", "00:00:00"]
        for k, txt in enumerate(qa_fields):
            arr = np.zeros((len_string,), dtype="S1")
            for j, ch in enumerate(txt[: len_string - 1]):
                arr[j] = ch.encode("ascii", errors="ignore")
            qa[0, k, :] = arr


# ==========================================================================
# CLI
# ==========================================================================

def list_mesh(mesh):
    print(f"num_nodes = {mesh.num_nodes}   num_elem = {mesh.num_elem}")
    print("Element blocks:")
    for bi in range(mesh.num_el_blk):
        name = mesh.eb_names[bi] if mesh.eb_names else ""
        print(f"  id={mesh.eb_ids[bi]:<6} name={name:<20} type={mesh.block_elem_type[bi]:<8} "
              f"n_elem={mesh.block_conn[bi].shape[0]}")
    print("Side sets:")
    for i in range(mesh.num_side_sets):
        name = mesh.ss_names[i] if mesh.ss_names else ""
        print(f"  id={mesh.ss_ids[i]:<6} name={name:<20} n_faces={len(mesh.ss_elem[i])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input Exodus II mesh (all-HEX8 blocks)")
    ap.add_argument("output", nargs="?", help="output Exodus II mesh")
    ap.add_argument("--list", action="store_true",
                     help="print block/side-set ids and names, then exit")
    ap.add_argument("--wall-sideset-ids", nargs="+", type=int, default=[],
                     help="side set IDs to treat as walls (start of each column)")
    ap.add_argument("--wall-sideset-names", nargs="+", default=[],
                     help="side set names to treat as walls (start of each column)")
    ap.add_argument("--wedge-id-offset", type=int, default=1000,
                     help="new wedge block id = source hex block id + this offset "
                          "(default: 1000)")
    args = ap.parse_args()

    mesh = ExodusMesh(args.input)

    if args.list:
        list_mesh(mesh)
        return

    if not args.output:
        sys.exit("output path required unless --list is given")

    wall_ss_indices = []
    for i, sid in enumerate(mesh.ss_ids):
        if sid in args.wall_sideset_ids:
            wall_ss_indices.append(i)
    if args.wall_sideset_names and mesh.ss_names:
        for i, nm in enumerate(mesh.ss_names):
            if nm in args.wall_sideset_names:
                wall_ss_indices.append(i)
    wall_ss_indices = sorted(set(wall_ss_indices))
    if not wall_ss_indices:
        sys.exit("No matching wall side sets found. Run with --list to see available ids/names.")

    print("Wall side sets used as column starting points:")
    for i in wall_ss_indices:
        nm = mesh.ss_names[i] if mesh.ss_names else ""
        print(f"  id={mesh.ss_ids[i]} name={nm} n_faces={len(mesh.ss_elem[i])}")

    wall_faces = []
    for i in wall_ss_indices:
        for e, f in zip(mesh.ss_elem[i], mesh.ss_side[i]):
            wall_faces.append((int(e), int(f)))

    hex_block_indices = [bi for bi, t in enumerate(mesh.block_elem_type)
                          if t.upper().startswith("HEX")]
    hex_block_set = set(hex_block_indices)
    if not hex_block_indices:
        sys.exit("No HEX8 element blocks found in this file.")

    print("Building hex-to-hex face adjacency ...")
    adjacency = build_hex_adjacency(mesh, hex_block_indices)

    print("Walking wall-normal columns and splitting hexes into prisms ...")
    converted = walk_and_split(mesh, adjacency, wall_faces, hex_block_set, mesh.coord)

    print("Rebuilding element blocks ...")
    new_blocks, old_to_new, new_conn_lookup, num_elem_new = rebuild(
        mesh, converted, args.wedge_id_offset
    )
    print(f"  new total element count: {num_elem_new} (was {mesh.num_elem})")

    print("Remapping side sets ...")
    new_ss = remap_side_sets(mesh, old_to_new, new_conn_lookup)

    print(f"Writing {args.output} ...")
    write_exodus(args.output, mesh, new_blocks, new_ss)
    mesh.close()
    print("Done.")


if __name__ == "__main__":
    main()
