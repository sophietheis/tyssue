import itertools
import logging
import warnings
from functools import wraps


import numpy as np
import pandas as pd

from ..core.monolayer import Monolayer
from ..core.objects import _is_closed_cell, euler_characteristic
from ..core.sheet import get_opposite
from ..geometry.utils import rotation_matrix
from .base_topology import add_vert, close_face, collapse_edge, remove_face, merge_vertices, condition_4i, condition_4ii
from .base_topology import split_vert as base_split_vert
from .sheet_topology import face_division

logger = logging.getLogger(name=__name__)
MAX_ITER = 10


def remove_cell(eptm, cell):
    """Removes a tetrahedral cell from the epithelium."""
    eptm.get_opposite_faces()
    edges = eptm.edge_df.query(f"cell == {cell}")
    if not edges.shape[0] == 12:
        warnings.warn(f"{cell} is not a tetrahedral cell, aborting.")
        return -1
    faces = eptm.face_df.loc[edges["face"].unique()]
    oppo = faces["opposite"][faces["opposite"] != -1]
    verts = eptm.vert_df.loc[edges["srce"].unique()].copy()
    eptm.vert_df = pd.concat(
        [eptm.vert_df, pd.DataFrame(verts.mean(numeric_only=True))], ignore_index=True
    )
    new_vert = eptm.vert_df.index[-1]

    eptm.vert_df.loc[new_vert, "segment"] = "basal"
    eptm.edge_df.replace(
        {"srce": verts.index, "trgt": verts.index}, new_vert, inplace=True
    )

    collapsed = eptm.edge_df.query("srce == trgt")

    eptm.face_df.drop(faces.index, axis=0, inplace=True)
    eptm.face_df.drop(oppo, axis=0, inplace=True)

    eptm.edge_df.drop(collapsed.index, axis=0, inplace=True)

    eptm.cell_df.drop(cell, axis=0, inplace=True)
    eptm.vert_df.drop(verts.index, axis=0, inplace=True)
    eptm.reset_index()
    eptm.reset_topo()
    return 0


def close_cell(eptm, cell):
    """Closes the cell by adding a face. Assumes a single face is missing"""
    face_edges = eptm.edge_df[eptm.edge_df["cell"] == cell]
    euler_c = euler_characteristic(face_edges)

    if euler_c == 2:
        logger.warning("cell %s is already closed", cell)
        return 0

    if euler_c != 1:
        raise ValueError("Cell has more than one hole")

    eptm.face_df = pd.concat([eptm.face_df, eptm.face_df.loc[0:0]], ignore_index=True)
    new_face = eptm.face_df.index[-1]

    oppo = get_opposite(face_edges, raise_if_invalid=True)
    new_edges = face_edges[oppo == -1].copy()
    logger.info("closing cell %d", cell)
    new_edges[["srce", "trgt"]] = new_edges[["trgt", "srce"]]
    new_edges["face"] = new_face
    new_edges.index = new_edges.index + eptm.edge_df.index.max()

    eptm.edge_df = pd.concat([eptm.edge_df, new_edges], ignore_index=False)

    eptm.reset_index()
    eptm.reset_topo()
    return 0

def find_rearangements(eptm):
    """Finds the candidates for IH and HI transitions
    Returns
    -------
    edges_HI: set of indexes of short edges
    faces_IH: set of indexes of small triangular faces
    """
    l_th = eptm.settings.get("threshold_length", 1e-3)
    shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return np.array([]), np.array([])
    edges_IH = find_IHs(eptm, shorts)
    faces_HI = find_HIs(eptm, shorts)
    return edges_IH, faces_HI


def find_IHs(eptm, shorts=None):

    l_th = eptm.settings.get("threshold_length", 1e-3)
    if shorts is None:
        shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return []

    edges_IH = shorts.groupby("srce").apply(
        lambda df: pd.Series(
            {
                "edge": df.index[0],
                "length": df["length"].iloc[0],
                "num_sides": min(eptm.face_df.loc[df["face"], "num_sides"]),
                "pair": frozenset(df.iloc[0][["srce", "trgt"]]),
            }
        )
    )
    # keep only one of the edges per vertex pair and sort by length
    edges_IH = (
        edges_IH[edges_IH["num_sides"] > 3]
        .drop_duplicates("pair")
        .sort_values("length")
    )
    return edges_IH["edge"].values


def find_HIs(eptm, shorts=None):
    l_th = eptm.settings.get("threshold_length", 1e-3)
    if shorts is None:
        shorts = eptm.edge_df[(eptm.edge_df["length"] < l_th)]
    if not shorts.shape[0]:
        return []

    max_f_length = shorts.groupby("face")["length"].apply(max)
    short_faces = eptm.face_df.loc[max_f_length[max_f_length < l_th].index]
    faces_HI = short_faces[short_faces["num_sides"] == 3].sort_values("area").index
    return faces_HI
def check_condition4(func):
    @wraps(func)
    def decorated(eptm, *args, **kwargs):
        eptm.backup()
        res = func(eptm, *args, **kwargs)
        if len(condition_4i(eptm)) or len(condition_4ii(eptm)):
            print("Invalid epithelium produced, restoring")
            # print("4i on", condition_4i(eptm))
            # print("4ii on", condition_4ii(eptm))
            eptm.restore()
            eptm.topo_changed = True
        return res

    return decorated

def _set_new_pos_IH(eptm, e_1011, vertices, cells):
    """Okuda 2013 equations 46 to 56
    """
    Dl_th = eptm.settings["threshold_length"]
    (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) = vertices

    # eq. 49
    r_1011 = -eptm.edge_df.loc[e_1011, eptm.dcoords].values
    u_T = r_1011 / np.linalg.norm(r_1011)
    # eq. 50
    r0 = eptm.vert_df.loc[[v10, v11], eptm.coords].mean(axis=0).values

    v_0ns = []
    for vi, vj, vk in zip((v1, v2, v3), (v4, v5, v6), (v7, v8, v9)):
        # eq. 54 - 56
        r0i, r0j = eptm.vert_df.loc[[vi, vj], eptm.coords].values - r0[np.newaxis, :]
        w_0k = (r0i / np.linalg.norm(r0i) + r0j / np.linalg.norm(r0j)) / 2
        # eq. 51 - 53
        v_0k = w_0k - (np.dot(w_0k, u_T)) * u_T
        v_0ns.append(v_0k)

    # see definition of l_max bellow eq. 56
    l_max = np.max(
        [np.linalg.norm(v_n - v_m) for (v_n, v_m) in itertools.combinations(v_0ns, 2)]
    )
    # eq. 46 - 49
    for i, vk, v_0k in zip(range(3), (v7, v8, v9), v_0ns):
        if i == 2:
            # eptm.vert_df.loc[vk, eptm.coords] = r0 + (Dl_th / l_max) * v_0k
            # eptm.vert_df.loc[vk, eptm.coords] = r0
            eptm.vert_df.loc[vk, eptm.coords] = eptm.vert_df.loc[[v3, v6, v10, v11], eptm.coords].mean(axis=0).values
            eptm.vert_df.loc[vk, 'z'] = 0
        else:
            # eptm.vert_df.loc[vk, eptm.coords] = r0 + ((Dl_th) / l_max) * v_0k
            eptm.vert_df.loc[vk, eptm.coords] = r0 + v_0k

    # # ecarter v3, v6
    # (cA, cB, cC, cD, cE) = cells
    # cA_center = eptm.cell_df.loc[cA, eptm.coords].values
    # v3_z_save = eptm.vert_df.loc[v3, 'z']
    # eptm.vert_df.loc[v3, eptm.coords] = (eptm.vert_df.loc[v3, eptm.coords].values + (eptm.vert_df.loc[[v3], eptm.coords].values - cA_center[np.newaxis, :]))[0]
    # eptm.vert_df.loc[v3, 'z'] = v3_z_save
    #
    # cB_center = eptm.cell_df.loc[cB, eptm.coords].values
    # v6_z_save = eptm.vert_df.loc[v6, 'z']
    # eptm.vert_df.loc[v6, eptm.coords] = (eptm.vert_df.loc[v6, eptm.coords].values + (eptm.vert_df.loc[[v6], eptm.coords].values - cB_center[np.newaxis, :]))[0]
    # eptm.vert_df.loc[v6, 'z'] = v6_z_save

def _get_vertex_pairs_IH(eptm, e_1011):

    srce_face_orbits = eptm.get_orbits("srce", "face")
    v10, v11 = eptm.edge_df.loc[e_1011, ["srce", "trgt"]]
    common_faces = set(srce_face_orbits.loc[v10]).intersection(
        srce_face_orbits.loc[v11]
    )
    if eptm.face_df.loc[common_faces, "num_sides"].min() < 4:
        logger.warning(
            "Edge %i has adjacent triangular faces"
            " can't perform IH transition, aborting",
            e_1011,
        )
        return None

    v10_out = set(eptm.edge_df[eptm.edge_df["srce"] == v10]["trgt"]) - {v11}
    faces_123 = {
        v: set(srce_face_orbits.loc[v])  # .intersection(srce_face_orbits.loc[v10])
        for v in v10_out
    }

    v11_out = set(eptm.edge_df[eptm.edge_df["srce"] == v11]["trgt"]) - {v10}
    faces_456 = {
        v: set(srce_face_orbits.loc[v])  # .intersection(srce_face_orbits.loc[v11])
        for v in v11_out
    }
    v_pairs = []
    for vi in v10_out:
        for vj in v11_out:
            common_face = faces_123[vi].intersection(faces_456[vj])
            if common_face:
                v_pairs.append((vi, vj))
                break
        else:
            return None
    return v_pairs
def _set_new_pos_HI(eptm, fa, v10, v11):

    r0 = eptm.face_df.loc[fa, eptm.coords].values

    norm_a = eptm.edge_df[eptm.edge_df["face"] == fa][eptm.ncoords].mean(axis=0).values
    norm_a = norm_a / np.linalg.norm(norm_a)
    norm_b = -norm_a
    Dl_th = eptm.settings["threshold_length"] * 1.01
    eptm.vert_df.loc[v10, eptm.coords] = r0 + Dl_th / 2 * norm_b
    eptm.vert_df.loc[v11, eptm.coords] = r0 + Dl_th / 2 * norm_a
@check_condition4
def HI_transition(eptm, face):
    """
    H → I transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the definition of the
    edges, which follow the one in the above article
    """
    if eptm.face_df.loc[face, "num_sides"] != 3:
        raise ValueError("Only three sided faces can undergo a H-I transition")

    fa = face
    f_edges = eptm.edge_df[eptm.edge_df["face"] == face]
    v7 = f_edges.iloc[0]["srce"]
    v8 = f_edges.iloc[0]["trgt"]
    v9 = f_edges[f_edges["srce"] == v8]["trgt"].iloc[0]

    cA = f_edges["cell"].iloc[0]

    eptm.get_opposite_faces()
    fb = eptm.face_df["opposite"].loc[face]
    if fb > 0:
        cB = eptm.edge_df[eptm.edge_df["face"] == fb]["cell"].iloc[0]
    else:
        cB = None

    cA_edges = eptm.edge_df[eptm.edge_df["cell"] == cA]

    v_pairs = []
    for vk in (v7, v8, v9):
        vis = set(cA_edges[cA_edges["srce"] == vk]["trgt"])
        try:
            vi, = vis.difference({v7, v8, v9})
        except ValueError:
            warnings.warn("Invalid topology for a HI transition, aborting")
            return -1
        vjs = set(eptm.edge_df[eptm.edge_df["srce"] == vk]["trgt"])
        try:
            vj, = vjs.difference({v7, v8, v9, vi})
        except ValueError:
            warnings.warn("Invalid topology for a HI transition, aborting")
            return -1
        v_pairs.append((vi, vj))

    (v1, v4), (v2, v5), (v3, v6) = v_pairs

    srce_cell_orbit = eptm.get_orbits("srce", "cell")
    cells = [cA, cB]
    for (vi, vj, vk) in [(v1, v2, v4), (v2, v3, v5), (v1, v3, v4)]:
        cell = list(
            set(srce_cell_orbit.loc[vi])
            .intersection(srce_cell_orbit.loc[vj])
            .intersection(srce_cell_orbit.loc[vk])
        )

        cells.append(cell[0] if cell else None)

    cA, cB, cC, cD, cE = cells

    eptm.vert_df = eptm.vert_df.append(eptm.vert_df.loc[[v8, v9]], ignore_index=True)
    eptm.vert_df.index.name = "vert"
    v10, v11 = eptm.vert_df.index[-2:]
    _set_new_pos_HI(eptm, fa, v10, v11)

    for vi, vj, vk in zip((v1, v2, v3), (v4, v5, v6), (v7, v8, v9)):
        e_iks = eptm.edge_df[
            (eptm.edge_df["srce"] == vi) & (eptm.edge_df["trgt"] == vk)
        ].index
        eptm.edge_df.loc[e_iks, "trgt"] = v10

        e_kis = eptm.edge_df[
            (eptm.edge_df["srce"] == vk) & (eptm.edge_df["trgt"] == vi)
        ].index
        eptm.edge_df.loc[e_kis, "srce"] = v10

        e_jks = eptm.edge_df[
            (eptm.edge_df["srce"] == vj) & (eptm.edge_df["trgt"] == vk)
        ].index
        eptm.edge_df.loc[e_jks, "trgt"] = v11

        e_kjs = eptm.edge_df[
            (eptm.edge_df["srce"] == vk) & (eptm.edge_df["trgt"] == vj)
        ].index
        eptm.edge_df.loc[e_kjs, "srce"] = v11

    # Closing the faces with v10 → v11 edges
    for cell in cells:
        for face in eptm.edge_df[eptm.edge_df["cell"] == cell]["face"]:
            close_face(eptm, face)

    # Removing the remaining edges and vertices
    todel_edges = eptm.edge_df[
        (eptm.edge_df["srce"] == v7)
        | (eptm.edge_df["trgt"] == v7)
        | (eptm.edge_df["srce"] == v8)
        | (eptm.edge_df["trgt"] == v8)
        | (eptm.edge_df["srce"] == v9)
        | (eptm.edge_df["trgt"] == v9)
    ].index

    eptm.edge_df = eptm.edge_df.loc[eptm.edge_df.index.delete(todel_edges)]
    eptm.vert_df = eptm.vert_df.loc[eptm.vert_df.index.delete([v7, v8, v9])]
    orphan_faces = set(eptm.face_df.index).difference(eptm.edge_df.face)
    eptm.face_df = eptm.face_df.loc[
        eptm.face_df.index.delete(list(orphan_faces))
    ].copy()
    eptm.edge_df.index.name = "edge"
    eptm.reset_index()
    eptm.reset_topo()
    logger.info(f"HI transition on edge {face}")
    return 0
@check_condition4
def IH_transition(eptm, e_1011):
    """
    I → H transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the definition of the
    edges, which follow the one in the above article
    """

    v10, v11 = eptm.edge_df.loc[e_1011, ["srce", "trgt"]]
    v_pairs = _get_vertex_pairs_IH(eptm, e_1011)
    if v_pairs is None:
        logger.warning(
            "Edge %i is not a valid junction to perform IH transition, aborting", e_1011
        )
        return -1

    try:
        (v1, v4), (v2, v5), (v3, v6) = v_pairs
    except ValueError:
        logger.warning(
            "Edge %i is not a valid junction to perform IH transition, aborting", e_1011
        )
        return -1

    if len({v1, v4, v2, v5, v3, v6}) != 6:
        # raise ValueError(
        #     """
        # Topology cannot be correctly determined around edge %i
        # """,
        #     e_1011,
        # )
        logger.warning(
                """
            Topology cannot be correctly determined around edge %i
            """,
                e_1011,
            )
        return -1

    new_vs = eptm.vert_df.loc[[v1, v2, v3]].copy()
    eptm.vert_df = eptm.vert_df.append(new_vs, ignore_index=True)
    v7, v8, v9 = eptm.vert_df.index[-3:]
    eptm.vert_df.loc[v9, 'segment'] = 'lateral'

    cells = []
    srce_cell_orbits = eptm.get_orbits("srce", "cell")
    for vi, vj, vk in [
        (v1, v2, v3),
        (v4, v5, v6),
        (v1, v2, v11),
        (v2, v3, v11),
        (v3, v1, v11),
    ]:
        cell = list(
            set(srce_cell_orbits.loc[vi])
            .intersection(srce_cell_orbits.loc[vj])
            .intersection(srce_cell_orbits.loc[vk])
        )
        cells.append(cell[0] if cell else None)

    cA, cB, cC, cD, cE = cells
    if cA is not None:
        # orient vertices 1,2,3 positively
        r_12 = (
            eptm.vert_df.loc[v2, eptm.coords].values
            - eptm.vert_df.loc[v1, eptm.coords].values
        ).astype(np.float)
        r_23 = (
            eptm.vert_df.loc[v3, eptm.coords].values
            - eptm.vert_df.loc[v2, eptm.coords].values
        ).astype(np.float)
        r_123 = eptm.vert_df.loc[[v1, v2, v3], eptm.coords].mean(axis=0).values
        r_A = eptm.cell_df.loc[cA, eptm.coords].values
        orient = np.dot(np.cross(r_12, r_23), (r_123 - r_A))
    elif cB is not None:
        # orient vertices 4,5,6 negatively
        r_45 = (
            eptm.vert_df.loc[v5, eptm.coords].values
            - eptm.vert_df.loc[v4, eptm.coords].values
        ).astype(np.float)
        r_56 = (
            eptm.vert_df.loc[v6, eptm.coords].values
            - eptm.vert_df.loc[v5, eptm.coords].values
        ).astype(np.float)
        r_456 = eptm.vert_df.loc[[v4, v5, v6], eptm.coords].mean(axis=0).values
        r_B = eptm.cell_df.loc[cB, eptm.coords].values
        orient = -np.dot(np.cross(r_45, r_56), (r_456 - r_B))
    else:
        logger.warning(
            "I - H transition is not possible without cells on either ends"
            " of the edge - would result in a hole"
        )
        return -1

    if orient < 0:
        v1, v2, v3 = v1, v3, v2
        v4, v5, v6 = v4, v6, v5
        cC, cE = cE, cC
    vertices = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]

    for i, va, vb, new in zip(range(3), (v1, v2, v3), (v4, v5, v6), (v7, v8, v9)):
        # assign v1 -> v10 edges to  v1 -> v7
        e_a10s = eptm.edge_df[
            (eptm.edge_df["srce"] == va) & (eptm.edge_df["trgt"] == v10)
        ].index
        eptm.edge_df.loc[e_a10s, "trgt"] = new
        # assign v10 -> v1 edges to  v7 -> v1
        e_10as = eptm.edge_df[
            (eptm.edge_df["srce"] == v10) & (eptm.edge_df["trgt"] == va)
        ].index
        eptm.edge_df.loc[e_10as, "srce"] = new
        # assign v4 -> v11 edges to  v4 -> v7
        e_b11s = eptm.edge_df[
            (eptm.edge_df["srce"] == vb) & (eptm.edge_df["trgt"] == v11)
        ].index
        eptm.edge_df.loc[e_b11s, "trgt"] = new
        # assign v11 -> v4 edges to  v7 -> v4
        e_11bs = eptm.edge_df[
            (eptm.edge_df["srce"] == v11) & (eptm.edge_df["trgt"] == vb)
        ].index
        eptm.edge_df.loc[e_11bs, "srce"] = new
        # if i == 2:
        #     eptm.edge_df.loc[e_a10s, "segment"] = "lateral"
        #     eptm.edge_df.loc[e_10as, "segment"] = "lateral"
        #     eptm.edge_df.loc[e_b11s, "segment"] = "lateral"
        #     eptm.edge_df.loc[e_11bs, "segment"] = "lateral"

    cells = (cA, cB, cC, cD, cE)
    _set_new_pos_IH(eptm, e_1011, vertices, cells)

    face = eptm.edge_df.loc[e_1011, "face"]
    new_fs = eptm.face_df.loc[[face, face]].copy()
    eptm.face_df = eptm.face_df.append(new_fs, ignore_index=True)
    fa, fb = eptm.face_df.index[-2:]
    eptm.face_df.loc[fa, 'segment'] = "lateral"
    eptm.face_df.loc[fb, 'segment'] = "lateral"
    eptm.face_df.loc[fa, 'prefered_perimeter'] = 2.8
    eptm.face_df.loc[fb, 'prefered_perimeter'] = 2.8
    eptm.face_df.loc[fa, 'prefered_area'] = 2
    eptm.face_df.loc[fb, 'prefered_area'] = 2
    edges_fa_fb = eptm.edge_df.loc[[e_1011] * 6].copy()
    eptm.edge_df = eptm.edge_df.append(edges_fa_fb, ignore_index=True)
    new_es = eptm.edge_df.index[-6:]
    for i, eA, eB, (vi, vj) in zip(range(3),
        new_es[::2], new_es[1::2], [(v7, v8), (v8, v9), (v9, v7)]
    ):
        # if i == 0:
        eptm.edge_df.loc[eA, ["srce", "trgt", "face", "cell"]] = vi, vj, fa, cA
        eptm.edge_df.loc[eB, ["srce", "trgt", "face", "cell"]] = vj, vi, fb, cB
        # else:
        #     eptm.edge_df.loc[eA, ["srce", "trgt", "face", "cell", "segment"]] = vi, vj, fa, cA, "lateral"
        #     eptm.edge_df.loc[eB, ["srce", "trgt", "face", "cell", "segment"]] = vj, vi, fb, cB, "lateral"

    for cell in cells:
        for face in eptm.edge_df[eptm.edge_df["cell"] == cell]["face"]:
            close_face(eptm, face)

    # Removing the remaining edges and vertices
    todel_edges = eptm.edge_df[
        (eptm.edge_df["srce"] == v10)
        | (eptm.edge_df["trgt"] == v10)
        | (eptm.edge_df["srce"] == v11)
        | (eptm.edge_df["trgt"] == v11)
        | pd.isna(eptm.edge_df["cell"])
    ].index

    eptm.edge_df = eptm.edge_df.loc[eptm.edge_df.index.delete(todel_edges)]
    eptm.vert_df = eptm.vert_df.loc[set(eptm.edge_df.sort_values("srce")["srce"])]
    eptm.face_df = eptm.face_df.loc[set(eptm.edge_df.sort_values("face")["face"])]
    eptm.cell_df = eptm.cell_df.loc[set(eptm.edge_df.sort_values("cell")["cell"])]

    eptm.edge_df.index.name = "edge"
    if isinstance(eptm, Monolayer):
        for vert in (v7, v8, v9):
            eptm.guess_vert_segment(vert)
        for face in fa, fb:
            eptm.guess_face_segment(face)

    eptm.reset_index()
    eptm.reset_topo()
    logger.info(f"IH transition on edge {e_1011}")
    return 0


def split_vert(eptm, vert, face=None, multiplier=1.5, recenter=False):
    """Splits a vertex towards a face.

    Parameters
    ----------
    eptm : a :class:`tyssue.Epithelium` instance
    vert : int the vertex to split
    face : int, optional, the face to split
        if face is None, one face will be chosen at random
    multiplier: float, default 1.5
        length of the new edge(s) in units of eptm.settings["threshold_length"]

    Note on the algorithm
    ---------------------

    For a given face, we look for the adjacent cell with the lowest number
    of faces converging on the vertex. If this number is higher than 4
    we raise a ValueError

    If it's 3, we do a OI transition, resulting in a new edge but no new faces
    If it's 4, we do a IH transition, resulting in a new face and 2 ne edges.

    see ../doc/illus/IH_transition.png
    """
    print("split vert bulk topology")
    all_edges = eptm.edge_df[
        (eptm.edge_df["trgt"] == vert) | (eptm.edge_df["srce"] == vert)
    ]
    print("all edges:")
    print(all_edges[['srce', 'trgt', 'face', 'segment', 'cell']])
    faces = all_edges.groupby("face").apply(
        lambda df: pd.Series(
            {
                "verts": frozenset(df[["srce", "trgt"]].values.ravel()),
                "cell": df["cell"].iloc[0],
                "segment": df["segment"].iloc[0],
            }
        )
    )
    print(faces)
    cells = all_edges.groupby("cell").apply(
        lambda df: pd.Series(
            {
                "verts": frozenset(df[["srce", "trgt"]].values.ravel()),
                "faces": frozenset(df["face"]),
                "size": df.shape[0] // 2,
            }
        )
    )
    print(cells)
    # choose a face
    if face is None:
        # faces_ = faces[(faces['segment'] == 'apical') | (faces['segment'] == 'basal')]
        face = np.random.choice(faces.index)

    pair = faces[faces["verts"] == faces.loc[face, "verts"]].index
    print(pair)
    # Take the cell adjacent to the face with the smallest size
    cell = cells.loc[faces.loc[pair, "cell"], "size"].idxmin()
    face = pair[0] if pair[0] in cells.loc[cell, "faces"] else pair[1]
    elements = vert, face, cell
    print(vert,eptm.vert_df.loc[vert, 'segment'], face, eptm.face_df.loc[face, 'segment'], cell)
    print(eptm.face_df.loc[face])
    print(eptm.edge_df[eptm.edge_df['face']==face][['srce', 'trgt']])
    print("choose transition")

    if len(cells.iloc[np.where(cells['size'] == 4)].index) == 0:
        logger.info(f"OI for face {face} of cell {cell}")
        print("OI_transition")
        _OI_transition(eptm, all_edges, elements, multiplier, recenter=recenter)
    else:
        face = cells.iloc[np.where(cells['size'] == 4)].index[0]
        cell = cells.loc[faces.loc[pair, "cell"], "size"].idxmin()
        # face = pair[0] if pair[0] in cells.loc[cell, "faces"] else pair[1]
        elements = vert, face, cell
        logger.info(f"OI for face {face} of cell {cell}")
        print("OH_transition")
        _OH_transition(eptm, all_edges, elements, multiplier, recenter=recenter)

    # if cells.loc[cell, "size"] == 3:
    #     logger.info(f"OI for face {face} of cell {cell}")
    #     print("OI_transition")
    #     _OI_transition(eptm, all_edges, elements, multiplier, recenter=recenter)
    # elif cells.loc[cell, "size"] == 4:
    #     logger.info(f"OH for face {face} of cell {cell}")
    #     print('OH transition')
    #     _OH_transition(eptm, all_edges, elements, multiplier, recenter=recenter)
    # else:
    #     logger.info("Nothing happened ")
    #     return 1


    # Tidy up
    new_edges = []
    for face in all_edges["face"].unique():
        new_edge = close_face(eptm, face)
        if new_edge is not None:
            new_edges.append(new_edge)
    eptm.reset_index()
    eptm.reset_topo()
    print(new_edges)

    # for cell in all_edges["cell"].unique():
    for cell in range(eptm.Nc):
        try:
            close_cell(eptm, cell)
        except ValueError as e:
            print("close failed for cell")
            logger.error(f"Close failed for cell {cell}")
            raise e

    eptm.reset_index()
    eptm.reset_topo()

    if isinstance(eptm, Monolayer):
        for vert_ in eptm.vert_df.index[-2:]:
            eptm.guess_vert_segment(vert_)
        for face_ in eptm.face_df.index[-2:]:
            eptm.guess_face_segment(face_)

    eptm.reset_index()
    eptm.reset_topo()

    # #remove unconected vert # a supprimer cette partie
    # val, count = np.unique(eptm.edge_df['srce'], return_counts=True)
    # vert_id = val[np.where(count == 4)]
    # eptm.vert_df = eptm.vert_df.drop(vert_id, axis=1)
    # for face in all_edges["face"].unique():
    #     new_edge = close_face(eptm, face)
    #     if new_edge is not None:
    #         new_edges.append(new_edge)
    # eptm.reset_index()
    # eptm.reset_topo()
    # import sys
    # sys.exit()
    print('END split_vert bulk topology')
    return 0


def _OI_transition(eptm, all_edges, elements, multiplier=1.5, recenter=False,
                   cell_A=None, cell_B=None, cell_C=None, cell_D=None):
    print('inside OI')
    eptm.get_opposite_faces()
    epsilon = eptm.settings.get("threshold_length", 0.1) * multiplier
    vert, face, cell = elements
    print(vert, face, eptm.face_df.loc[face, 'segment'], cell)
    print("all_edges")
    print(all_edges[['srce', 'trgt', 'face', 'segment', 'cell']])
    if ('apical' not in all_edges['segment']) & ('basal' not in all_edges['segment']):
        return
    # Get cell and assign a "function"
    cells = np.unique(all_edges["cell"])
    if cell_A is None:
        #if more than 2 cells are in the border => abort
        val, count = np.unique(all_edges['is_border'], return_counts=True)
        print(val, count)
        if True in val:
            print("true in count")
            print(count[np.where(val==True)])
            if count[np.where(val==True)] >= 10:
                print('abort border')
                return
    a_or_b_edges = all_edges[(all_edges['segment'] == "apical") |
                             (all_edges['segment'] == "basal")]

    if cell_A is None :
        if len(cells) == 4:
            print("np_cell")
            print(cells)

            cell_C = cell

            srce = a_or_b_edges[(a_or_b_edges['cell'] == cell)].iloc[0]['srce']
            trgt = a_or_b_edges[(a_or_b_edges['cell'] == cell)].iloc[0]['trgt']
            cell_A = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                                  (a_or_b_edges['trgt'] == srce)]['cell'].to_numpy()[0]

            srce = a_or_b_edges[(a_or_b_edges['cell'] == cell)].iloc[1]['srce']
            trgt = a_or_b_edges[(a_or_b_edges['cell'] == cell)].iloc[1]['trgt']
            cell_B = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                                  (a_or_b_edges['trgt'] == srce)]['cell'].to_numpy()[0]

            cell_D = list(set(cells).symmetric_difference(set([cell_A, cell_B, cell_C,])))[0]

        elif len(cells) > 4:
            print("np_cell")
            print(cells)
            a_or_b_edges = all_edges[(all_edges['segment'] == "apical") |
                                     (all_edges['segment'] == "basal")]
            cell_C = cell

            srce = a_or_b_edges[(a_or_b_edges['cell'] == cell_C)].iloc[0]['srce']
            trgt = a_or_b_edges[(a_or_b_edges['cell'] == cell_C)].iloc[0]['trgt']
            cell_A = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                                   (a_or_b_edges['trgt'] == srce)]['cell'].to_numpy()[0]

            srce = a_or_b_edges[(a_or_b_edges['cell'] == cell_C)].iloc[1]['srce']
            trgt = a_or_b_edges[(a_or_b_edges['cell'] == cell_C)].iloc[1]['trgt']
            cell_B = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                                  (a_or_b_edges['trgt'] == srce)]['cell'].to_numpy()[0]

            srce = a_or_b_edges[(a_or_b_edges['cell'] == cell_B)].iloc[0]['srce']
            trgt = a_or_b_edges[(a_or_b_edges['cell'] == cell_B)].iloc[0]['trgt']
            cell_D = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                                  (a_or_b_edges['trgt'] == srce)]['cell'].to_numpy()[0]
            if cell_D == cell_B:
                srce = a_or_b_edges[(a_or_b_edges['cell'] == cell_B)].iloc[1]['srce']
                trgt = a_or_b_edges[(a_or_b_edges['cell'] == cell_B)].iloc[1]['trgt']
                cell_D = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                                      (a_or_b_edges['trgt'] == srce)]['cell'].to_numpy()[0]
            cell_A = [cell_A]
            cell_A.append(list(set(cells).symmetric_difference(set([cell_A[0], cell_B, cell_C, cell_D]))))

        elif len(cells) == 3:
            return
            print("np_cell")
            print(cells)

            center_cell = cells[0]
            if len(a_or_b_edges[(a_or_b_edges['cell']==cells[1]) &
                             ((a_or_b_edges['srce']==a_or_b_edges[a_or_b_edges['cell']==center_cell].iloc[0]['trgt']) &
                              (a_or_b_edges['trgt'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0]['srce'])) |
                             ((a_or_b_edges['srce'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1]['trgt']) &
                              (a_or_b_edges['trgt'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1]['srce']))
                             ]) == 1 & len(a_or_b_edges[(a_or_b_edges['cell'] == cells[2]) &
                                 ((a_or_b_edges['srce'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0][
                                     'trgt']) &
                                  (a_or_b_edges['trgt'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0][
                                      'srce'])) |
                                 ((a_or_b_edges['srce'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1][
                                     'trgt']) &
                                  (a_or_b_edges['trgt'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1][
                                      'srce']))
                                 ]) == 1 :
                cell_C = center_cell
                cell_A = cells[1]
                cell_B = cells[2]
            else:
                center_cell = cells[1]
                if len(a_or_b_edges[(a_or_b_edges['cell'] == cells[0]) &
                                    ((a_or_b_edges['srce'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0][
                                        'trgt']) &
                                     (a_or_b_edges['trgt'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0][
                                         'srce'])) |
                                    ((a_or_b_edges['srce'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1][
                                        'trgt']) &
                                     (a_or_b_edges['trgt'] == a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1][
                                         'srce']))
                       ]) == 1 & len(a_or_b_edges[(a_or_b_edges['cell'] == cells[2]) &
                                                  ((a_or_b_edges['srce'] ==
                                                    a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0][
                                                        'trgt']) &
                                                   (a_or_b_edges['trgt'] ==
                                                    a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[0][
                                                        'srce'])) |
                                                  ((a_or_b_edges['srce'] ==
                                                    a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1][
                                                        'trgt']) &
                                                   (a_or_b_edges['trgt'] ==
                                                    a_or_b_edges[a_or_b_edges['cell'] == center_cell].iloc[1][
                                                        'srce']))
                                     ]) == 1:
                    cell_C = center_cell
                    cell_A = cells[0]
                    cell_B = cells[2]
                else:
                    cell_C = cells[2]
                    cell_A = cells[0]
                    cell_B = cells[1]
            cell_D = np.NaN

    print("CELL")
    print(cell_A, cell_B, cell_C, cell_D)

    # Add all edges from cell B
    connected = all_edges[
        all_edges['cell'] == cell_B]
    print(connected[['srce', 'trgt', 'face', 'segment', 'cell']])

    # Add apical or basal edge for cell C
    srce = a_or_b_edges[a_or_b_edges['cell'] == cell_B].iloc[0]['srce']
    trgt = a_or_b_edges[a_or_b_edges['cell'] == cell_B].iloc[0]['trgt']
    new_edges = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                             (a_or_b_edges['trgt'] == srce)]
    connected = pd.concat((connected, new_edges))
    print(connected[['srce', 'trgt', 'face', 'segment', 'cell']])

    # Add apical or basal edge for cell D
    srce = a_or_b_edges[a_or_b_edges['cell'] == cell_B].iloc[1]['srce']
    trgt = a_or_b_edges[a_or_b_edges['cell'] == cell_B].iloc[1]['trgt']
    new_edges = a_or_b_edges[(a_or_b_edges['srce'] == trgt) &
                             (a_or_b_edges['trgt'] == srce)]
    connected = pd.concat((connected, new_edges))
    print(connected[['srce', 'trgt', 'face', 'segment', 'cell']])

    cell_B_lat_faces = connected[(connected['cell'] == cell_B) &
                                 (connected['segment'] == 'lateral')]['face']
    print("cell_B_lat_faces")
    print(cell_B_lat_faces)
    for f in cell_B_lat_faces:
        opp_face = eptm.face_df.loc[f, 'opposite']
        new_edges = all_edges[(all_edges['face'] == opp_face)]
        connected = pd.concat((connected, new_edges))
        print(connected[['srce', 'trgt', 'face', 'segment', 'cell']])



    # Get all the edges bordering this terahedron
    # cell_eges = eptm.edge_df.query(f"cell == {cell}")
    # prev_vs = cell_eges[cell_eges["trgt"] == vert]["srce"]
    # next_vs = cell_eges[cell_eges["srce"] == vert]["trgt"]
    #
    # connected = all_edges[
    #     all_edges["trgt"].isin(next_vs)
    #     | all_edges["srce"].isin(prev_vs)
    #     | all_edges["srce"].isin(next_vs)
    #     | all_edges["trgt"].isin(prev_vs)
    # ]
    print('connected')
    print(connected[['srce', 'trgt', 'face', 'segment', 'cell']])

    r_ia = eptm.face_df.loc[a_or_b_edges[(a_or_b_edges['cell'] == cell_B) &
                                         ((a_or_b_edges['segment']=='apical')|
                                          (a_or_b_edges['segment']=='basal'))]['face'].to_numpy()[0], eptm.coords] - eptm.vert_df.loc[vert, eptm.coords]
    shift = r_ia * epsilon / np.linalg.norm(r_ia)
    base_split_vert(eptm, vert, face, connected, epsilon, recenter, shift=shift)


def _OH_transition(eptm, all_edges, elements, multiplier=1.5, recenter=False):
    print('start oh_transition')
    epsilon = eptm.settings.get("threshold_length", 0.1) * multiplier
    vert, face, cell = elements

    # Get cell and assign a "function"
    cells = np.unique(all_edges["cell"])
    # if more than 2 cells are in the border => abort
    val, count = np.unique(all_edges['is_border'], return_counts=True)
    print(val, count)
    if True in val:
        print("true in count")
        print(count[np.where(val == True)])
        if count[np.where(val == True)] >= 10:
            print('abort border')
            return

    if len(cells) == 4:
        print("np_cell")
        print(cells)
        count_segment = all_edges[all_edges['segment'] == 'lateral'].value_counts('cell')
        # Les cellules à 4 cotés sont C et D
        cell_C, cell_D = count_segment[count_segment == 4].index
        # Les cellules à 6 cotés sont A et B
        cell_A, cell_B = count_segment[count_segment == 6].index
    else:
        return
    print("CELL")
    print(cell_A, cell_B, cell_C, cell_D)

    # 1-Create junction AB connected to lateral triangle cell and apical/basal surface
    cell_A_lateral_face = np.unique(all_edges[(all_edges['cell']==cell_A) & (all_edges['segment']=='lateral') ][['srce', 'trgt', 'face', 'segment', 'cell']]['face'])
    sides_3_lateral_face = eptm.face_df.loc[cell_A_lateral_face][eptm.face_df.loc[cell_A_lateral_face]['num_sides'] == 3].index[0]
    connected = all_edges[all_edges['segment']=='lateral']

    r_ia = eptm.face_df.loc[sides_3_lateral_face, eptm.coords] - \
           eptm.vert_df.loc[vert, eptm.coords]
    shift = r_ia * epsilon / np.linalg.norm(r_ia)
    base_split_vert(eptm, vert, face, connected, epsilon, recenter, shift=shift)

    ## Should be place differently if it is work...
    # Tidy up
    new_edges = []
    for face in all_edges["face"].unique():
        new_edge = close_face(eptm, face)
        if new_edge is not None:
            new_edges.append(new_edge)
    eptm.reset_index()
    eptm.reset_topo()
    print(new_edges)

    # for cell in all_edges["cell"].unique():
    for cell in range(eptm.Nc):
        try:
            close_cell(eptm, cell)
        except ValueError as e:
            print("close failed for cell")
            logger.error(f"Close failed for cell {cell}")
            raise e

    eptm.reset_index()
    eptm.reset_topo()

    if isinstance(eptm, Monolayer):
        for vert_ in eptm.vert_df.index[-2:]:
            eptm.guess_vert_segment(vert_)
        for face_ in eptm.face_df.index[-2:]:
            eptm.guess_face_segment(face_)

    eptm.reset_index()
    eptm.reset_topo()

    print(sides_3_lateral_face)
    print(eptm.face_df.loc[sides_3_lateral_face, 'num_sides'])
    print(eptm.face_df.loc[sides_3_lateral_face, 'opposite'])
    print(eptm.face_df.loc[eptm.face_df.loc[sides_3_lateral_face, 'opposite'], 'num_sides'])
    # call _OI?
    print("New_all_edges")
    print(all_edges.index)
    # all_edges = all_edges.drop(all_edges[all_edges['face'] == sides_3_lateral_face].index)
    # all_edges = all_edges.drop(all_edges[all_edges['face'] == eptm.face_df.loc[sides_3_lateral_face, 'opposite']].index)

    all_edges = eptm.edge_df[
        (eptm.edge_df["trgt"] == vert) | (eptm.edge_df["srce"] == vert)
        ]
    print(all_edges.index)
    _OI_transition(eptm, all_edges, elements, multiplier=5, recenter=False)



    # # all_cell_edges = eptm.edge_df.query(f'cell == {cell}').copy()
    # cell_edges = all_edges.query(f"cell == {cell}").copy()
    #
    # face_verts = cell_edges.groupby("face").apply(
    #     lambda df: set(df["srce"]).union(df["trgt"]) - {vert}
    # )
    #
    # for face_, verts_ in face_verts.items():
    #     if not verts_.intersection(face_verts.loc[face]):
    #         opp_face = face_
    #         break
    # else:
    #     raise ValueError
    #
    # for to_split in (face, opp_face):
    #     face_edges = all_edges.query(f"face == {to_split}").copy()
    #
    #     (prev_v,) = face_edges[face_edges["trgt"] == vert]["srce"]
    #     (next_v,) = face_edges[face_edges["srce"] == vert]["trgt"]
    #     connected = all_edges[
    #         all_edges["trgt"].isin((next_v, prev_v))
    #         | all_edges["srce"].isin((next_v, prev_v))
    #     ]
    #     base_split_vert(eptm, vert, to_split, connected, epsilon, recenter)


def get_division_edges(
    eptm, mother, plane_normal, plane_center=None, return_verts=False
):
    """Returns an index of the mother cell edges crossed by the division plane, ordered
    clockwize around the division plane normal.



    """
    if plane_normal is None:
        plane_normal = np.random.normal(size=3)

    plane_normal = np.asarray(plane_normal)
    if plane_center is None:
        plane_center = eptm.cell_df.loc[mother, eptm.coords]

    n_xy = np.linalg.norm(plane_normal[:2])
    theta = -np.arctan2(n_xy, plane_normal[2])
    if np.linalg.norm(plane_normal[:2]) < 1e-10:
        rot = None
    else:
        direction = [plane_normal[1], -plane_normal[0], 0]
        rot = rotation_matrix(theta, direction)

    cell_verts = frozenset(eptm.edge_df[eptm.edge_df["cell"] == mother]["srce"])
    vert_pos = eptm.vert_df.loc[cell_verts, eptm.coords]
    for coord in eptm.coords:
        vert_pos[coord] -= plane_center[coord]
    if rot is not None:
        vert_pos[:] = np.dot(vert_pos, rot)

    mother_edges = eptm.edge_df[eptm.edge_df["cell"] == mother]
    srce_z = vert_pos.loc[mother_edges["srce"], "z"]
    srce_z.index = mother_edges.index
    trgt_z = vert_pos.loc[mother_edges["trgt"], "z"]
    trgt_z.index = mother_edges.index
    division_edges = mother_edges[((srce_z < 0) & (trgt_z >= 0))]
    mother_verts = mother_edges[(srce_z < 0) & (trgt_z < 0)]["srce"].unique()
    daughter_verts = mother_edges[(srce_z >= 0) & (trgt_z >= 0)]["srce"].unique()

    # Order the returned edges so that their centers
    # are oriented counterclockwize in the division plane
    # in preparation for septum creation
    srce_pos = vert_pos.loc[division_edges["srce"], eptm.coords].values
    trgt_pos = vert_pos.loc[division_edges["trgt"], eptm.coords].values
    centers = (srce_pos + trgt_pos) / 2
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    if not return_verts:
        return division_edges.iloc[np.argsort(theta)].index
    return division_edges.iloc[np.argsort(theta)].index, mother_verts, daughter_verts


def get_division_vertices(
    eptm,
    division_edges=None,
    mother=None,
    plane_normal=None,
    plane_center=None,
    return_all=False,
):
    if division_edges is None:
        division_edges, mother_verts, daughter_verts = get_division_edges(
            eptm, mother, plane_normal, plane_center, return_verts=True
        )
    else:
        return_all = False

    septum_vertices = []
    for edge in division_edges:
        new_vert, *_ = add_vert(eptm, edge)
        septum_vertices.append(new_vert)
    if not return_all:
        return septum_vertices
    return septum_vertices, mother_verts, daughter_verts


# @check_condition4
def cell_division(
    eptm, mother, geom, vertices=None, mother_verts=None, daughter_verts=None
):
    if vertices is None:
        vertices, mother_verts, daughter_verts = get_division_vertices(
            eptm,
            mother=mother,
            return_all=True,
        )
    cell_cols = eptm.cell_df.loc[mother:mother]

    eptm.cell_df = pd.concat([eptm.cell_df, cell_cols], ignore_index=True)
    eptm.cell_df.index.name = "cell"
    daughter = eptm.cell_df.index[-1]
    if "id" not in eptm.cell_df.columns:
        warnings.warn(
            """Adding 'id' columns to cell_df, as dataframe index is not a reliable
identifier. Consider doing this at initialisation time
    """
        )
        eptm.cell_df["id"] = eptm.cell_df.index.copy()

    daughter_id = eptm.cell_df.id.max() + 1
    mother_id = eptm.cell_df.loc[mother, "id"]

    eptm.cell_df.loc[daughter, "id"] = daughter_id
    pairs = {
        frozenset([v1, v2])
        for v1, v2 in itertools.product(vertices, vertices)
        if v1 != v2
    }

    # divide existing faces-
    daughter_faces = []

    for v1, v2 in pairs:
        v1_faces = eptm.edge_df[eptm.edge_df["srce"] == v1]["face"]
        v2_faces = eptm.edge_df[eptm.edge_df["srce"] == v2]["face"]
        # we should devide a face if both v1 and v2
        # are part of it
        faces = set(v1_faces).intersection(v2_faces)
        for face in faces:
            daughter_faces.append(face_division(eptm, face, v1, v2))

    # septum
    face_cols = eptm.face_df.iloc[-2:]
    eptm.face_df = pd.concat([eptm.face_df, face_cols], ignore_index=True)
    eptm.face_df.index.name = "face"
    septum = eptm.face_df.index[-2:]

    num_v = len(vertices)
    num_new_edges = num_v * 2

    edge_cols = eptm.edge_df.iloc[-num_new_edges:]

    eptm.edge_df = pd.concat([eptm.edge_df, edge_cols], ignore_index=True)
    eptm.edge_df.index.name = "edge"
    new_edges = eptm.edge_df.index[-num_new_edges:]

    # To keep mother orientation, the first septum face
    # belongs to mother
    for v1, v2, edge, oppo in zip(
        vertices, np.roll(vertices, -1), new_edges[:num_v], new_edges[num_v:]
    ):
        # Mother septum
        eptm.edge_df.loc[edge, ["srce", "trgt", "face", "cell"]] = (
            v1,
            v2,
            septum[0],
            mother,
        )
        # Daughter septum
        eptm.edge_df.loc[oppo, ["srce", "trgt", "face", "cell"]] = (
            v2,
            v1,
            septum[1],
            daughter,
        )

    if (mother_verts is not None) and (daughter_verts is not None):
        # assign edges linked to daughter verts to daughter
        daughter_faces = eptm.edge_df.loc[
            eptm.edge_df["srce"].isin(daughter_verts) & (eptm.edge_df["cell"] == mother)
        ]["face"].unique()

        eptm.edge_df.loc[eptm.edge_df["face"].isin(daughter_faces), "cell"] = daughter
        eptm.edge_df.loc[eptm.edge_df["face"] == septum[1], "cell"] = daughter
        eptm.reset_index()
        eptm.reset_topo()
        geom.update_all(eptm)

    else:
        warnings.warn(
            "This method in cell_division is deprecated and can produce inconsistencies"
        )
        eptm.reset_index()
        eptm.reset_topo()
        geom.update_all(eptm)

        m_septum_edges = eptm.edge_df[eptm.edge_df["face"] == septum[0]]
        m_septum_norm = m_septum_edges[eptm.ncoords].mean()
        m_septum_pos = eptm.face_df.loc[septum[0], eptm.coords]
        if eptm.cell_df[eptm.cell_df["id"] == mother_id].index[0] != mother:
            raise RuntimeError

        # splitting the faces between mother and daughter
        # based on the orientation of the vector from septum
        # center to each face center w/r to the septum norm
        mother_faces = set(eptm.edge_df[eptm.edge_df["cell"] == mother]["face"])
        for face in mother_faces:
            if face == septum[0]:
                continue

            dr = eptm.face_df.loc[face, eptm.coords] - m_septum_pos
            proj = (dr.values * m_septum_norm).sum(axis=0)
            f_edges = eptm.edge_df[eptm.edge_df["face"] == face].index
            if proj < 0:
                eptm.edge_df.loc[f_edges, "cell"] = mother
            else:
                eptm.edge_df.loc[f_edges, "cell"] = daughter

        eptm.reset_index()
        eptm.reset_topo()
    return daughter


def find_rearangements(eptm):
    """Finds the candidates for IH and HI transitions
    Returns
    -------
    edges_HI: set of indexes of short edges
    faces_IH: set of indexes of small triangular faces
    """
    l_th = eptm.settings.get("threshold_length", 1e-2)
    shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return np.array([]), np.array([])
    edges_IH = find_IHs(eptm, shorts)
    faces_HI = find_HIs(eptm, shorts)
    return edges_IH, faces_HI


def find_IHs(eptm, shorts=None):

    l_th = eptm.settings.get("threshold_length", 1e-2)
    if shorts is None:
        shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return []

    edges_IH = shorts.groupby("srce").apply(
        lambda df: pd.Series(
            {
                "edge": df.index[0],
                "length": df["length"].iloc[0],
                "num_sides": min(eptm.face_df.loc[df["face"], "num_sides"]),
                "pair": frozenset(df.iloc[0][["srce", "trgt"]]),
            }
        )
    )
    # keep only one of the edges per vertex pair and sort by length
    edges_IH = (
        edges_IH[edges_IH["num_sides"] > 3]
        .drop_duplicates("pair")
        .sort_values("length")
    )
    return edges_IH["edge"].values


def find_HIs(eptm, shorts=None):
    l_th = eptm.settings.get("threshold_length", 1e-2)
    if shorts is None:
        shorts = eptm.edge_df[(eptm.edge_df["length"] < l_th)]
    if not shorts.shape[0]:
        return []

    max_f_length = shorts.groupby("face")["length"].apply(max)
    short_faces = eptm.face_df.loc[max_f_length[max_f_length < l_th].index]
    faces_HI = short_faces[short_faces["num_sides"] == 3].sort_values("area").index
    return faces_HI


# # @check_condition4
# def IH_transition(eptm, edge, recenter=False):
#     """
#     I → H transition as defined in Okuda et al. 2013
#     (DOI 10.1007/s10237-012-0430-7).
#     See tyssue/doc/illus/IH_transition.png for the algorithm
#     """
#     print('ih transition')
#     srce, trgt, face, cell = eptm.edge_df.loc[edge, ["srce", "trgt", "face", "cell"]]
#     vert = min(srce, trgt)
#     collapse_edge(eptm, edge)
#
#     split_vert(eptm, vert, face, recenter=recenter)
#
#     logger.info("IH transition on edge %d", edge)
#     return 0
#
#
# # @check_condition4
# def HI_transition(eptm, face, recenter=False):
#     """
#     H → I transition as defined in Okuda et al. 2013
#     (DOI 10.1007/s10237-012-0430-7).
#     See tyssue/doc/illus/IH_transition.png for the algorithm
#     """
#     remove_face(eptm, face)
#     vert = eptm.vert_df.index[-1]
#     all_edges = eptm.edge_df[
#         (eptm.edge_df["srce"] == vert) | (eptm.edge_df["trgt"] == vert)
#     ]
#
#     cells = all_edges.groupby("cell").size()
#     cell = cells.idxmin()
#     face = all_edges[all_edges["cell"] == cell]["face"].iloc[0]
#     split_vert(eptm, vert, face, recenter=recenter)
#
#     logger.info("HI transition on face %d", face)
#     return 0


def fix_pinch(eptm):
    """Due to rearangements, some faces in an epithelium will have
    more than one opposite face.

    This method fixes the issue so we can have a valid epithelium back.
    """
    logger.debug("Fixing pinch")
    face_v = eptm.edge_df.groupby("face").apply(lambda df: frozenset(df["srce"]))
    face_v2 = pd.Series(data=face_v.index, index=face_v.values)
    grouped = face_v2.groupby(level=0)
    cardinal = grouped.apply(len)
    faces = face_v2[cardinal > 2].to_list()
    if not faces:
        logger.debug("no pinch found")
        return
    cells = eptm.edge_df.loc[eptm.edge_df["face"].isin(faces), "cell"].unique()
    bad_cells = []
    for cell in cells:
        if not _is_closed_cell(eptm.edge_df.query(f"cell == {cell}")):
            bad_cells.append(cell)

    logger.info("Fixing pinch for cells %s", bad_cells)
    to_remove = eptm.edge_df.loc[
        eptm.edge_df["face"].isin(faces) & (eptm.edge_df["cell"].isin(bad_cells))
    ]

    bad_faces = to_remove["face"].unique()
    bad_edges = to_remove.index.values

    eptm.edge_df = eptm.edge_df.drop(bad_edges)
    eptm.face_df = eptm.face_df.drop(bad_faces)
    eptm.reset_index()
    eptm.reset_topo()

