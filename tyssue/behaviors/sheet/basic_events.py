"""
Small event module
=======================


"""
import logging
import random
import numpy as np
from itertools import count

from ...geometry.sheet_geometry import SheetGeometry
from ...topology.sheet_topology import cell_division
from ...utils.decorators import face_lookup
from .actions import (
    decrease,
    detach_vertices,
    exchange,
    increase,
    increase_linear_tension,
    merge_vertices,
    remove,
)
MAX_ITER = 10
logger = logging.getLogger(__name__)
from ...topology.bulk_topology import IH_transition, HI_transition, find_rearangements
def reconnect_3D(sheet, manager, **kwargs):
    sheet.get_opposite_faces()
    manager.append(reconnect_3D, **kwargs)

    # twist faces
    lateral_face_index = sheet.face_df[(sheet.face_df['segment'] == "lateral") &
                                       (sheet.face_df['num_sides'] == 4) &
                                       (sheet.face_df['opposite'] != -1)].index
    angle = []
    edges = []
    for f in lateral_face_index:
        if sheet.edge_df[(sheet.edge_df['face'] == f) &
                         (sheet.edge_df['dz'] < 0.2) &
                         (sheet.edge_df['dz'] > -0.2)].shape[0] == 2:
            angle.append(np.arctan2(sheet.edge_df[(sheet.edge_df['face'] == f) &
                                                  (sheet.edge_df['dz'] < 0.2) &
                                                  (sheet.edge_df['dz'] > -0.2)][['dx', 'dy']].diff(axis=0).iloc[1][
                                        'dy'],
                                    sheet.edge_df[(sheet.edge_df['face'] == f) &
                                                  (sheet.edge_df['dz'] < 0.2) &
                                                  (sheet.edge_df['dz'] > -0.2)][['dx', 'dy']].diff(axis=0).iloc[1][
                                        'dx']))
            edges.append(sheet.edge_df[(sheet.edge_df['face'] == f) &
                                       (sheet.edge_df['dz'] < 0.2) &
                                       (sheet.edge_df['dz'] > -0.2)].index[0])

    angle = np.array(angle) * 180 / np.pi

    angle = [180 + a if a < 0 else a for a in angle]
    angle = [180 - a if a > 90 else a for a in angle]
    angle = np.array(angle)
    twist_face = lateral_face_index[np.where(angle > 45)]
    twist_edges = np.array(edges)[np.where(angle > 45)]

    twist_face = twist_face[0::2]
    twist_edges = twist_edges[0::2]
    print("len twist edges")
    print(len(twist_edges))
    for e in twist_edges:
        print("twist edges")

        retcode = IH_transition(sheet, e)
        if not retcode:
            return 0



    edges, faces = find_rearangements(sheet)
    if len(edges):
        for i in count():
            if i == MAX_ITER:
                return 3
            retcode = IH_transition(sheet, np.random.choice(edges))
            if not retcode:
                return 0

    elif len(faces) and with_t3:
        for i in count():
            if i == MAX_ITER:
                return 3
            retcode = HI_transition(sheet, np.random.choice(faces))
            if not retcode:
                return 0
    return 1


    # # IH_transition
    # short = sheet.edge_df[sheet.edge_df["length"] < d_min].index.to_numpy()
    # already_done_edges = []
    # random.shuffle(short)
    # while (short.shape[0]) and (short[0] not in already_done_edges):
    #     IH_transition(sheet, short[0])
    #     already_done_edges.append(short[0])
    #     short = sheet.edge_df[sheet.edge_df["length"] < d_min].index.to_numpy()
    #     random.shuffle(short)
    #
    # # HI transition
    # short = sheet.edge_df[(sheet.edge_df["length"] < d_min)]
    # max_f_length = short.groupby("face")["length"].apply(max)
    # short_faces = sheet.face_df.loc[max_f_length[max_f_length < d_min].index]
    # faces_HI = short_faces[short_faces["num_sides"] == 3].sort_values("area").index
    # random.shuffle(faces_HI.to_numpy())
    # already_done_faces = []
    # while (faces_HI.shape[0]) and (faces_HI[0] not in already_done_faces):
    #     HI_transition(sheet, faces_HI[0])
    #     already_done_faces.append(faces_HI[0])
    #
    #     short = sheet.edge_df[(sheet.edge_df["length"] < d_min)]
    #     max_f_length = short.groupby("face")["length"].apply(max)
    #     short_faces = sheet.face_df.loc[max_f_length[max_f_length < d_min].index]
    #     faces_HI = short_faces[short_faces["num_sides"] == 3].sort_values("area").index
    #     random.shuffle(faces_HI.to_numpy())



def reconnect(sheet, manager, **kwargs):
    """Performs reconnections (vertex merging / splitting) following Finegan et al. 2019

    kwargs overwrite their corresponding `sheet.settings` entries

    Keyword Arguments
    -----------------
    threshold_length : the threshold length at which vertex merging is performed
    p_4 : the probability per unit time to perform a detachement from a rank 4 vertex
    p_5p : the probability per unit time to perform a detachement from a rank 5
        or more vertex


    See Also
    --------

    **The tricellular vertex-specific adhesion molecule Sidekick
    facilitates polarised cell intercalation during Drosophila axis
    extension** _Tara M Finegan, Nathan Hervieux, Alexander
    Nestor-Bergmann, Alexander G. Fletcher, Guy B Blanchard, Benedicte
    Sanson_ bioRxiv 704932; doi: https://doi.org/10.1101/704932

    """
    sheet.get_opposite_faces()
    sheet.settings.update(kwargs)
    nv = sheet.Nv
    merge_vertices(sheet)
    if nv != sheet.Nv:
        logger.info(f"Merged {nv - sheet.Nv+1} vertices")
    nv = sheet.Nv
    retval = detach_vertices(sheet)
    if retval:
        logger.info("Failed to detach, skipping")

    if nv != sheet.Nv:
        logger.info(f"Detached {sheet.Nv - nv} vertices")

    manager.append(reconnect, **kwargs)


default_division_spec = {
    "face_id": -1,
    "face": -1,
    "growth_rate": 0.1,
    "critical_volume": 2.0,
    "geom": SheetGeometry,
}


@face_lookup
def division(sheet, manager, **kwargs):
    """Cell division happens through cell growth up to a critical volume,
    followed by actual division of the face.

    Parameters
    ----------
    sheet : a `Sheet` object
    manager : an `EventManager` instance
    face_id : int,
      index of the mother face
    growth_rate : float, default 0.1
      rate of increase of the prefered volume
    critical_volume : float, default 2.
      volume at which the cells stops to grow and devides

    """
    division_spec = default_division_spec
    division_spec.update(**kwargs)

    face = division_spec["face"]

    division_spec["critical_volume"] *= sheet.specs["face"]["prefered_volume"]

    print(sheet.face_df.loc[face, "volume"], division_spec["critical_volume"])

    if sheet.face_df.loc[face, "volume"] < division_spec["critical_volume"]:
        increase(
            sheet, "face", face, division_spec["growth_rate"], "prefered_volume", True
        )
        manager.append(division, **division_spec)
    else:
        daughter = cell_division(sheet, face, division_spec["geom"])
        sheet.face_df.loc[daughter, "id"] = sheet.face_df.id.max() + 1


default_contraction_spec = {
    "face_id": -1,
    "face": -1,
    "contractile_increase": 1.0,
    "critical_area": 1e-2,
    "max_contractility": 10,
    "multiply": False,
    "contraction_column": "contractility",
    "unique": True,
}


@face_lookup
def contraction(sheet, manager, **kwargs):
    """Single step contraction event."""
    contraction_spec = default_contraction_spec
    contraction_spec.update(**kwargs)
    face = contraction_spec["face"]

    if (sheet.face_df.loc[face, "area"] < contraction_spec["critical_area"]) or (
        sheet.face_df.loc[face, contraction_spec["contraction_column"]]
        > contraction_spec["max_contractility"]
    ):
        return
    increase(
        sheet,
        "face",
        face,
        contraction_spec["contractile_increase"],
        contraction_spec["contraction_column"],
        contraction_spec["multiply"],
    )


default_type1_transition_spec = {
    "face_id": -1,
    "face": -1,
    "critical_length": 0.1,
    "geom": SheetGeometry,
}


@face_lookup
def type1_transition(sheet, manager, **kwargs):
    """Custom type 1 transition event that tests if
    the the shorter edge of the face is smaller than
    the critical length.
    """
    type1_transition_spec = default_type1_transition_spec
    type1_transition_spec.update(**kwargs)
    face = type1_transition_spec["face"]

    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    if min(edges["length"]) < type1_transition_spec["critical_length"]:
        exchange(sheet, face, type1_transition_spec["geom"])


default_face_elimination_spec = {"face_id": -1, "face": -1, "geom": SheetGeometry}


@face_lookup
def face_elimination(sheet, manager, **kwargs):
    """Removes the face with if face_id from the sheet."""
    face_elimination_spec = default_face_elimination_spec
    face_elimination_spec.update(**kwargs)
    remove(sheet, face_elimination_spec["face"], face_elimination_spec["geom"])


default_check_tri_face_spec = {"geom": SheetGeometry}


def check_tri_faces(sheet, manager, **kwargs):
    """Three neighbourghs cell elimination
    Add all cells with three neighbourghs in the manager
    to be eliminated at the next time step.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    """
    check_tri_faces_spec = default_check_tri_face_spec
    check_tri_faces_spec.update(**kwargs)

    tri_faces = sheet.face_df[(sheet.face_df["num_sides"] < 4)].id
    manager.extend(
        [
            (face_elimination, {"face_id": f, "geom": check_tri_faces_spec["geom"]})
            for f in tri_faces
        ]
    )


default_contraction_line_tension_spec = {
    "face_id": -1,
    "face": -1,
    "shrink_rate": 1.05,
    "contractile_increase": 1.0,
    "critical_area": 1e-2,
    "max_contractility": 10,
    "multiply": True,
    "contraction_column": "line_tension",
    "unique": True,
}


@face_lookup
def contraction_line_tension(sheet, manager, **kwargs):
    """
    Single step contraction event
    """
    contraction_spec = default_contraction_line_tension_spec
    contraction_spec.update(**kwargs)
    face = contraction_spec["face"]

    if sheet.face_df.loc[face, "area"] < contraction_spec["critical_area"]:
        return

    # reduce prefered_area
    decrease(
        sheet,
        "face",
        face,
        contraction_spec["shrink_rate"],
        col="prefered_area",
        divide=True,
        bound=contraction_spec["critical_area"] / 2,
    )

    increase_linear_tension(
        sheet,
        face,
        contraction_spec["contractile_increase"],
        multiply=contraction_spec["multiply"],
        isotropic=True,
        limit=100,
    )
