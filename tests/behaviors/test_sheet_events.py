import os
import tempfile

import numpy as np
import pytest

from tyssue import config
from tyssue.behaviors.event_manager import EventManager, wait
from tyssue.behaviors.sheet.actions import ab_pull, decrease
from tyssue.behaviors.sheet.actions import exchange as type1_at_shorter
from tyssue.behaviors.sheet.actions import grow, increase, increase_linear_tension
from tyssue.behaviors.sheet.actions import remove as type3
from tyssue.behaviors.sheet.actions import set_value
from tyssue.behaviors.sheet.apoptosis_events import apoptosis
from tyssue.behaviors.sheet.basic_events import (
    check_tri_faces,
    contraction,
    division,
    face_elimination,
    type1_transition,
)
from tyssue.behaviors.sheet.delamination_events import constriction
from tyssue.core.sheet import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir


def test_add_events():

    manager = EventManager("face")
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (wait, {"face_id": 3, "n_steps": 4}),
        (apoptosis, {"face_id": 5}),
        (constriction, {"face_id": 2}),
    ]
    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 4


def test_add_only_once():
    manager = EventManager("face")
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (apoptosis, {"face_id": 3}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 2


def test_add_several_time_same_events():
    manager = EventManager("face")
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (contraction, {"face_id": 2, "unique": False}),
        (contraction, {"face_id": 3, "unique": False}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (contraction, {"face_id": 2, "unique": False}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 5


def test_logging():

    tf = tempfile.mktemp()
    manager = EventManager("face", tf)
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (apoptosis, {"face_id": 3}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    with open(tf, "r") as fh:
        l0, l1, l2 = fh.readlines()

    assert l2 == "0, -1, wait\n"


def test_execute_constriction():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    geom.update_all(sheet)
    sheet.settings["constriction"] = {
        "contractile_increase": 5,
        "critical_area": 5,
        "max_traction": 10,
    }
    sheet.face_df["id"] = sheet.face_df.index.values
    init_nb_cells = len(sheet.face_df)

    sheet.face_df["is_mesoderm"] = 0
    sheet.vert_df["radial_tension"] = 0
    manager = EventManager("face")
    face_id = 17
    sheet.face_df.loc[face_id, "is_mesoderm"] = 1

    sheet.settings["constriction"].update({"face_id": face_id})
    initial_cell_event = (constriction, sheet.settings["constriction"])
    manager.current.append(initial_cell_event)
    manager.execute(sheet)
    manager.update()
    assert len(manager.current) > 0

    for i in range(20):
        manager.execute(sheet)
        manager.update()
    if len(sheet.face_df) == init_nb_cells:
        assert len(manager.current) > 0


def test_execute_apoptosis():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    geom.update_all(sheet)
    sheet.settings["apoptosis"] = {"contractile_increase": 2.0, "critical_area": 0.1}
    sheet.face_df["id"] = sheet.face_df.index.values
    init_nb_faces = len(sheet.face_df)
    manager = EventManager("face")
    face_id = 17
    face_area = sheet.face_df.loc[face_id, "area"]
    initial_nbsides = sheet.face_df.loc[face_id, "num_sides"]

    sheet.settings["apoptosis"].update({"face_id": face_id})
    initial_cell_event = (apoptosis, sheet.settings["apoptosis"])

    manager.current.append(initial_cell_event)
    manager.execute(sheet)
    manager.update()
    assert len(manager.current) > 0
    manager.execute(sheet)
    manager.update()

    sheet.settings["apoptosis"] = {
        "contractile_increase": 2.0,
        "critical_area": 2 * face_area,
    }
    manager.current.clear()
    sheet.settings["apoptosis"].update({"face_id": face_id})
    modified_cell_event = (apoptosis, sheet.settings["apoptosis"])

    manager.current.append(modified_cell_event)
    manager.execute(sheet)
    manager.update()
    next_nbsides = sheet.face_df.loc[sheet.idx_lookup(face_id, "face"), "num_sides"]
    i = 1
    while next_nbsides > 3:
        assert next_nbsides == initial_nbsides - i
        assert len(manager.current) > 0
        i = i + 1
        manager.execute(sheet)
        manager.update()
        face = sheet.idx_lookup(face_id, "face")
        if face is not None:
            next_nbsides = sheet.face_df.loc[face, "num_sides"]
        else:
            break
        if i > 20:
            raise RecursionError
    manager.execute(sheet)
    manager.update()
    assert len(sheet.face_df) < init_nb_faces


def test_execute_division():

    sheet = Sheet("emin", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.face_df["volume"] = 1.0
    sheet.specs["face"]["prefered_volume"] = 1.0
    sheet.face_df["prefered_volume"] = 1.0

    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    face_id = 1
    event = (division, {"face_id": face_id, "growth_rate": 1.2, "critical_volume": 1.5})
    manager.current.append(event)
    V0 = sheet.face_df.loc[1, "prefered_volume"]
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[1, "prefered_volume"] == V0 * 1.2
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[1, "prefered_volume"] == V0 * 1.44
    sheet.face_df.loc[1, "volume"] *= 1.6
    manager.execute(sheet)
    assert sheet.Nf == 4


def test_execute_contraction():
    sheet = Sheet("emin", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.face_df["contractility"] = 1.12

    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    face_id = 1
    event = (
        contraction,
        {"face_id": face_id, "contractile_increase": 0.2, "critical_area": 1e-2},
    )
    manager.current.append(event)
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[face_id, "contractility"] == 1.32


def test_type1_transition():
    sheet = Sheet("emin", *three_faces_sheet())
    geom.update_all(sheet)

    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    face_id = 1
    nb_neighbors_start = len(sheet.edge_df[sheet.edge_df["face"] == face_id])
    edge_to_modify = sheet.edge_df[sheet.edge_df["face"] == face_id].index[0]
    sheet.edge_df.loc[edge_to_modify, "length"] = 0.2
    event = (type1_transition, {"face_id": face_id, "critical_length": 0.3})
    manager.current.append(event)
    manager.execute(sheet)
    manager.update()

    nb_neighbors_end = len(sheet.edge_df[sheet.edge_df["face"] == face_id])

    assert nb_neighbors_end == nb_neighbors_start - 1


def test_check_tri_faces():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    initial_nb_cells = len(sheet.face_df)
    nb_tri_cells = len(sheet.face_df[(sheet.face_df["num_sides"] < 4)])
    geom.update_all(sheet)
    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    manager.current.append((check_tri_faces, {}))
    manager.execute(sheet)
    manager.update()

    manager.execute(sheet)
    manager.update()
    assert len(sheet.face_df) == initial_nb_cells - nb_tri_cells


def test_face_elimination():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["id"] = sheet.face_df.index
    face_id = 0
    manager = EventManager("face")
    event = (face_elimination, {"face_id": face_id})
    manager.current.append(event)
    manager.execute(sheet)
    manager.update()
    assert sheet.Nf == 2
    assert sheet.Nv == 8
    assert sheet.Ne == 10


def test_type1_at_shorter():

    sheet = Sheet("emin", *three_faces_sheet())
    sheet.vert_df.loc[0, "x"] += 0.5
    geom.update_all(sheet)
    type1_at_shorter(sheet, 0, geom)
    np.all(sheet.face_df.num_sides.values == [5, 7, 5])


def test_remove_face():

    sheet = Sheet("emin", *three_faces_sheet())
    type3(sheet, 0, geom)
    assert sheet.Nf == 2
    assert sheet.Nv == 8
    assert sheet.Ne == 10


def test_ab_pull():

    sheet = Sheet("emin", *three_faces_sheet())
    sheet.vert_df["radial_tension"] = 1.0
    ab_pull(sheet, 0, 1.0, distributed=False)
    np.testing.assert_array_equal(
        sheet.vert_df.loc[0:5, "radial_tension"], np.ones(6) * 2.0
    )
    ab_pull(sheet, 0, 3.0, distributed=True)
    np.testing.assert_array_equal(
        sheet.vert_df.loc[0:5, "radial_tension"], np.ones(6) * 2.5
    )


def test_set_value():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_volume"] = 1.0
    set_value(sheet, "face", 0, 12.0, "prefered_volume")
    assert sheet.face_df.loc[0, "prefered_volume"] == 12.0
    assert sheet.face_df.loc[1, "prefered_volume"] == 1.0


def test_increase():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_volume"] = 1.0
    increase(sheet, "face", 0, 1.2, "prefered_volume")
    assert sheet.face_df.loc[0, "prefered_volume"] == 1.2


def test_decrease():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_volume"] = 1.0
    decrease(sheet, "face", 0, 1.6, "prefered_volume")
    assert sheet.face_df.loc[0, "prefered_volume"] == 0.625


def test_increase_line_tension():
    sheet = Sheet("emin", *three_faces_sheet())

    sheet.edge_df["line_tension"] = 1
    edges = sheet.edge_df[sheet.edge_df["face"] == 4]
    for index, edge in edges.iterrows():
        sheet.edge_df.loc[edge, "line_tension"] == 2

    increase_linear_tension(sheet, 0, 4)

    edges = sheet.edge_df[sheet.edge_df["face"] == 4]
    for index, edge in edges.iterrows():
        angle_ = np.arctan2(sheet.edge_df.dx, sheet.edge_df.dy)
        if np.abs(angle_) < np.pi / 4:
            assert sheet.edge_df.loc[edge, "line_tension"] == 8


def test_grow():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_volume"] = 1.0
    with pytest.warns(UserWarning):
        grow(sheet, 0, 1.2)
        assert sheet.face_df.loc[0, "prefered_volume"] == 1.2
