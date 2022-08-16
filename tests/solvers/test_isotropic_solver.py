import os

from numpy.testing import assert_almost_equal

from tyssue import Sheet
from tyssue import SheetGeometry as geom
from tyssue import config
from tyssue.dynamics import SheetModel as model
from tyssue.io.hdf5 import load_datasets
from tyssue.solvers.isotropic_solver import bruteforce_isotropic_relax
from tyssue.stores import stores_dir

TOLERANCE = 1e-8


def test_iso_solver():

    pth = os.path.join(stores_dir, "rod_sheet.hf5")
    datasets = load_datasets(pth)
    specs = config.geometry.rod_sheet()
    sheet = Sheet("rod", datasets, specs)
    geom.reset_scafold(sheet)
    geom.update_all(sheet)

    dyn_specs = config.dynamics.quasistatic_sheet_spec()
    dyn_specs["vert"]["basal_shift"] = 0.0
    dyn_specs["face"]["prefered_volume"] = 1.0

    sheet.update_specs(dyn_specs, reset=True)
    geom.update_all(sheet)
    res = bruteforce_isotropic_relax(sheet, geom, model)
    assert_almost_equal(res["x"], 1.6001116383)
