import os
import time

import jax
import pymatgen as mg
from absl.testing import absltest, parameterized

from jax_md.a2c.crystallizer_utils import (
  get_subcells_to_crystallize,
  get_subcells_to_crystallize_parallel,
)

jax.config.update('jax_enable_x64', True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
AMORPHOUS_SI64_PATH = os.path.join(DATA_DIR, 'amorphous_si64.cif')


class A2CTest(parameterized.TestCase):
    """Test the A2C workflow."""


    def test_subcells_to_crystallize_parallel(self):
        """Test the parallel version of the subcells to crystallize function."""
        amorphous_structure = mg.core.Structure.from_file(AMORPHOUS_SI64_PATH)

        # Run both versions of the crystallize function, timing each, and compare the results
        d_frac = 0.07
        nmin = 1
        nmax = 16

        start_time = time.time()
        orig = get_subcells_to_crystallize(amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax)
        orig_time = time.time() - start_time

        start_time = time.time()
        par = get_subcells_to_crystallize_parallel(amorphous_structure, d_frac=d_frac, nmin=nmin, nmax=nmax, n_workers=os.cpu_count()-1)
        par_time = time.time() - start_time


        print(f"Serial crystallize function time   ({len(orig)} subcells):  {orig_time:.4f}s")
        print(f"Parallel crystallize function time ({len(par)} subcells):  {par_time:.4f}s")
        print(f"Parallelization speedup (CPUs: {os.cpu_count()-2}):\t{orig_time / par_time:.2f}x")

        # Sort by (tuple(ids), tuple(l), tuple(h)) and compare
        orig = sorted(orig, key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])))
        par = sorted(par, key=lambda x: (tuple(x[0]), tuple(x[1]), tuple(x[2])))
        self.assertEqual(
            len(orig),
            len(par),
            f"Original function and parallel function returned different number of subcells: {len(orig)} != {len(par)}",
        )

if __name__ == '__main__':
    absltest.main()
