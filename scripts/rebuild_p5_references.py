"""Rebuild the cached P5 FEM references (results/reference/, ~35 MB, excluded from zips). ~1 minute."""
import sys, pathlib, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.problems.p2d import P5Film, P5FilmThick, P5FilmThin
jobs = [(P5Film, k) for k in (5, 20, 80, 400, 2000, 0.2, 0.05, 0.0125, 0.0025, 0.0005)] + \
       [(P5FilmThick, 80), (P5FilmThick, 2000), (P5FilmThin, 80), (P5FilmThin, 2000)]
t0 = time.time()
for c, k in jobs:
    s = c(float(k)).build_reference(); print(c.__name__, k, "dofs", s.dofs, f"{time.time() - t0:.0f}s", flush=True)
