#!/bin/sh
# Recreate the sandbox environment used for Phases 1-10 (Ubuntu 24, Python 3.12, CPU only).
set -e
pip install "jax[cpu]==0.11.2" optax==0.2.8 scikit-fem==12.0.2 gmsh==4.15.2 pytest==9.1.1 pyyaml sympy --break-system-packages -q
# gmsh needs these system libraries (archive.ubuntu.com is reachable from the sandbox)
(sudo -n apt-get install -y -q libglu1-mesa libxcursor1 libxinerama1 libxft2 libxrender1 || apt-get install -y -q libglu1-mesa libxcursor1 libxinerama1 libxft2 libxrender1) > /tmp/apt.log 2>&1
python -c "import jax, optax, skfem, gmsh; gmsh.initialize(); print('jax', jax.__version__, 'optax', optax.__version__, 'skfem', skfem.__version__, 'gmsh', gmsh.__version__); gmsh.finalize()"
