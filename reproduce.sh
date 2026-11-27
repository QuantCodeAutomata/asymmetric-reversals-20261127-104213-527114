#!/usr/bin/env bash
# Runs the test suite and every experiment from the repository root.
set -e
python -m pytest
python -m exp.exp1_decomposition_fit
python -m exp.exp2_res_sys_sorts
python -m exp.exp3_ivol_air
python -m exp.exp4_zoo_spanning
python -m exp.exp5_irf
python -m exp.exp6_asres
