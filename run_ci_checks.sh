#!/bin/bash
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.bandu_stacking_pylintrc
./run_autoformat.sh
