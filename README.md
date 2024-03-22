# bandu-stacking
A library for experimenting with bandu stacking

## Installation
(Note: Performing the installation in a separate virtualenv or conda environment is
recommended!)
To install the package, run 
```
pip install -e .
```

If you're doing experiments on the robot, make sure you also compile
```
cd bandu_stacking/inverse_kinematics/franka_panda/
python setup.py
```


## Run an Experiment

`python run_experiment.py --object-set=<bandu|blocks> --algorithm=<random|skeleton_planner>`

## Contributing
- Run `pip install -e ".[develop]"` to install all dependencies for development.
- You can't push directly to master. Make a new branch in this repository (don't use a fork, since that will not properly trigger the checks when you make a PR). When your code is ready for review, make a PR and request reviews from the appropriate people.
- To merge a PR, you need at least one approval, and you have to pass the 2 checks defined in `.github/workflows/ci.yml`, which you can run automatically with `./run_ci_checks.sh`, or individually as follows:
- `mypy .`

- `pytest . --pylint -m pylint --pylint-rcfile=.bandu_stacking_pylintrc`

- `./run_autoformat.sh`

