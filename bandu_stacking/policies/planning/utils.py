from collections import namedtuple

import numpy as np

X_AXIS = np.array([1, 0, 0])  # TODO: make immutable
Z_AXIS = np.array([0, 0, 1])

Plane = namedtuple("Plane", ["normal", "origin"])
