import numpy as np


def get_custom_intrinsics(camera_name):
    mtx = np.array(
        [
            [1342.451693, 0.0, 1005.799801],
            [0.0, 1339.678018, 545.304949],
            [0.0, 0.0, 1.0],
        ]
    )

    dist = np.array([[0.069644, -0.154332, -0.004702, 0.004893, 0.000000]])
    return mtx, dist
