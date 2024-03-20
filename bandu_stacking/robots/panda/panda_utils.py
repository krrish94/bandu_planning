import os

import numpy as np
from pybullet_tools.ikfast.utils import IKFastInfo
from pybullet_tools.utils import (
    CameraImage,
    get_link_pose,
    joint_from_name,
    link_from_name,
    multiply,
    set_joint_position,
)

from bandu_stacking.policies.simulation.controller import SimulatedController
from bandu_stacking.policies.simulation.entities import Camera, Manipulator, Robot
from bandu_stacking.policies.simulation.lis import (
    CAMERA_MATRIX as SIMULATED_CAMERA_MATRIX,)
# from bandu_stacking.policies.simulation.lis import (
#     ,
# )
from bandu_stacking.policies.simulation.utils import save_camera_images
from bandu_stacking.robots.panda.panda_controller import PandaController

# from run_estimator import *




CAMERA_FRAME = "camera_frame"
CAMERA_OPTICAL_FRAME = "camera_frame"
PANDA_INFO = IKFastInfo(
    module_name="franka_panda.ikfast_panda_arm",
    base_link="panda_link0",
    ee_link="panda_link8",
    free_joints=["panda_joint7"],
)
PANDA_PATH = os.path.abspath(
    "bandu_stacking/models/srl/franka_description/robots/panda_arm_hand.urdf"
)


class PandaRobot(Robot):
    def __init__(
        self,
        robot_body,
        link_names={},
        client=None,
        real_camera=False,
        real_execute=False,
        *args,
        **kwargs,
    ):
        self.link_names = link_names
        self.body = robot_body
        self.client = client
        self.real_camera = real_camera
        self.real_execute = real_execute

        if self.real_execute:
            self.controller = PandaController(self, client=self.client)
        else:
            self.controller = SimulatedController(self, client=self.client)

        self.arms = ["main_arm"]

        PANDA_GROUPS = {
            "base": [],
            "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
            "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
        }

        PANDA_TOOL_FRAMES = {
            "main_arm": "panda_tool_tip",  # l_gripper_palm_link | l_gripper_tool_frame
        }

        panda_manipulators = {
            side_from_arm(arm): Manipulator(
                arm, gripper_from_arm(arm), PANDA_TOOL_FRAMES[arm]
            )
            for arm in self.arms
        }
        panda_ik_infos = {side_from_arm(arm): PANDA_INFO for arm in self.arms}

        if not self.real_camera:
            cameras = [
                Camera(
                    self,
                    link=link_from_name(self.body, CAMERA_FRAME, client=client),
                    optical_frame=link_from_name(
                        self.body, CAMERA_OPTICAL_FRAME, client=client
                    ),
                    camera_matrix=SIMULATED_CAMERA_MATRIX,
                    client=client,
                )
            ]
        else:
            cameras = []

        super(PandaRobot, self).__init__(
            robot_body,
            ik_info=panda_ik_infos,
            manipulators=panda_manipulators,
            cameras=cameras,
            joint_groups=PANDA_GROUPS,
            link_names=link_names,
            client=client,
            *args,
            **kwargs,
        )
        self.max_depth = 3.0
        self.min_z = 0.0
        self.BASE_LINK = "panda_link0"
        self.MAX_PANDA_FINGER = 0.045

    def get_default_conf(self):
        conf = {
            "main_arm": [
                -0.17988800437733973,
                -1.7542757487882648,
                0.0780776345918053,
                -2.915167734547665,
                0.17729089081287383,
                1.9184515755971272,
                0.5894591040300567,
            ],
            "main_gripper": [self.MAX_PANDA_FINGER, self.MAX_PANDA_FINGER],
        }
        return conf

    def arm_from_side(self, side):
        return arm_from_side(side)

    def side_from_arm(self, arm):
        return side_from_arm(arm)

    def arm_conf(self, arm, config):
        return config

    def get_closed_positions(self):
        return {"panda_finger_joint1": 0, "panda_finger_joint2": 0}

    def get_open_positions(self):
        return {
            "panda_finger_joint1": self.MAX_PANDA_FINGER,
            "panda_finger_joint2": self.MAX_PANDA_FINGER,
        }

    @property
    def groups(self):
        return self.joint_groups

    @property
    def default_mobile_base_arm(self):
        return self.get_default_conf()["main_arm"]

    @property
    def default_fixed_base_arm(self):
        return self.get_default_conf()["main_arm"]

    @property
    def base_link(self):
        return link_from_name(self.robot, self.BASE_LINK, client=self.client)

    def update_conf(self):
        # Updates the simulated robot joints to match the real robot joints.
        # If execution in sim, this does nothing
        conf = dict(self.controller.joint_positions)
        for name, position in conf.items():
            joint = joint_from_name(self, name, client=self.client)  # TODO: do in batch
            set_joint_position(self, joint, position, client=self.client)
        return conf

    def reset(self):
        conf = self.get_default_conf()
        for group, positions in conf.items():
            if self.real_execute:
                group_dict = {
                    name: pos for pos, name in zip(positions, self.joint_groups[group])
                }
                self.controller.command_group_dict(group, group_dict)
            else:
                self.set_group_positions(group, positions)

    def get_image(self, seg_network=None):
        if not self.real_camera:
            [camera] = self.cameras
            camera_image = camera.get_image()  # TODO: remove_alpha
            save_camera_images(camera_image, client=self.client)

        else:
            panda_hand_link = link_from_name(self, "panda_hand", client=self.client)
            camera_image = self.controller.capture_image()
            rgb, depth, camera_intrinsics = camera_image

            self.update_conf()

            ############################################
            # hand_to_depth = ((0.04268000721548824, -0.01696075177674074, 0.06000526018408979),
            #                 (-0.06123049355593103, -0.05318248430843586, 0.7020996612061071, 0.7074450620054885)) # from mike calibration 1

            hand_to_depth = (
                (0.046075853709858734, -0.029139794371743268, 0.06463421403706494),
                (
                    -0.05740158346036244,
                    -0.06151583228663349,
                    0.7072010879545587,
                    0.7019882347947254,
                ),
            )

            camera_intrinsics = np.array(
                [
                    [606.92871094, 0.0, 415.18270874],
                    [0.0, 606.51989746, 258.89492798],
                    [0.0, 0.0, 1.0],
                ]
            )  # from rs pipeline

            ###########################################

            panda_hand_pose = get_link_pose(
                self.robot, panda_hand_link, client=self.client
            )
            camera_pose = multiply(panda_hand_pose, hand_to_depth)
            camera_image = CameraImage(
                rgb, depth / 1000.0, None, camera_pose, camera_intrinsics
            )

            save_camera_images(camera_image, client=self.client)

            # with open("./temp/"+str(time.time())+'.pkl', 'wb') as handle:
            #     pickle.dump(camera_image, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return camera_image


def side_from_arm(arm):
    side = arm.split("_")[0]
    return side


def arm_from_side(side):
    return "{}_arm".format(side)


def gripper_from_arm(arm):  # TODO: deprecate
    side = side_from_arm(arm)
    return "{}_gripper".format(side)
