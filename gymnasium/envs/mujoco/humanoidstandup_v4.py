import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}


class HumanoidStandupEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            "humanoidstandup.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        # Pull all required arrays as local variables to reduce repeated attribute lookups
        d = self.data
        # Avoid Python list concatenation; prebuild slices and concatenate only once
        # (using views to avoid copies until the final .concatenate, which is needed for the API)
        qpos = d.qpos.flat
        qvel = d.qvel.flat
        cinert = d.cinert.flat
        cvel = d.cvel.flat
        qfrc_actuator = d.qfrc_actuator.flat
        cfrc_ext = d.cfrc_ext.flat
        # Instead of creating lists each call, pass tuple directly
        return np.concatenate(
            (
                qpos[2:],
                qvel,
                cinert,
                cvel,
                qfrc_actuator,
                cfrc_ext,
            ),
            axis=0,
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.data.qpos[2]
        data = self.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            self._get_obs(),
            reward,
            False,
            False,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()
