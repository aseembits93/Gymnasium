import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


class InvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 20,
    }
    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "inverted_double_pendulum.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        utils.EzPickle.__init__(self, **kwargs)

        # Preallocate observation buffer for reuse to avoid allocations
        self._obs_buf = np.empty(11, dtype=np.float64)

    def step(self, action):
        # Cache attributes for faster access and reduce attribute lookups
        data = self.data

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        site_xpos_0 = data.site_xpos[0]
        x, y = site_xpos_0[0], site_xpos_0[2]

        dist_penalty = 0.01 * x * x + (y - 2.0) * (y - 2.0)
        qvel1, qvel2 = data.qvel[1], data.qvel[2]
        vel_penalty = 1e-3 * qvel1 * qvel1 + 5e-3 * qvel2 * qvel2
        r = 10.0 - dist_penalty - vel_penalty
        terminated = y <= 1.0
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, r, terminated, False, {}

    def _get_obs(self):
        # Vectorized, buffer reuse, no redundant allocations.
        data = self.data
        buf = self._obs_buf
        qpos = data.qpos      # (3,)
        qvel = data.qvel      # (3,)
        qfrc = data.qfrc_constraint  # (3,)

        buf[0] = qpos[0]
        np.sin(qpos[1:], out=buf[1:3])
        np.cos(qpos[1:], out=buf[3:5])
        # Use np.clip with out to avoid allocating temporaries
        np.clip(qvel, -10, 10, out=buf[5:8])
        np.clip(qfrc, -10, 10, out=buf[8:11])

        return buf.copy()  # Return a copy so caller can't mutate internal buffer

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()
