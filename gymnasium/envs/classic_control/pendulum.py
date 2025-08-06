import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from os import path

__credits__ = ["Carlos Luis"]

from os import path

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumEnv(gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: .

    Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
    the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
    The default value is `g = 10.0`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
    (array([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})

    ```

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        # Avoid repeated attribute access and localize variables
        state = self.state
        th = state[0]
        thdot = state[1]

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # u is an array (always shape [1]) - use fast access
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        th_cost = angle_normalize(th)
        costs = th_cost * th_cost + 0.1 * thdot * thdot + 0.001 * (u * u)

        s_th = np.sin(th)
        newthdot = thdot + (3 * g / (2 * l) * s_th + 3.0 / (m * l * l) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        # Avoid repeated self.render_mode attribute access
        render_mode = self.render_mode
        if render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        # Cache math functions for a tiny gain
        state = self.state
        theta = state[0]
        thetadot = state[1]
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([c, s, thetadot], dtype=np.float32)

    def render(self):
        # Fast exit path
        render_mode = self.render_mode
        if render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        screen_dim = self.screen_dim
        offset = screen_dim // 2

        # Initialization - only once as possible
        if self.screen is None:
            pygame.init()
            if render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_dim, screen_dim))
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((screen_dim, screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create surface and clear color
        surf = pygame.Surface((screen_dim, screen_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = screen_dim / (bound * 2)

        rod_length = scale
        rod_width = 0.2 * scale
        half_rod_width = rod_width / 2

        # Build rectangle coords clockwise, rotated about origin by (theta+pi/2)
        coords_base = np.array([
            [0, -half_rod_width],
            [0,  half_rod_width],
            [rod_length,  half_rod_width],
            [rod_length, -half_rod_width]
        ])
        theta = self.state[0] + PI / 2
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_mat = np.array(
            [[cos_t, -sin_t],
             [sin_t,  cos_t]],
            dtype=np.float32
        )
        # Apply rotation, then translate to screen center
        coords = coords_base @ rot_mat.T
        coords[:, 0] += offset
        coords[:, 1] += offset
        polygon_coords = [tuple(map(int, c)) for c in coords]

        # Draw rod
        gfxdraw.aapolygon(surf, polygon_coords, (204, 77, 77))
        gfxdraw.filled_polygon(surf, polygon_coords, (204, 77, 77))

        # Draw origin axle
        axle_radius = int(0.05 * scale)
        gfxdraw.aacircle(surf, offset, offset, axle_radius, (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, axle_radius, (0, 0, 0))

        # Draw origin hub on the rod
        hub_rad = int(half_rod_width)
        gfxdraw.aacircle(surf, offset, offset, hub_rad, (204, 77, 77))
        gfxdraw.filled_circle(surf, offset, offset, hub_rad, (204, 77, 77))

        # Draw outer tip
        rod_end = np.array([rod_length, 0])
        tip_rot = np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ], dtype=np.float32)
        tip_xy = rod_end @ tip_rot.T
        tip_px = (int(tip_xy[0] + offset), int(tip_xy[1] + offset))
        gfxdraw.aacircle(surf, tip_px[0], tip_px[1], hub_rad, (204, 77, 77))
        gfxdraw.filled_circle(surf, tip_px[0], tip_px[1], hub_rad, (204, 77, 77))

        # Draw action arrow only if self.last_u is set, and non-null
        last_u = getattr(self, "last_u", None)
        if last_u is not None:
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            img = pygame.image.load(fname)
            img_size = int(scale * abs(last_u) / 2)
            if img_size > 0:
                scale_img = pygame.transform.smoothscale(
                    img, (img_size, img_size)
                )
                is_flip = bool(last_u > 0)
                scale_img = pygame.transform.flip(scale_img, is_flip, True)
                surf.blit(
                    scale_img,
                    (
                        offset - scale_img.get_rect().centerx,
                        offset - scale_img.get_rect().centery,
                    ),
                )

        # Flip for correct visual orientation
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            # "rgb_array"
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + PI) % TWO_PI) - PI

PI = np.pi

TWO_PI = 2 * np.pi
