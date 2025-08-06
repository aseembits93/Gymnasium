"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class MountainCarEnv(gym.Env):
    """
    ## Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
    This version is the one with discrete actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ## Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min   | Max  | Unit         |
    |-----|--------------------------------------|-------|------|--------------|
    | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
    | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

    ## Action Space

    There are 3 discrete deterministic actions:

    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right

    ## Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.

    ## Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.

    ## Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.

    ## Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.

    ## Arguments

    Mountain Car has two parameters for `gymnasium.make` with `render_mode` and `goal_velocity`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<MountainCarEnv<MountainCar-v0>>>>>
    >>> env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})  # default x_init=np.pi, y_init=1.0
    (array([-0.46352962,  0.        ], dtype=float32), {})

    ```

    ## Version History

    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self._track_res = 100  # Track resolution for rendering

        # Cache track profile and scale so we can reuse across renders
        self._render_geom_initialized = False
        self._car_base_shape = [(-20, 0), (-20, 20), (20, 20), (20, 0)]  # carwidth=40, carheight=20

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        # `xs` is usually an ndarray or float
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        if self.render_mode is None:
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

        # Only initialize and cache static/geometry objects once
        if not self._render_geom_initialized:
            self._initialize_render_geometry()

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Instead of recreating surfaces every time, always draw on a cached one
        if not hasattr(self, "_surf") or self._surf is None:
            self._surf = pygame.Surface((self.screen_width, self.screen_height))
        surf = self._surf
        surf.fill((255, 255, 255))

        pos = self.state[0]

        # Use precomputed XYS for most of the track
        xys = self._cached_xys

        pygame.draw.aalines(surf, points=xys.tolist(), closed=False, color=(0, 0, 0))

        clearance = 10
        carwidth = 40
        carheight = 20

        # For car body polygon, rotate and translate cached base points using math (not pygame.Vector2 objects, which are slow for this)
        theta = math.cos(3 * pos)

        # Compute height at car position only once
        car_h = float(self._height(pos)) * self._scale
        car_x = (pos - self.min_position) * self._scale
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        coords = []
        for x0, y0 in self._car_base_shape:
            # Affine rotation, followed by translation
            x_rot = x0 * cos_theta - y0 * sin_theta
            y_rot = x0 * sin_theta + y0 * cos_theta
            coords.append((
                x_rot + car_x,
                y_rot + clearance + car_h,
            ))

        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))

        # Wheels: precompute values, not per render
        wheel_rad = int(carheight / 2.5)
        for wx in [carwidth / 4, -carwidth / 4]:
            x_rot = wx * cos_theta
            y_rot = wx * sin_theta
            wheel = (
                int(x_rot + car_x),
                int(y_rot + clearance + car_h)
            )
            gfxdraw.aacircle(
                surf, wheel[0], wheel[1], wheel_rad, (128, 128, 128)
            )
            gfxdraw.filled_circle(
                surf, wheel[0], wheel[1], wheel_rad, (128, 128, 128)
            )

        # Flag
        flagx = int((self.goal_position - self.min_position) * self._scale)
        flagy1 = int(float(self._height(self.goal_position)) * self._scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))
        flag_tri = [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        gfxdraw.aapolygon(surf, flag_tri, (204, 204, 0))
        gfxdraw.filled_polygon(surf, flag_tri, (204, 204, 0))

        # Only perform flip/transpose as needed for rgb_array
        if self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            # Avoid double array conversion and transpose, use blit only as needed
            # We flip vertically by reversing lines (much faster than pygame.transform.flip)
            self.screen.blit(surf, (0, 0))
            arr = pygame.surfarray.pixels3d(self.screen)
            # Fastest vertical flip: np.flip(arr, axis=1) operates inplace view
            return np.transpose(np.flip(arr, axis=1), axes=(1, 0, 2)).copy()  # extra copy guards against memory leaks

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _initialize_render_geometry(self):
        # Called once when render assets (track, scales, precomputed quantities) can be reused
        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        self._scale = scale

        # Precompute track geometry/profile
        xs = np.linspace(self.min_position, self.max_position, self._track_res)
        ys = self._height(xs)
        xys = np.empty((self._track_res, 2))
        xys[:, 0] = (xs - self.min_position) * scale
        xys[:, 1] = ys * scale
        self._cached_xs = xs
        self._cached_ys = ys
        self._cached_xys = xys
        self._render_geom_initialized = True
