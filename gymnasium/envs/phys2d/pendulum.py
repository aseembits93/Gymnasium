"""Implementation of a Jax-accelerated pendulum environment."""

from __future__ import annotations

from os import path
from typing import Any, Optional, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

import gymnasium as gym
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv, FunctionalJaxVectorEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv
from gymnasium.utils import EzPickle
from gymnasium.vector import AutoresetMode
from functools import lru_cache


PRNGKeyType: TypeAlias = jax.Array
StateType: TypeAlias = jax.Array
RenderStateType = tuple["pygame.Surface", "pygame.time.Clock", Optional[float]]  # type: ignore  # noqa: F821


@struct.dataclass
class PendulumParams:
    """Parameters for the jax Pendulum environment."""

    max_speed: float = 8.0
    dt: float = 0.05
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    high_x: float = jnp.pi
    high_y: float = 1.0
    screen_dim: int = 500


class PendulumFunctional(
    FuncEnv[StateType, jax.Array, int, float, bool, RenderStateType, PendulumParams]
):
    """Pendulum but in jax and functional structure."""

    max_torque: float = 2.0

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-max_torque, max_torque, shape=(1,), dtype=np.float32)

    def initial(
        self, rng: PRNGKeyType, params: PendulumParams = PendulumParams
    ) -> StateType:
        """Initial state generation."""
        high = jnp.array([params.high_x, params.high_y])
        return jax.random.uniform(key=rng, minval=-high, maxval=high, shape=high.shape)

    def transition(
        self,
        state: StateType,
        action: int | jax.Array,
        rng: None = None,
        params: PendulumParams = PendulumParams,
    ) -> StateType:
        """Pendulum transition."""
        th, thdot = state  # th := theta
        u = action

        g = params.g
        m = params.m
        l = params.l
        dt = params.dt

        u = jnp.clip(u, -self.max_torque, self.max_torque)[0]

        newthdot = thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = th + newthdot * dt

        new_state = jnp.array([newth, newthdot])
        return new_state

    def observation(
        self, state: StateType, rng: Any, params: PendulumParams = PendulumParams
    ) -> jax.Array:
        """Generates an observation based on the state."""
        theta, thetadot = state
        return jnp.array([jnp.cos(theta), jnp.sin(theta), thetadot])

    def reward(
        self,
        state: StateType,
        action: ActType,
        next_state: StateType,
        rng: Any,
        params: PendulumParams = PendulumParams,
    ) -> float:
        """Generates the reward based on the state, action and next state."""
        th, thdot = state  # th := theta
        u = action

        u = jnp.clip(u, -self.max_torque, self.max_torque)[0]

        th_normalized = ((th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        costs = th_normalized**2 + 0.1 * thdot**2 + 0.001 * (u**2)

        return -costs

    def terminal(
        self, state: StateType, rng: Any, params: PendulumParams = PendulumParams
    ) -> bool:
        """Determines if the state is a terminal state."""
        return False

    def render_image(
        self,
        state: StateType,
        render_state: RenderStateType,
        params: PendulumParams = PendulumParams,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an RGB image."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        screen, clock, last_u = render_state

        surf_dim = params.screen_dim
        surf = pygame.Surface((surf_dim, surf_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = surf_dim / (bound * 2)
        offset = surf_dim // 2

        rod_length = scale
        rod_width = 0.2 * scale

        angle = float(state[0])
        poly_pts = _calc_rod_coords(angle, rod_length, rod_width, offset)

        color_rod = (204, 77, 77)
        color_pin = (0, 0, 0)

        gfxdraw.aapolygon(surf, poly_pts, color_rod)
        gfxdraw.filled_polygon(surf, poly_pts, color_rod)

        half_rodw = int(rod_width / 2)
        # Pin at center
        gfxdraw.aacircle(surf, offset, offset, half_rodw, color_rod)
        gfxdraw.filled_circle(surf, offset, offset, half_rodw, color_rod)

        # Pin at rod end
        endx, endy = _calc_rod_end(angle, rod_length, offset)
        gfxdraw.aacircle(surf, endx, endy, half_rodw, color_rod)
        gfxdraw.filled_circle(surf, endx, endy, half_rodw, color_rod)

        # If torque applied, show marker
        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        if last_u is not None:
            img = _load_img(fname)
            uscale = max(1, int(scale * np.abs(last_u) / 2))
            scale_img = pygame.transform.smoothscale(img, (uscale, uscale))
            is_flip = bool(last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            blit_rect = scale_img.get_rect(center=(offset, offset))
            surf.blit(scale_img, blit_rect.topleft)

        pinrad = int(0.05 * scale)
        gfxdraw.aacircle(surf, offset, offset, pinrad, color_pin)
        gfxdraw.filled_circle(surf, offset, offset, pinrad, color_pin)

        # Only one vertical flip needed
        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        # Only make a copy of screen array, not screen itself
        render_arr = np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

        return (screen, clock, last_u), render_arr

    def render_init(
        self,
        screen_width: int = 600,
        screen_height: int = 400,
        params: PendulumParams = PendulumParams,
    ) -> RenderStateType:
        """Initialises the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        clock = pygame.time.Clock()

        return screen, clock, None

    def render_close(
        self,
        render_state: RenderStateType,
        params: PendulumParams = PendulumParams,
    ):
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()

    def get_default_params(self, **kwargs) -> PendulumParams:
        """Returns the default parameters for the environment."""
        return PendulumParams(**kwargs)


class PendulumJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based pendulum environment using the functional version as base."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
        "jax": True,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor where the kwargs are passed to the base environment to modify the parameters."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = PendulumFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class PendulumJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
    """Jax-based implementation of the vectorized CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(
        self,
        num_envs: int,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        **kwargs: Any,
    ):
        """Constructor for the vectorized CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(
            self,
            num_envs=num_envs,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        env = PendulumFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )


@lru_cache(maxsize=4)
def _load_img(fname):
    import pygame
    return pygame.image.load(fname)


def _calc_rod_coords(angle, rod_length, rod_width, offset):
    import math
    sin_a = math.sin(angle + np.pi / 2)
    cos_a = math.cos(angle + np.pi / 2)
    x0, x1 = 0, rod_length
    y0 = -rod_width / 2
    y1 = rod_width / 2
    rel = [
        (x0, y0),
        (x0, y1),
        (x1, y1),
        (x1, y0),
    ]
    pts = []
    for xx, yy in rel:
        rx = xx * cos_a - yy * sin_a + offset
        ry = xx * sin_a + yy * cos_a + offset
        pts.append((int(rx), int(ry)))
    return pts


def _calc_rod_end(angle, rod_length, offset):
    import math
    sin_a = math.sin(angle + np.pi / 2)
    cos_a = math.cos(angle + np.pi / 2)
    rx = rod_length * cos_a + offset
    ry = rod_length * sin_a + offset
    return (int(rx), int(ry))
