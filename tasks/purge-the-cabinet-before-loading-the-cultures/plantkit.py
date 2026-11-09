"""plantkit: lamps, a clock, level strips and placement checks for process-plant task files on yam_arena.

Pure Python (NumPy only). It sits beside ``benchkit`` (tools with sliders, per-step hooks, ``recolor``) and
covers what a plant bench adds: indicator lamps that change colour, a clock counting control steps, a strip of
segments that lights up as a tank fills, and the usual "is it standing in the collar" checks.

Nothing here touches the simulator except through ``benchkit.recolor``, which does nothing outside it, so task
files and their tests load without a GPU.
"""

from __future__ import annotations

import numpy as np
from benchkit import _ENV, _np, recolor
from yam_arena.task_file import Box, ObjectState

HZ = 20  # control steps per second: runs use -E control_hz=20
UP = (0.0, 0.0, 0.0, 1.0)
Color = tuple[float, float, float]

# Colours shared across the plant tasks, so lamps and marks look the same everywhere.
LAMP_OFF = (0.55, 0.56, 0.58)
LAMP_RED = (0.90, 0.06, 0.05)
LAMP_GREEN = (0.10, 0.70, 0.20)
LAMP_AMBER = (0.95, 0.65, 0.05)
WATER_BLUE = (0.25, 0.50, 0.85)
STRIP_DARK = (0.20, 0.21, 0.24)
STEEL = (0.62, 0.64, 0.67)
DARK = (0.25, 0.27, 0.30)
GREY = (0.55, 0.56, 0.58)
YELLOW = (0.95, 0.80, 0.10)
RED = (0.75, 0.08, 0.08)
GREEN = (0.15, 0.60, 0.25)
WHITE = (0.93, 0.93, 0.95)


class Clock:
    """Counts control steps; ``t`` is simulated seconds since the reset."""

    def __init__(self) -> None:
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def tick(self) -> None:
        self.step += 1

    @property
    def t(self) -> float:
        return self.step / HZ


class Lamp:
    """A fixed box whose colour shows a state. ``colours`` maps each state to an sRGB colour; ``start`` is the
    state after a reset. ``set`` recolours the box only when the state changes."""

    def __init__(self, name: str, colours: dict[str, Color], start: str) -> None:
        if start not in colours:
            raise ValueError(f"{name}: start state {start!r} is not one of {sorted(colours)}")
        self.name, self.colours, self.start = name, dict(colours), start
        self.state = start
        self.since = 0  # the step the current state began

    def box(self, pos, size) -> Box:
        return Box(self.name, pos=tuple(pos), size=tuple(size), color=self.colours[self.start])

    def reset(self) -> None:
        self.state = self.start
        self.since = 0
        recolor(self.name, self.colours[self.start])

    def set(self, state: str, step: int = 0) -> None:
        if state not in self.colours:
            raise ValueError(f"{self.name}: no state {state!r}")
        if state != self.state:
            self.state = state
            self.since = step
            recolor(self.name, self.colours[state])

    def is_(self, state: str) -> bool:
        return self.state == state


class Level:
    """A strip of ``n`` segment boxes stacked up a tank's face, lit from the bottom to show how full it is.
    ``show(fraction)`` lights ``round(fraction * n)`` segments."""

    def __init__(self, prefix: str, n: int, empty: Color = STRIP_DARK, full: Color = WATER_BLUE) -> None:
        self.prefix, self.n, self.empty, self.full = prefix, n, empty, full
        self.lit = 0

    def names(self) -> list[str]:
        return [f"{self.prefix}{i}" for i in range(self.n)]

    def boxes(self, xy, z0: float, size, lit: int = 0, gap: float = 0.002) -> list[Box]:
        """``n`` fixed segments of ``size`` (x, y, z) stacked from ``z0`` (the bottom of the lowest), centred on
        ``xy``; the lowest ``lit`` start full."""
        self.lit = lit
        x, y = xy
        return [
            Box(name, pos=(x, y, z0 + i * (size[2] + gap) + size[2] / 2), size=tuple(size),
                color=self.full if i < lit else self.empty)
            for i, name in enumerate(self.names())
        ]

    def show(self, fraction: float) -> None:
        lit = int(round(min(max(fraction, 0.0), 1.0) * self.n))
        if lit != self.lit:
            for i, name in enumerate(self.names()):
                was, now = i < self.lit, i < lit
                if was != now:
                    recolor(name, self.full if now else self.empty)
            self.lit = lit


# -- reading sliders and objects ------------------------------------------------------------------------------------


def slider_fraction(state, name: str, travel: float) -> float:
    """How far a slider is pushed, 0 at its rest stop to 1 at full travel."""
    return float(min(abs(state.sliders[name]) / travel, 1.0)) if travel > 0 else 0.0


def pushed(state, name: str, travel: float, at: float = 0.7) -> bool:
    """Past ``at`` of its travel: a button pressed, a gate shut, a handle thrown."""
    return slider_fraction(state, name, travel) >= at


def at_rest(state, name: str, travel: float, within: float = 0.3) -> bool:
    """Back within ``within`` of its travel from the rest stop."""
    return slider_fraction(state, name, travel) <= within


def tilt_deg(state) -> float:
    """How far an object's vertical axis leans from the vertical."""
    x, y, z, w = state.quat_xyzw
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))))


def over(state, xy, half) -> bool:
    """The object's centre is within ``half`` (x, y) of ``xy`` seen from above."""
    return bool(abs(state.pos[0] - xy[0]) <= half[0] and abs(state.pos[1] - xy[1]) <= half[1])


def in_collar(state, xy, inner: float, floor_z: float, bottom_within: float = 0.015, max_tilt: float = 35.0) -> bool:
    """Standing in a square collar: centre inside its opening, bottom near its floor, leaning no more than
    ``max_tilt`` degrees."""
    return bool(over(state, xy, (inner / 2, inner / 2)) and state.low[2] < floor_z + bottom_within
                and tilt_deg(state) <= max_tilt)


def in_bin(state, xy, inner, floor_z: float, rim_z: float) -> bool:
    """Inside an open box: centre inside the walls and below the rim, however it landed."""
    return bool(over(state, xy, (inner[0] / 2, inner[1] / 2)) and floor_z - 0.005 < state.pos[2] < rim_z)


def distance_xy(state, xy) -> float:
    return float(np.linalg.norm(state.pos[:2] - np.asarray(xy, dtype=float)))


def gap_to(point, state) -> float:
    """Metres from a point to an object's bounding box, 0 inside it."""
    point = np.asarray(point, dtype=float)
    return float(np.linalg.norm(np.maximum(state.low - point, 0.0) + np.maximum(point - state.high, 0.0)))


def tool_points() -> list[np.ndarray]:
    """Each gripper's tool point (between the pads) from the arms' joints, left then right. The object states
    don't include the arms. Empty outside the simulator."""
    env = _ENV["env"]
    if env is None:
        return []
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    out = []
    for arm in ("left", "right"):
        robot = env.unwrapped.scene[f"{arm}_robot"]
        names = list(robot.joint_names)
        q = _np(robot.data.joint_pos)[0][[names.index(j) for j in ARM_JOINTS]]
        out.append(ik.tcp_pose(q, station.BASES[arm])[0])
    return out


# -- states by hand, for tests ---------------------------------------------------------------------------------------


def box_state(pos, size, quat=UP, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> ObjectState:
    """An ``ObjectState`` for an upright box of ``size`` centred at ``pos``: what the simulator reports, built by
    hand. A tilted ``quat`` keeps the box-aligned bounds, which is enough for the checks above."""
    pos, half = np.asarray(pos, dtype=float), np.asarray(size, dtype=float) / 2
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, dtype=float), lin_vel=np.asarray(lin_vel, dtype=float),
                       ang_vel=np.asarray(ang_vel, dtype=float), low=pos - half, high=pos + half)


def cylinder_state(pos, radius: float, height: float, quat=UP, **kw) -> ObjectState:
    return box_state(pos, (2 * radius, 2 * radius, height), quat, **kw)


def leaning(deg: float) -> tuple[float, float, float, float]:
    """A quaternion (xyzw) leaning ``deg`` degrees from the vertical about the x axis, for tests."""
    h = np.radians(deg) / 2
    return (float(np.sin(h)), 0.0, 0.0, float(np.cos(h)))
