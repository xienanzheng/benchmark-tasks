"""On the station table, in the middle and further back, the work area of a biosafety cabinet: a steel work tray with a
dark grille strip along its front edge and a wider one along its back edge, and a light grey back wall behind them.
Along the wall's front face, high up, runs a UV lamp tube, lit violet. A ready lamp sits on top of the wall at its right
end, grey now. In front of the right arm stand two push buttons on dark bases: the one nearer the middle has a violet
cap and switches the UV lamp, the outer one has a green cap and switches the cabinet's blower. Each press of a cap
switches it over, from on to off or from off to on. The ready lamp is grey while the blower is off, amber while the
blower runs up, and green once it has run up; switching the blower off turns it grey again. In front of the left arm a
white culture rack stands on the bench.

Run it:  inspect-robots run --embodiment yam_arena --scorer task_score ...
         -E task=purge_the_cabinet_before_loading_the_cultures.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from benchkit import Slider, ToolState, on_reset, on_step, tool  # noqa: E402
from plantkit import (  # noqa: E402
    DARK, GREEN, HZ, LAMP_AMBER, LAMP_GREEN, LAMP_OFF, STEEL, Clock, Lamp, at_rest, pushed, tilt_deg,
)
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Longer than the run limit: the UV lamp could be switched back on, or the blower off, after the rack is on the tray,
# so the end state decides, with every step it was in the cabinet checked on the way.
SUCCESS_HOLD_S = 130.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = (
    "Get the safety cabinet ready and load the culture rack: switch the UV lamp off, switch the blower on, wait until "
    "the ready lamp turns green, then put the culture rack on the cabinet's work tray."
)

WALL_GREY = (0.80, 0.81, 0.83)
UV_VIOLET = (0.55, 0.25, 0.95)  # the UV tube while it is on
CAP_VIOLET = (0.42, 0.18, 0.70)
RACK_WHITE = (0.95, 0.95, 0.97)

# -- the cabinet, all fixed (built in): the tray, a grille along its front and back edges, the back wall ---------------
TRAY_XY = (0.31, 0.0)
TRAY = (0.14, 0.20, 0.010)  # steel work tray: x 24 to 38 cm, y -10 to 10 cm
TRAY_TOP = TOP + TRAY[2]
GRILLE = (0.025, 0.20, 0.004)  # the dark front grille, x 21.5 to 24 cm
GRILLE_X = TRAY_XY[0] - TRAY[0] / 2 - GRILLE[0] / 2
# Loading the far half of the tray, the left arm's wrist link (link4) reaches 9 cm past the TCP, 18 to 30 cm above the
# table: a wall at the tray's back edge (x 38 cm) would be in its way. The wall stands 5 cm further back, and a rear
# grille, as a real class II cabinet has, fills the gap.
REAR_GRILLE = (0.050, 0.20, 0.004)  # x 38 to 43 cm
REAR_GRILLE_X = TRAY_XY[0] + TRAY[0] / 2 + REAR_GRILLE[0] / 2
WALL = (0.020, 0.26, 0.22)  # the back wall, x 43 to 45 cm, 22 cm tall
WALL_X = REAR_GRILLE_X + REAR_GRILLE[0] / 2 + WALL[0] / 2
WALL_FACE_X = WALL_X - WALL[0] / 2  # its front face, at the rear grille's back edge
TUBE = (0.015, 0.16, 0.015)  # the UV tube along the wall's front face, its top 1 cm under the wall's top
TUBE_POS = (WALL_FACE_X - TUBE[0] / 2, 0.0, TOP + WALL[2] - 0.010 - TUBE[2] / 2)
# The top camera looks over the arms from behind them. On the bench beside the buttons the right arm, which stands on
# that lane, hides the ready lamp while it presses; on top of the wall neither arm hides it at any step of the demo, and
# the right arm never hides the tube. No side walls: the right one would stand 1.1 cm from the right wrist camera as it
# presses the UV switch. The tests ray-cast both lamps through the arms' meshes.
READY_SIZE = (0.020, 0.050, 0.030)  # the ready lamp, on the wall's top at its right end
READY_POS = (WALL_X, -0.10, TOP + WALL[2] + READY_SIZE[2] / 2)
UV_LAMP = Lamp("uv_lamp", {"on": UV_VIOLET, "off": LAMP_OFF}, "on")
READY_LAMP = Lamp("ready_lamp", {"off": LAMP_OFF, "purging": LAMP_AMBER, "ready": LAMP_GREEN}, "off")

# -- the two switches, in front of the right arm: push-on/push-off buttons like the exemplars' -------------------------
SWITCH_BASE = (0.07, 0.07, 0.030)
SWITCH_CAP = (0.040, 0.040, 0.015)
SWITCH_TRAVEL = 0.008
SWITCH_Z = TOP + SWITCH_BASE[2] / 2
CAP_REST = (0.0, 0.0, SWITCH_BASE[2] / 2 + SWITCH_CAP[2] / 2)  # the cap's centre at rest (tool frame)
CAP_TOP_Z = TOP + SWITCH_BASE[2] + SWITCH_CAP[2]  # the cap's top at rest (world)
UV_SWITCH_XY = (0.20, -0.14)
BLOWER_SWITCH_XY = (0.20, -0.22)


def _switch(name: str, xy, cap_colour):
    return tool(
        name,
        pos=(*xy, SWITCH_Z),
        boxes=[((0.0, 0.0, 0.0), SWITCH_BASE, DARK)],
        sliders=[Slider("cap", rest=CAP_REST, size=SWITCH_CAP, axis="Z", lower=-SWITCH_TRAVEL, upper=0.0,
                        color=cap_colour, rate=300.0, damping=8.0, mass=0.01)],
        mass=3.0,  # heavy enough to stay put when pressed; taped down on a real bench
    )


UV_SWITCH = _switch("uv_switch", UV_SWITCH_XY, CAP_VIOLET)
BLOWER_SWITCH = _switch("blower_switch", BLOWER_SWITCH_XY, GREEN)
SWITCHES = ("uv_switch", "blower_switch")

# -- the culture rack: a tube rack with its tubes, as one block, in front of the left arm ------------------------------
RACK = (0.09, 0.04, 0.05)
RACK_MASS = 0.20
RACK_XY = (0.20, 0.18)
RACK_Z = TOP + RACK[2] / 2

OBJECTS = [
    Box("work_tray", pos=(*TRAY_XY, TOP + TRAY[2] / 2), size=TRAY, color=STEEL),
    Box("front_grille", pos=(GRILLE_X, TRAY_XY[1], TOP + GRILLE[2] / 2), size=GRILLE, color=DARK),
    Box("rear_grille", pos=(REAR_GRILLE_X, TRAY_XY[1], TOP + REAR_GRILLE[2] / 2), size=REAR_GRILLE, color=DARK),
    Box("back_wall", pos=(WALL_X, TRAY_XY[1], TOP + WALL[2] / 2), size=WALL, color=WALL_GREY),
    UV_LAMP.box(TUBE_POS, TUBE),
    READY_LAMP.box(READY_POS, READY_SIZE),
    UV_SWITCH,
    BLOWER_SWITCH,
    Box("culture_rack", pos=(*RACK_XY, RACK_Z), size=RACK, mass=RACK_MASS, color=RACK_WHITE),
]

# -- where the rack is: inside the cabinet, and resting on its tray ----------------------------------------------------
CABINET_FRONT_X = GRILLE_X - GRILLE[0] / 2  # m: the front grille's front edge, x 21.5 cm: the cabinet's opening
CABINET_HALF_Y = TRAY[1] / 2  # m: the cabinet is as wide as the tray
BEHIND_GRILLE = 0.005  # m: on the tray, the rack's front edge is at least this far behind the grille (x 24.5 cm)
ON_TRAY_WITHIN = 0.005  # m: its bottom within this of the tray's top: resting on it, not held up
UPRIGHT_DEG = 20.0
SPILLED_DEG = 45.0  # leaning more than this, its tubes are spilled: inside the cabinet, or on the open bench


def in_cabinet(rack) -> bool:
    """Any part of the rack inside the cabinet, at any height: its footprint past the front grille's front edge and
    over the tray's width, with its centre still in front of the back wall (on top of the wall or behind it, it is
    outside). A rack across the grille, or reaching in from the bench, is in."""
    lo, hi = rack.low, rack.high
    over = lo[1] < TRAY_XY[1] + CABINET_HALF_Y and hi[1] > TRAY_XY[1] - CABINET_HALF_Y
    return bool(hi[0] > CABINET_FRONT_X and rack.pos[0] < WALL_FACE_X and over)


def on_tray(rack) -> bool:
    """Resting on the tray: its whole footprint on the tray and 5 mm or more behind the grille, its bottom within
    5 mm of the tray's top, leaning under 20 degrees."""
    lo, hi = rack.low, rack.high
    x0, x1 = TRAY_XY[0] - TRAY[0] / 2, TRAY_XY[0] + TRAY[0] / 2
    y0, y1 = TRAY_XY[1] - TRAY[1] / 2, TRAY_XY[1] + TRAY[1] / 2
    footprint = lo[0] >= x0 + BEHIND_GRILLE and hi[0] <= x1 and lo[1] >= y0 and hi[1] <= y1
    return bool(footprint and abs(lo[2] - TRAY_TOP) <= ON_TRAY_WITHIN and tilt_deg(rack) < UPRIGHT_DEG)


def off_the_table(rack) -> bool:
    return bool(rack.pos[2] < TOP)


# -- what happened: the two switches, the blower's run-up timer and the three faults, every control step ---------------
PRESS_AT = 0.7  # of the cap's travel: a press (5.6 mm in)
REARM_AT = 0.3  # back out under this before the next press counts (2.4 mm)
PURGE_S = 20.0  # the blower runs this long, without a break, before the ready lamp turns green
PURGE_STEPS = round(PURGE_S * HZ)
CLOCK = Clock()


class Cabinet:
    """The cabinet's switches and timer, and what the rack met inside it. A press of a cap toggles its switch; the UV
    lamp starts on and the blower off. The ready lamp is off while the blower is off, amber (purging) while it has run
    for less than PURGE_S without a break, and green (ready) from then on; switching the blower off resets the timer.
    Four faults, each for the rest of the trial: the rack in the cabinet while the UV lamp is on, the rack in the
    cabinet while the ready lamp is not green, the blower switched off with the rack in the cabinet, and the rack
    tipped over SPILLED_DEG outside the cabinet (its tubes spilled on the open bench)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.uv_on = True
        self.blower_on = False
        self.blower_since = None  # the step the blower last went on
        self.down = dict.fromkeys(SWITCHES, False)  # each cap in, with hysteresis
        self.presses = dict.fromkeys(SWITCHES, 0)
        self.faults = {"uv_on": False, "not_ready": False, "blower_off": False, "spilled": False}
        self.entered_at = None  # the first step the rack was in the cabinet

    def ready(self, step: int) -> bool:
        return bool(self.blower_on and self.blower_since is not None and step - self.blower_since >= PURGE_STEPS)

    def lamp(self, step: int) -> str:
        if not self.blower_on:
            return "off"
        return "ready" if self.ready(step) else "purging"

    def _press(self, name: str, switch: ToolState) -> bool:
        """True on the step the cap passes 70 % of its travel, if it was back under 30 % since its last press."""
        if pushed(switch, "cap", SWITCH_TRAVEL, PRESS_AT):
            edge = not self.down[name]
            self.down[name] = True
        else:
            edge = False
            if at_rest(switch, "cap", SWITCH_TRAVEL, REARM_AT):
                self.down[name] = False
        self.presses[name] += edge
        return edge

    def update(self, states, step: int) -> None:
        rack = states["culture_rack"]
        inside = in_cabinet(rack)
        if not inside and tilt_deg(rack) > SPILLED_DEG:
            self.faults["spilled"] = True
        if self._press("uv_switch", states["uv_switch"]):
            self.uv_on = not self.uv_on
        if self._press("blower_switch", states["blower_switch"]):
            self.blower_on = not self.blower_on
            self.blower_since = step if self.blower_on else None
            if not self.blower_on and inside:
                self.faults["blower_off"] = True
        if inside:
            if self.entered_at is None:
                self.entered_at = step
            if self.uv_on:
                self.faults["uv_on"] = True
            if not self.ready(step):
                self.faults["not_ready"] = True


CABINET = Cabinet()


@on_reset
def _reset(states) -> None:
    CLOCK.reset()
    CABINET.reset()
    UV_LAMP.reset()
    READY_LAMP.reset()


@on_step
def _watch(states) -> None:
    CLOCK.tick()
    CABINET.update(states, CLOCK.step)
    UV_LAMP.set("on" if CABINET.uv_on else "off", CLOCK.step)
    READY_LAMP.set(CABINET.lamp(CLOCK.step), CLOCK.step)


# -- scoring: the end state, unless a fault was latched on the way -----------------------------------------------------
LOADED = 1.0
IN_THE_CABINET = 0.5  # in the cabinet with no fault, but not resting on the tray
READY_OUTSIDE = 0.25  # UV off and the lamp green, the rack still outside
ONE_STEP = 0.1  # the UV off, or the blower on, or both but not yet green; the rack outside
SPILLED = 0.1  # in the cabinet with no fault, but tipped over: its tubes spilled inside the cabinet
SWITCH_UPRIGHT_DEG = 20.0
ON_TABLE_WITHIN = 0.010  # m: a switch's base within this above or below its resting height is on the table


def standing(switch: ToolState) -> bool:
    """A switch upright on the table: leaning 20 degrees or less, its base's centre within 1 cm of its resting height.
    Sliding along the table changes neither."""
    return bool(switch.tilt_deg() <= SWITCH_UPRIGHT_DEG and abs(switch.pos[2] - SWITCH_Z) <= ON_TABLE_WITHIN)


def grade(objects, uv_on: bool, blower_on: bool, ready: bool, faults: dict) -> float:
    """The score as the outcome table states it. ``uv_on``, ``blower_on`` and ``ready`` are the switches and the ready
    lamp now; ``faults`` are the four latched faults."""
    rack = objects["culture_rack"]
    if any(faults.values()) or off_the_table(rack) or not all(standing(objects[n]) for n in SWITCHES):
        return 0.0
    if in_cabinet(rack):
        if uv_on or not ready:
            return 0.0  # the step hook latches this as a fault; a state built by hand is scored the same way
        if tilt_deg(rack) > SPILLED_DEG:
            return SPILLED
        return LOADED if on_tray(rack) else IN_THE_CABINET
    if tilt_deg(rack) > SPILLED_DEG:
        return 0.0  # spilled on the open bench: the step hook latches it; a state built by hand is scored the same way
    if not uv_on and ready:
        return READY_OUTSIDE
    if not uv_on or blower_on:
        return ONE_STEP
    return 0.0


def _grade_now(objects) -> float:
    return grade(objects, CABINET.uv_on, CABINET.blower_on, CABINET.ready(CLOCK.step), CABINET.faults)


def success(objects) -> bool:
    return _grade_now(objects) >= 1.0 and objects["culture_rack"].resting(max_speed=0.02, max_spin=1.0)


def score(objects) -> float:
    return _grade_now(objects)


def distance(objects) -> float:
    """Metres still to go: a cap's travel for each switch not yet set (the UV lamp on, the blower off), plus the rack's
    centre's distance from the tray, 0 once it is over the tray behind the grille."""
    rack = objects["culture_rack"]
    x0, x1 = TRAY_XY[0] - TRAY[0] / 2 + BEHIND_GRILLE, TRAY_XY[0] + TRAY[0] / 2
    dx = max(x0 - rack.pos[0], 0.0, rack.pos[0] - x1)
    dy = max(abs(rack.pos[1] - TRAY_XY[1]) - TRAY[1] / 2, 0.0)
    owed = SWITCH_TRAVEL * (int(CABINET.uv_on) + int(not CABINET.blower_on))
    return float(owed + math.hypot(dx, dy))


# -- the scripted demo: the right arm sets the two switches and parks; the left arm loads the rack once it is green ----
PRESS_Z = CAP_TOP_Z + PAD_TIP_PAST_TCP - SWITCH_TRAVEL - 0.002  # the pad tips 2 mm past the cap's full travel
ABOVE_Z = CAP_TOP_Z + 0.060
TO_UV_S = 1.5
PRESS_S = 1.0  # each cap passes 70 % about 0.8 s into its stroke
HOLD_S = 0.8
RELEASE_S = 0.8
ACROSS_S = 0.8  # from over the UV switch to over the blower switch
PARK_S = 1.5
BLOWER_PRESS_AT = TO_UV_S + PRESS_S + HOLD_S + RELEASE_S + ACROSS_S  # s: the blower stroke starts (4.9 s)
GREEN_BY = BLOWER_PRESS_AT + PRESS_S + PURGE_S  # s: the latest the lamp can turn green (25.9 s)
RIGHT_PARK_XY = (BLOWER_SWITCH_XY[0], BLOWER_SWITCH_XY[1] - 0.15)  # 15 cm to its own side of the blower switch

# The rack is gripped across its 4 cm width, turned a quarter at the carry height before it goes in, and set down
# lengthwise along y: the left wrist camera, which hangs 7 cm ahead of the fingers, then points along the tray at the
# middle of the cabinet. Gripped straight, it would pass within 1 to 3 cm of the UV tube at its height (4 cm is the
# rule) and meet it on the tray's far half. The open fingers, which close along x after the turn, keep clear of the
# wall.
GRIP_ABOVE_BOTTOM = 0.025  # m: the TCP above the rack's bottom while it is held; the pad tips 1.2 cm above it
GRIP_Z = TOP + GRIP_ABOVE_BOTTOM
TRAVEL_Z = TOP + 0.150  # the held rack's bottom rides 11.5 cm over the tray
TURN_YAW = -math.pi / 2
PLACE_XY = (0.30, 0.04)  # the rack's centre on the tray: x 28 to 32 cm and y -0.5 to 8.5 cm once turned
OVER_Z = TRAY_TOP + GRIP_ABOVE_BOTTOM + 0.030  # the rack's bottom 3 cm over the tray
SEAT_Z = TRAY_TOP + GRIP_ABOVE_BOTTOM + 0.004  # let go with the bottom 4 mm over the tray
RELEASE_OPEN = -0.030  # finger target: pads 6 cm apart, 1 cm clear of the rack on each side
PACE = 0.7  # pick_place_waypoints' pace, for the pick and the carry
TO_HOVER_S = 2.0
TURN_S = 1.5
SETTLE_S = 0.4
SEAT_S = 1.2
LEFT_START = 28.0  # s: the left arm goes down to the rack 2.1 s after the latest the ready lamp turns green
LEFT_PARK_XY = (PLACE_XY[0], PLACE_XY[1] + 0.15)  # 15 cm to its own side of the tray


def _press(xy, label: str) -> list[Waypoint]:
    x, y = xy
    return [
        Waypoint(PRESS_S, (x, y, PRESS_Z), FINGER_CLOSED, label),
        Waypoint(HOLD_S, (x, y, PRESS_Z), FINGER_CLOSED, "hold"),
        Waypoint(RELEASE_S, (x, y, ABOVE_Z), FINGER_CLOSED, "release"),
    ]


def _right_plan() -> list[Waypoint]:
    plan = [Waypoint(TO_UV_S, (*UV_SWITCH_XY, ABOVE_Z), FINGER_CLOSED, "over the UV switch")]
    plan += _press(UV_SWITCH_XY, "switch the UV lamp off")
    plan.append(Waypoint(ACROSS_S, (*BLOWER_SWITCH_XY, ABOVE_Z), FINGER_CLOSED, "over the blower switch"))
    plan += _press(BLOWER_SWITCH_XY, "switch the blower on")
    plan.append(Waypoint(PARK_S, (*RIGHT_PARK_XY, ABOVE_Z), FINGER_CLOSED, "park to the side"))
    return plan


def _left_plan(start: float = LEFT_START) -> list[Waypoint]:
    rx, ry = RACK_XY
    px, py = PLACE_XY
    k = 1.0 / PACE
    return [
        Waypoint(TO_HOVER_S, (rx, ry, TRAVEL_Z), FINGER_OPEN, "over the rack"),
        Waypoint(start - TO_HOVER_S, (rx, ry, TRAVEL_Z), FINGER_OPEN, "wait for the green lamp"),
        Waypoint(1.5 * k, (rx, ry, GRIP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.8 * k, (rx, ry, GRIP_Z), FINGER_CLOSED, "close"),
        Waypoint(1.2 * k, (rx, ry, TRAVEL_Z), FINGER_CLOSED, "lift"),
        Waypoint(TURN_S, (rx, ry, TRAVEL_Z), FINGER_CLOSED, "turn a quarter", TURN_YAW),
        Waypoint(1.8 * k, (px, py, TRAVEL_Z), FINGER_CLOSED, "carry into the cabinet", TURN_YAW),
        Waypoint(1.2 * k, (px, py, OVER_Z), FINGER_CLOSED, "lower over the tray", TURN_YAW),
        Waypoint(SETTLE_S, (px, py, OVER_Z), FINGER_CLOSED, "settle", TURN_YAW),
        Waypoint(SEAT_S, (px, py, SEAT_Z), FINGER_CLOSED, "set down", TURN_YAW),
        Waypoint(0.6 * k, (px, py, SEAT_Z), RELEASE_OPEN, "release", TURN_YAW),
        Waypoint(1.0 * k, (px, py, TRAVEL_Z), RELEASE_OPEN, "lift off", TURN_YAW),
        Waypoint(PARK_S, (*LEFT_PARK_XY, TRAVEL_Z), RELEASE_OPEN, "park to the side", TURN_YAW),
    ]


def demo():
    """Right arm: press the violet cap (UV lamp off) and the green cap (blower on), then park 15 cm to its side. Left
    arm: hover over the rack until 2.1 s after the latest the ready lamp can turn green, then pick the rack up, turn
    it a quarter, carry it in and set it down on the tray, and park 15 cm to its side."""
    return {"left": _left_plan(), "right": _right_plan()}
