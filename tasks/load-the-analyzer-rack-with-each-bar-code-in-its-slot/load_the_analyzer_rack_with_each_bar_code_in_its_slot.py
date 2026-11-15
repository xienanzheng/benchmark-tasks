"""On the station table, in front of the left arm: a bench rack of three grey collars in a row across the table, each
holding a capped sample tube. The tubes are dark red and square in section, with purple caps, and each carries a white
label with a black stripe down one of its four sides. On the tube nearest the middle of the table the label faces the
robot's left; on the middle tube it faces the arms; on the outermost tube it faces the robot's right.

Near the middle of the table, in front of both arms, stands a blue analyzer rack with three sockets in a row
across the table. Its front wall, on the arms' side, has a vertical slot, open at the top, in front of each socket. In
front of the right arm is a load button: a green cap on a dark base.

Run it:  inspect-robots run --embodiment yam_arena -E task=load_the_analyzer_rack_with_each_bar_code_in_its_slot.py
         --scorer task_score ...
"""

from __future__ import annotations

import itertools
import math
import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from benchkit import Slider, ToolState, on_reset, on_step, tool  # noqa: E402
from plantkit import DARK, GREEN, GREY, Clock, pushed, tilt_deg  # noqa: E402
from yam_arena.constants import FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint, pick_place_waypoints  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = (
    "Load the three sample tubes into the analyzer rack, one per position, each turned so its bar code label faces "
    "out through the rack's front slot, then press the rack's load button."
)

TUBE_RED = (0.45, 0.05, 0.07)  # dark red, as a clear tube of blood looks: the white label stands out
CAP_PURPLE = (0.48, 0.22, 0.62)
LABEL_WHITE = (0.97, 0.97, 0.96)
BAR_BLACK = (0.06, 0.06, 0.07)
RACK_BLUE = (0.18, 0.32, 0.62)

# -- the tubes (tool frame: origin at the body's centre, z up, the label on the body's -x face) ------------------------
BODY = (0.020, 0.020, 0.075)  # m: square in section, so a tube set down turned stays turned
CAP = (0.022, 0.022, 0.012)  # m: on top of the body
# The label covers the face's full width: in a socket 2.8 cm from front to back its edges meet the walls once a tube
# facing the slot is turned 22.6 degrees, inside the 25-degree band that counts, so a seated tube reads plainly as
# facing the slot, a side wall or the back. (1.6 cm wide, it could sit 29 degrees off and read neither way.)
LABEL = (0.002, 0.020, 0.045)  # m: x is its thickness
LABEL_UP = 0.015  # m: its lower edge above the tube's bottom; it reaches 6 cm up
BAR = (0.0005, 0.010, 0.035)  # m: the black stripe standing in for the bar code, centred on the label
LABEL_OUT = LABEL[0] + BAR[0]  # m: how far the label and its stripe stand proud of the body's face
FACE_X = -BODY[0] / 2  # the labelled face, in the tube's frame
LABEL_Z = -BODY[2] / 2 + LABEL_UP + LABEL[2] / 2  # the label's centre, in the tube's frame
LABEL_NORMAL = (-1.0, 0.0, 0.0)  # the label's outward normal, in the tube's frame
LABEL_ARM = BODY[0] / 2 + LABEL_OUT  # m: the stripe's face from the tube's axis
TUBE_MASS = 0.03  # kg
TUBE_H = BODY[2] + CAP[2]  # m: 8.7 cm, bottom to the top of the cap
TUBE_NAMES = ("tube1", "tube2", "tube3")

# -- the bench rack, in front of the left arm: three collars across the table ------------------------------------------
BENCH_X = 0.20
BENCH_Y = {"tube1": 0.12, "tube2": 0.17, "tube3": 0.22}
START_YAW = {"tube1": -math.pi / 2, "tube2": 0.0, "tube3": math.pi / 2}  # labels to +y, to -x (the arms), to -y
BENCH_INNER = 0.030
BENCH_WALL = 0.003
BENCH_H = 0.030


def _collar(name: str, x: float, y: float, inner: float, height: float, wall: float, color) -> list[Box]:
    off = inner / 2 + wall / 2
    z = TOP + height / 2
    long = inner + 2 * wall
    return [
        Box(f"{name}_f", pos=(x + off, y, z), size=(wall, long, height), color=color),
        Box(f"{name}_b", pos=(x - off, y, z), size=(wall, long, height), color=color),
        Box(f"{name}_l", pos=(x, y + off, z), size=(inner, wall, height), color=color),
        Box(f"{name}_r", pos=(x, y - off, z), size=(inner, wall, height), color=color),
    ]


def _tube(name: str):
    return tool(
        name,
        pos=(BENCH_X, BENCH_Y[name], TOP + BODY[2] / 2),
        boxes=[
            ((0.0, 0.0, 0.0), BODY, TUBE_RED),
            ((0.0, 0.0, BODY[2] / 2 + CAP[2] / 2), CAP, CAP_PURPLE),
            ((FACE_X - LABEL[0] / 2, 0.0, LABEL_Z), LABEL, LABEL_WHITE),
            ((FACE_X - LABEL[0] - BAR[0] / 2, 0.0, LABEL_Z), BAR, BAR_BLACK),
        ],
        mass=TUBE_MASS,  # one rigid body: the label turns with the tube
        yaw=START_YAW[name],
        points=[("label", (FACE_X - LABEL_OUT, 0.0, LABEL_Z)), ("bottom", (0.0, 0.0, -BODY[2] / 2))],
    )


TUBES = [_tube(name) for name in TUBE_NAMES]

# -- the analyzer rack, near the middle of the table: three sockets across it, a slotted front wall towards the arms ---
RACK_XY = (0.26, 0.0)  # m: at x 0.30 the left arm, carrying a tube 14 cm up, couldn't reach the right-hand socket
RACK_DEPTH = 0.060  # m, along x
RACK_LENGTH = 0.150  # m, along y
RACK_PLATE = 0.004  # m: the base plate the sockets stand on
WALL_H = 0.040  # m: the walls, above the plate
# m: across the label (x, front to back) and along the row (y). 2.8 cm front to back keys the turn; 3.3 cm along the
# row gives the set-down 6.5 mm a side there, and a tube with its label to a side wall still sits 62 degrees or more
# off the slot.
SOCKET_INNER = (0.028, 0.033)
SOCKET_Y = (-0.045, 0.0, 0.045)  # m: the robot's right to its left
SLOT_W = 0.018  # m: the slot in the front wall before each socket, open at the top, the wall's full height
SIDE_T = (RACK_DEPTH - SOCKET_INNER[0]) / 2  # m: the front and back walls, 1.6 cm
FLOOR_Z = TOP + RACK_PLATE
WALL_TOP = FLOOR_Z + WALL_H
SOCKETS = tuple((RACK_XY[0], RACK_XY[1] + y) for y in SOCKET_Y)
# Where a tube's axis stands with its label to the front and its outline centred in a socket: the label and stripe
# stand 2.5 mm proud of the body's face, so the axis sits 1.25 mm behind the socket's centre.
SEATED_AXIS = tuple((x + LABEL_OUT / 2, y) for x, y in SOCKETS)


def _rack() -> list[Box]:
    x, y0 = RACK_XY
    z = FLOOR_Z + WALL_H / 2
    front_x = x - SOCKET_INNER[0] / 2 - SIDE_T / 2
    back_x = x + SOCKET_INNER[0] / 2 + SIDE_T / 2
    ends = (y0 - RACK_LENGTH / 2, y0 + RACK_LENGTH / 2)
    boxes = [
        Box("rack_base", pos=(x, y0, TOP + RACK_PLATE / 2), size=(RACK_DEPTH, RACK_LENGTH, RACK_PLATE),
            color=RACK_BLUE),
        Box("rack_back", pos=(back_x, y0, z), size=(SIDE_T, RACK_LENGTH, WALL_H), color=RACK_BLUE),
    ]
    # The front wall, cut by a slot before each socket: four posts.
    edges = [ends[0], *(e for _, y in SOCKETS for e in (y - SLOT_W / 2, y + SLOT_W / 2)), ends[1]]
    for i, (a, b) in enumerate(zip(edges[0::2], edges[1::2], strict=True)):
        boxes.append(Box(f"rack_front{i}", pos=(front_x, (a + b) / 2, z), size=(SIDE_T, b - a, WALL_H),
                         color=RACK_BLUE))
    # Across the rack between and beyond the sockets: four walls.
    edges = [ends[0], *(e for _, y in SOCKETS for e in (y - SOCKET_INNER[1] / 2, y + SOCKET_INNER[1] / 2)), ends[1]]
    for i, (a, b) in enumerate(zip(edges[0::2], edges[1::2], strict=True)):
        boxes.append(Box(f"rack_wall{i}", pos=(x, (a + b) / 2, z), size=(SOCKET_INNER[0], b - a, WALL_H),
                         color=RACK_BLUE))
    return boxes


# -- the load button, in front of the right arm ------------------------------------------------------------------------
BUTTON_XY = (0.22, -0.16)
BUTTON_BASE = (0.07, 0.07, 0.030)
BUTTON_TRAVEL = 0.008
BUTTON_CAP = (0.040, 0.040, 0.015)
BUTTON_Z = TOP + BUTTON_BASE[2] / 2
BUTTON = tool(
    "load_button",
    pos=(*BUTTON_XY, BUTTON_Z),
    boxes=[((0.0, 0.0, 0.0), BUTTON_BASE, DARK)],
    sliders=[Slider("cap", rest=(0.0, 0.0, BUTTON_BASE[2] / 2 + BUTTON_CAP[2] / 2), size=BUTTON_CAP, axis="Z",
                    lower=-BUTTON_TRAVEL, upper=0.0, color=GREEN, rate=300.0, damping=8.0, mass=0.01)],
    mass=3.0,  # heavy enough to stay put when pressed; taped down on a real bench
)

OBJECTS = [
    *(w for name in TUBE_NAMES for w in _collar(f"bench{name[-1]}", BENCH_X, BENCH_Y[name], BENCH_INNER, BENCH_H,
                                                  BENCH_WALL, GREY)),
    *_rack(),
    *TUBES,
    BUTTON,
]

# -- reading the tubes: everything from each tube's own pose, nothing from which socket it is in -----------------------
SEATED_WITHIN = 0.005  # m: a tube's bottom this close to the socket floor: down on it, not hanging in the fingers
UPRIGHT_DEG = 10.0  # degrees: it leans no more than this
FACING_DEG = 25.0  # degrees: its label within this of -x, seen from above, faces out through the slot
SLOT_YAW_DEG = 180.0  # -x: out through the front wall, towards the arms


def label_yaw_deg(tube: ToolState) -> float:
    """Which way the label faces seen from above, in degrees anticlockwise from +x (away from the arms), from the tube's
    own rotation."""
    n = tube.rotation() @ np.array(LABEL_NORMAL)
    return float(math.degrees(math.atan2(n[1], n[0])))


def off_slot_deg(tube: ToolState) -> float:
    """Degrees between the label's direction and -x (out through the rack's front slot): 0 to 180."""
    return abs((label_yaw_deg(tube) - SLOT_YAW_DEG + 180.0) % 360.0 - 180.0)


def upright(tube: ToolState) -> bool:
    return tilt_deg(tube) <= UPRIGHT_DEG


def facing_slot(tube: ToolState) -> bool:
    """Upright, with the label within 25 degrees of -x."""
    return upright(tube) and off_slot_deg(tube) <= FACING_DEG


def in_socket(tube: ToolState, socket: int) -> bool:
    """Down in the socket: its centre inside the socket's 2.8 x 3.3 cm opening, seen from above, and its bottom within
    5 mm of the socket's floor, which a tube hanging in the fingers above it is not."""
    x, y = SOCKETS[socket]
    return bool(abs(tube.pos[0] - x) <= SOCKET_INNER[0] / 2 and abs(tube.pos[1] - y) <= SOCKET_INNER[1] / 2
                and tube.low[2] <= FLOOR_Z + SEATED_WITHIN)


def loaded(objects) -> tuple[bool, ...]:
    """For each socket, whether a tube stands down in it, upright, with its label facing out through the slot."""
    tubes = [objects[name] for name in TUBE_NAMES]
    return tuple(any(in_socket(t, s) and facing_slot(t) for t in tubes) for s in range(len(SOCKETS)))


def positions_loaded(objects) -> int:
    return sum(loaded(objects))


def off_bench(tube: ToolState) -> bool:
    """Fallen off the table: its centre below the table top."""
    return bool(tube.pos[2] < TOP)


# -- what happened: the first press of the load button, and what the rack held then ------------------------------------
PRESS_AT = 0.7  # of the cap's travel: 5.6 mm
CLOCK = Clock()
LOAD = {"down": False, "at": None, "held": None}  # the first press: when (s), and how many positions were loaded then
WHEN = dict.fromkeys(TUBE_NAMES)  # when each tube first stood loaded in a socket (s): a record, not scored


@on_reset
def _reset(states) -> None:
    CLOCK.reset()
    LOAD.update(down=False, at=None, held=None)
    for name in WHEN:
        WHEN[name] = None


@on_step
def _watch(states) -> None:
    CLOCK.tick()
    down = pushed(states["load_button"], "cap", BUTTON_TRAVEL, PRESS_AT)
    if down and not LOAD["down"] and LOAD["at"] is None:  # the analyzer takes the rack on the first press
        LOAD["at"], LOAD["held"] = CLOCK.t, positions_loaded(states)
    LOAD["down"] = down
    for name in TUBE_NAMES:
        tube = states[name]
        if WHEN[name] is None and facing_slot(tube) and any(in_socket(tube, s) for s in range(len(SOCKETS))):
            WHEN[name] = CLOCK.t


# -- scoring: the end state, and what the rack held at the first press -------------------------------------------------
PER_POSITION = 0.25
EARLY_PRESS = 0.1  # pressed before all three positions were loaded: the most a trial can score


def grade(objects, held_at_press: int | None) -> float:
    """The score as the outcome table states it. ``held_at_press`` is how many positions were loaded when the load
    button was first pressed, or None if it never was."""
    if any(off_bench(objects[name]) for name in TUBE_NAMES):
        return 0.0
    partial = round(PER_POSITION * positions_loaded(objects), 6)
    if held_at_press is None:
        return partial
    if held_at_press < len(SOCKETS):
        return min(EARLY_PRESS, partial)  # the rack went in short, or with a label turned away from its slot
    return 1.0 if positions_loaded(objects) == len(SOCKETS) else partial


def success(objects) -> bool:
    return grade(objects, LOAD["held"]) >= 1.0 and all(objects[name].resting(max_speed=0.02, max_spin=1.0)
                                                       for name in TUBE_NAMES)


def score(objects) -> float:
    return grade(objects, LOAD["held"])


def distance(objects) -> float:
    """Metres still to go: each tube's distance, seen from above, from where it would stand in the socket it is matched
    with (the matching that leaves least), plus its bottom's height over the socket floor, plus the arc its label still
    has to swing through to face out through the slot."""
    tubes = [objects[name] for name in TUBE_NAMES]
    across = min(sum(float(np.hypot(*(t.pos[:2] - np.array(SEATED_AXIS[s]))))
                     for t, s in zip(tubes, order, strict=True))
                 for order in itertools.permutations(range(len(SOCKETS))))
    return across + sum(max(float(t.low[2]) - FLOOR_Z, 0.0) + LABEL_ARM * math.radians(off_slot_deg(t)) for t in tubes)


# -- the scripted demo: the left arm loads the tubes, turning two of them in the hand; the right arm presses load ------
GRIP_ABOVE_BOTTOM = 0.060  # m: the TCP above a tube's bottom; the pads close on the body, below the cap
PICK_Z = TOP + GRIP_ABOVE_BOTTOM
TRAVEL_Z = TOP + 0.140  # m: a held tube's bottom 3.6 cm over the rack's walls; higher, the right-hand socket is out of
#                         the left arm's reach
OVER_Z = WALL_TOP + 0.012 + GRIP_ABOVE_BOTTOM  # a held tube's bottom 1.2 cm over the rack's walls
PLACE_GAP = 0.007  # m: let go with the tube's bottom this far over the floor: loaded once it drops onto it
PLACE_Z = FLOOR_Z + PLACE_GAP + GRIP_ABOVE_BOTTOM  # the pad tips 1.4 cm over the rack's walls
SPEED = 0.7  # pick_place pace for everything but the set-down
# Fully open, the fingers reach 5 cm or more either side of the tube, onto a tube standing 4.5 or 5 cm away. So they
# open only part way: 3.8 cm apart to come down round a tube (finger joint target, m), 2.6 cm apart to let one go.
OPEN = -0.019
RELEASE = -0.013
# On the way down the arm trails its target, the wrist most (damping / stiffness = 0.15 s), and a quick lower swings a
# held part off its path. With 2.75 mm round a tube across its label (6.5 mm along the row), the set-down stops 1.2 cm
# over the rack, settles and goes in at under 2 cm/s, in place of pick_place's 1.2 s lower: with the joints lagging, a
# tube's bottom then keeps 1.8 mm from the walls across the label and 5.2 mm along the row as it enters, where a 2.2 s
# seat would leave 0.4 mm across.
LOWER_S = 0.6  # s: from the carry height to over the socket
SETTLE_S = 0.6  # s: the lag dies away
SEAT_S = 4.4  # s: into the socket
# (tube, socket, grip yaw at the pick, grip yaw at the place). The turn in the hand is the difference. joint6 runs
# about (the arm's heading - the grip's yaw): the left arm heads 23 to 52 degrees to its right here, so a grip yaw of
# +90 needs joint6 past its 120-degree stop, while -90 and 0 keep it inside 50 degrees. Tube 1's quarter turn
# anticlockwise is therefore gripped across x and set down across y; tube 3's quarter turn clockwise is gripped across
# y and set down across x, on its label and the opposite face, in the right-hand socket: set down across x, the wrist
# camera hangs 7 cm to the robot's right of the fingers, over open table there and over a loaded socket anywhere else.
MOVES = (
    ("tube1", 1, -math.pi / 2, 0.0),  # label from +y to -x: a quarter turn anticlockwise, into the middle socket
    ("tube2", 2, 0.0, 0.0),  # label already towards the arms: no turn, into the left-hand socket
    ("tube3", 0, 0.0, -math.pi / 2),  # label from -y to -x: a quarter turn clockwise, into the right-hand socket
)
# Tube 3 goes last and is carried round the front of the rack: straight across, its bottom would pass 1.1 cm below the
# tops of the tubes already standing in the middle and left-hand sockets.
VIA = {"tube3": (BENCH_X, SOCKET_Y[0])}
ACROSS_S = 1.2  # s: from in front of the rack to over the right-hand socket
LEFT_PARK = (BENCH_X, 0.26)  # the left arm ends on its own side, clear of the top camera's view of the slots
PARK_S = 2.0
# Demo-only aim offsets (m, x and y) for calibration on the GPU; never scored.
TRIM = {"pick": dict.fromkeys(TUBE_NAMES, (0.0, 0.0)), "place": dict.fromkeys(TUBE_NAMES, (0.0, 0.0))}


def grasp_offset(tube_yaw: float, grip_yaw: float) -> np.ndarray:
    """Where the TCP sits from the tube's axis (world x, y in m) once the pads have closed on it: midway between the
    faces they close on. A grip across the label centres LABEL_OUT / 2 towards it; one along the label, on the axis."""
    closing = np.array([-math.sin(grip_yaw), math.cos(grip_yaw)])
    label = np.array([-math.cos(tube_yaw), -math.sin(tube_yaw)])  # LABEL_NORMAL turned by the tube's yaw
    return (LABEL_OUT / 2) * float(closing @ label) * closing


def _move(name: str, socket: int, pick_yaw: float, place_yaw: float) -> list[Waypoint]:
    end_yaw = START_YAW[name] + place_yaw - pick_yaw  # the tube's yaw once it is set down
    pick = np.array([BENCH_X, BENCH_Y[name]]) + grasp_offset(START_YAW[name], pick_yaw) + TRIM["pick"][name]
    place = np.array(SEATED_AXIS[socket]) + grasp_offset(end_yaw, place_yaw) + TRIM["place"][name]
    (px, py), (qx, qy) = (float(v) for v in pick), (float(v) for v in place)
    plan: list[Waypoint] = []
    for w in pick_place_waypoints((px, py, PICK_Z), (qx, qy, PLACE_Z), travel_z=TRAVEL_Z, yaw=pick_yaw,
                                  place_yaw=place_yaw, speed=SPEED):
        if w.label in ("to the object", "descend"):
            plan.append(replace(w, finger=OPEN))
        elif w.label in ("release", "retreat"):
            plan.append(replace(w, finger=RELEASE))
        elif w.label == "carry" and name in VIA:
            plan += [Waypoint(w.duration, (*VIA[name], TRAVEL_Z), FINGER_CLOSED, "carry round the front", place_yaw),
                     Waypoint(ACROSS_S, (qx, qy, TRAVEL_Z), FINGER_CLOSED, "across to the socket", place_yaw)]
        elif w.label == "lower":
            plan += [Waypoint(LOWER_S, (qx, qy, OVER_Z), FINGER_CLOSED, "lower over the socket", place_yaw),
                     Waypoint(SETTLE_S, (qx, qy, OVER_Z), FINGER_CLOSED, "settle", place_yaw),
                     Waypoint(SEAT_S, (qx, qy, PLACE_Z), FINGER_CLOSED, "seat", place_yaw)]
        else:
            plan.append(w)
    return plan


def _left_plan() -> list[Waypoint]:
    plan: list[Waypoint] = []
    for move in MOVES:
        plan += _move(*move)
    plan.append(Waypoint(PARK_S, (*LEFT_PARK, TRAVEL_Z), RELEASE, "park to the side", 0.0))
    return plan


CAP_TOP = BUTTON_Z + BUTTON_BASE[2] / 2 + BUTTON_CAP[2]
PRESS_Z = CAP_TOP + 0.013 - BUTTON_TRAVEL - 0.002  # pad tips 2 mm past the cap's full travel
ABOVE_Z = CAP_TOP + 0.060
RIGHT_PARK = (BUTTON_XY[0], BUTTON_XY[1] - 0.15)  # 15 cm to the right arm's side of the button
TO_SIDE_S = 2.0
TO_BUTTON_S = 1.5
PRESS_S = 1.5
HOLD_S = 0.8
UP_S = 1.5


def _right_plan(start_after: float) -> list[Waypoint]:
    x, y = BUTTON_XY
    return [
        Waypoint(TO_SIDE_S, (*RIGHT_PARK, ABOVE_Z), FINGER_CLOSED, "to the side"),
        Waypoint(start_after - TO_SIDE_S, (*RIGHT_PARK, ABOVE_Z), FINGER_CLOSED, "wait for the tubes"),
        Waypoint(TO_BUTTON_S, (x, y, ABOVE_Z), FINGER_CLOSED, "over the button"),
        Waypoint(PRESS_S, (x, y, PRESS_Z), FINGER_CLOSED, "press load"),
        Waypoint(HOLD_S, (x, y, PRESS_Z), FINGER_CLOSED, "hold"),
        Waypoint(UP_S, (x, y, ABOVE_Z), FINGER_CLOSED, "release"),
        Waypoint(TO_SIDE_S, (*RIGHT_PARK, ABOVE_Z), FINGER_CLOSED, "park"),
    ]


def demo():
    """Left arm: tube 1 into the middle socket turned a quarter anticlockwise on the way, tube 2 into the left-hand
    socket as it is, tube 3 round the front of the rack into the right-hand socket turned a quarter clockwise, each
    label to the front, then aside. Right arm: waits to its side, and presses the load button once the left arm has
    let go of the last tube and lifted clear."""
    left = _left_plan()
    return {"left": left, "right": _right_plan(sum(w.duration for w in left[:-1]))}
