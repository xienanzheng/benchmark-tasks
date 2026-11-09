"""On the station table, near the middle: a cabinet's steel work tray, with a dark grille strip along its front edge and
a light grey wall along its back. On the tray two steel cups stand in grey collars, side by side, one each side of the
table's midline. Each cup has a white tube top set into its top face and a blue slide lid with a dark grip block on the
lid's end nearest the arms. Both lids stand slid towards the arms, off the cups' tops, on two short rails; a lid slides
away from the arms as far as it goes, to lie over its cup's top. In front of the right arm: a white centrifuge with two
dark collars on its top, its rotor slots, one nearer the middle of the table and one nearer the right arm, and a dark
hinge block along its back.

Run it:  inspect-robots run --embodiment yam_arena --scorer task_score ...
         -E task=close_the_safety_cups_in_the_cabinet.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from benchkit import Slider, ToolState, on_reset, on_step, tool  # noqa: E402
from plantkit import DARK, GREY, WHITE, Clock, distance_xy  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = (
    "Get the two samples into the centrifuge: close the lid on each safety cup while it is still in the cabinet, "
    "then move both cups into the centrifuge's two rotor slots."
)

CUP_STEEL = (0.50, 0.52, 0.55)
TRAY_STEEL = (0.70, 0.72, 0.74)
WALL_GREY = (0.82, 0.83, 0.85)
GRILLE_DARK = (0.18, 0.19, 0.21)
TUBE_WHITE = (0.97, 0.97, 0.95)
LID_BLUE = (0.16, 0.36, 0.72)

# -- the cabinet: a steel work tray, its front grille strip and its back wall, all fixed -----------------------------
TRAY_XY = (0.31, 0.005)
TRAY = (0.14, 0.16, 0.010)
TRAY_TOP = TOP + TRAY[2]
GRILLE = (0.025, 0.16, 0.004)
GRILLE_X = TRAY_XY[0] - TRAY[0] / 2 - GRILLE[0] / 2  # x 21.5 to 24: the strip just in front of the tray
BACK_WALL = (0.02, 0.16, 0.06)  # 6 cm tall: the right wrist camera passes over it at the picks
BACK_WALL_X = TRAY_XY[0] + TRAY[0] / 2 + BACK_WALL[0] / 2  # x 38 to 40, right behind the tray
# Inside the cabinet: a cup's centre over the tray, at least 1 cm behind the grille (x 25 to 38, y -7.5 to 8.5). The
# cabinet is scaled to the arms' reach: the cups stand 6 cm in from the grille, not the 10 cm a real cabinet asks for.
CABINET_X = (TRAY_XY[0] - TRAY[0] / 2 + 0.01, TRAY_XY[0] + TRAY[0] / 2)
CABINET_Y = (TRAY_XY[1] - TRAY[1] / 2, TRAY_XY[1] + TRAY[1] / 2)

# -- the cups' stands: fixed collars on the tray, cup A left of the midline and cup B right of it ---------------------
STAND_XY = ((0.30, 0.04), (0.30, -0.035))
STAND_INNER = 0.048  # 4 mm round a 4 cm cup
COLLAR_WALL = 0.003
STAND_H = 0.030

# -- a cup, in its own frame: origin at the body's centre, x away from the arms; the lid slides along +x ------------
BODY = (0.04, 0.04, 0.07)
HALF_H = BODY[2] / 2
MOUTH = (0.016, 0.016, 0.001)  # the tube's white top, set into the cup's top face, 0.5 mm proud
RAIL = (0.08, 0.004, 0.002)  # two rails on the top face that run out 4 cm towards the arms and carry the open lid
RAIL_X = -0.02  # their centre: from 4 cm in front of the cup's front face to its back face
RAIL_Y = 0.015
LID = (0.04, 0.036, 0.005)  # as long as the top: shut, it covers it end to end
LID_GAP = 0.002  # its underside on the rails, 2 mm over the top face
LID_TRAVEL = 0.04  # +x: at rest the lid lies wholly off the top, towards the arms; pushed 4 cm it covers it
LID_Z = HALF_H + LID_GAP + LID[2] / 2
LID_OPEN_X = -LID_TRAVEL  # the lid's centre at rest
GRIP_BLOCK = (0.012, 0.024, 0.030)  # on the lid's near end, 3 cm tall, so pushing fingers cannot ride over it
GRIP_LOCAL = (-LID[0] / 2 + GRIP_BLOCK[0] / 2, 0.0, LID[2] / 2 + GRIP_BLOCK[2] / 2)  # in the lid's frame
CUP_MASS = 0.146  # kg, the body; the lid and its grip block add 4 g
LID_MASS = 0.004
# The lid's slider has no spring and no friction, only damping, so on a leaning cup gravity slides it along its rails at
# m g sin(lean) / damping. With a 4 g lid and 18 N s/m that is 0.038 mm/s per degree: a shut cup held at 5 degrees takes
# over a minute to open past the 70 % line, and the demo's carries lean the cups 2 degrees at most.
LID_DAMPING = 18.0  # N s/m
CUPS = ("cup_a", "cup_b")
STAND_Z = TRAY_TOP + HALF_H  # a cup's centre, standing in its stand


def _cup(name: str, xy) -> object:
    return tool(
        name,
        pos=(*xy, STAND_Z),
        boxes=[
            ((0.0, 0.0, 0.0), BODY, CUP_STEEL),
            ((0.0, 0.0, HALF_H), MOUTH, TUBE_WHITE),
            ((RAIL_X, RAIL_Y, HALF_H + RAIL[2] / 2), RAIL, DARK),
            ((RAIL_X, -RAIL_Y, HALF_H + RAIL[2] / 2), RAIL, DARK),
        ],
        sliders=[
            # No spring: on a level cup the lid stays where it is pushed, and its joint limit is its stop over the top.
            Slider("lid", rest=(LID_OPEN_X, 0.0, LID_Z), size=LID, axis="X", lower=0.0, upper=LID_TRAVEL,
                   color=LID_BLUE, rate=0.0, damping=LID_DAMPING, mass=LID_MASS,
                   parts=((GRIP_LOCAL, GRIP_BLOCK, DARK),)),
        ],
        mass=CUP_MASS,
    )


CUP_A = _cup("cup_a", STAND_XY[0])
CUP_B = _cup("cup_b", STAND_XY[1])

# -- the centrifuge: a fixed white body with two collars on its top, its rotor slots ----------------------------------
CENTRIFUGE_XY = (0.24, -0.19)
CENTRIFUGE = (0.16, 0.16, 0.05)
SLOT_XY = ((0.24, -0.14), (0.24, -0.24))  # 5 cm either side of the centrifuge's centre: opposite rotor positions
SLOT_INNER = 0.050  # 5 mm round a 4 cm cup
SLOT_H = 0.030
SLOT_FLOOR = TOP + CENTRIFUGE[2]
HINGE = (0.015, 0.12, 0.02)  # the lid's hinge block along the back: decoration
HINGE_X = CENTRIFUGE_XY[0] + CENTRIFUGE[0] / 2 - HINGE[0] / 2


def _collar(name: str, xy, inner: float, height: float, floor_z: float, color) -> list[Box]:
    x, y = xy
    off = inner / 2 + COLLAR_WALL / 2
    z = floor_z + height / 2
    long = inner + 2 * COLLAR_WALL
    return [
        Box(f"{name}_f", pos=(x + off, y, z), size=(COLLAR_WALL, long, height), color=color),
        Box(f"{name}_b", pos=(x - off, y, z), size=(COLLAR_WALL, long, height), color=color),
        Box(f"{name}_l", pos=(x, y + off, z), size=(inner, COLLAR_WALL, height), color=color),
        Box(f"{name}_r", pos=(x, y - off, z), size=(inner, COLLAR_WALL, height), color=color),
    ]


OBJECTS = [
    Box("work_tray", pos=(*TRAY_XY, TOP + TRAY[2] / 2), size=TRAY, color=TRAY_STEEL),
    Box("front_grille", pos=(GRILLE_X, TRAY_XY[1], TOP + GRILLE[2] / 2), size=GRILLE, color=GRILLE_DARK),
    Box("back_wall", pos=(BACK_WALL_X, TRAY_XY[1], TOP + BACK_WALL[2] / 2), size=BACK_WALL, color=WALL_GREY),
    *_collar("stand_a", STAND_XY[0], STAND_INNER, STAND_H, TRAY_TOP, GREY),
    *_collar("stand_b", STAND_XY[1], STAND_INNER, STAND_H, TRAY_TOP, GREY),
    Box("centrifuge", pos=(*CENTRIFUGE_XY, TOP + CENTRIFUGE[2] / 2), size=CENTRIFUGE, color=WHITE),
    Box("hinge", pos=(HINGE_X, CENTRIFUGE_XY[1], SLOT_FLOOR + HINGE[2] / 2), size=HINGE, color=DARK),
    *_collar("slot_near", SLOT_XY[0], SLOT_INNER, SLOT_H, SLOT_FLOOR, DARK),
    *_collar("slot_far", SLOT_XY[1], SLOT_INNER, SLOT_H, SLOT_FLOOR, DARK),
    CUP_A,
    CUP_B,
]

# -- what happened: two things latch for good, a cup out of the cabinet with its lid not shut, and a tipped cup ------
CLOSED_AT = 0.7  # of the lid's travel (2.8 cm): its leading edge past the far side of the tube top, which it covers
SPILL_DEG = 30.0  # a cup leaning more than this at any step has spilled
UPRIGHT_DEG = 15.0  # standing in a rotor slot
IN_SLOT_UP = 0.015  # the cup's bottom within this of the slot's floor

CLOCK = Clock()
LATCHED = {"carried_open": False, "tipped": False}
# Seconds after the reset each cup's lid first counted as shut and each cup first left the cabinet, for the record.
WHEN = {"closed": [None, None], "out": [None, None]}


def lid_closed(cup: ToolState) -> bool:
    return cup.sliders["lid"] >= CLOSED_AT * LID_TRAVEL


def in_cabinet(cup: ToolState) -> bool:
    """The cup's centre over the work tray, 1 cm or more behind the grille, at any height."""
    x, y = cup.pos[:2]
    return bool(CABINET_X[0] < x <= CABINET_X[1] and CABINET_Y[0] <= y <= CABINET_Y[1])


def bottom(cup: ToolState) -> np.ndarray:
    """The centre of the cup's bottom face, in the world."""
    return cup.to_world((0.0, 0.0, -HALF_H))


def in_slot(cup: ToolState, xy) -> bool:
    """Standing in a rotor slot: its bottom inside the collar's opening and within 1.5 cm of the floor, leaning no
    more than 15 degrees (the collar alone stops a lean at about 10)."""
    b = bottom(cup)
    inside = abs(b[0] - xy[0]) <= SLOT_INNER / 2 and abs(b[1] - xy[1]) <= SLOT_INNER / 2
    return bool(inside and b[2] < SLOT_FLOOR + IN_SLOT_UP and cup.tilt_deg() <= UPRIGHT_DEG)


@on_reset
def _reset(states) -> None:
    CLOCK.reset()
    LATCHED["carried_open"] = LATCHED["tipped"] = False
    WHEN["closed"] = [None, None]
    WHEN["out"] = [None, None]


@on_step
def _watch(states) -> None:
    CLOCK.tick()
    for i, name in enumerate(CUPS):
        cup = states[name]
        shut, inside = lid_closed(cup), in_cabinet(cup)
        if cup.tilt_deg() > SPILL_DEG:
            LATCHED["tipped"] = True
        if not inside and not shut:
            LATCHED["carried_open"] = True  # carried out open, or opened again anywhere outside: the rotor included
        if shut and WHEN["closed"][i] is None:
            WHEN["closed"][i] = CLOCK.t
        if not inside and WHEN["out"][i] is None:
            WHEN["out"][i] = CLOCK.t


# -- scoring: the end state, after the two latches ---------------------------------------------------------------------
FIRST_LID = 0.1
SECOND_LID = 0.15
FIRST_IN_ROTOR = 0.25
SECOND_IN_ROTOR = 0.5


def slots_filled(cups) -> int:
    """Rotor slots holding a cup that stands in them with its lid shut, one cup to a slot."""
    return sum(any(lid_closed(c) and in_slot(c, xy) for c in cups) for xy in SLOT_XY)


def grade(objects, carried_open: bool, tipped: bool) -> float:
    """The score as the outcome table states it."""
    cups = [objects[n] for n in CUPS]
    if carried_open or tipped or any(c.tilt_deg() > SPILL_DEG for c in cups):
        return 0.0
    if any(not lid_closed(c) and not in_cabinet(c) for c in cups):
        return 0.0  # an open cup out of the cabinet: the latch, read again on the end state
    shut = sum(lid_closed(c) for c in cups)
    filled = slots_filled(cups)
    score = (FIRST_LID if shut >= 1 else 0.0) + (SECOND_LID if shut == 2 else 0.0)
    score += (FIRST_IN_ROTOR if filled >= 1 else 0.0) + (SECOND_IN_ROTOR if filled == 2 else 0.0)
    return round(score, 6)


def _grade_now(objects) -> float:
    return grade(objects, LATCHED["carried_open"], LATCHED["tipped"])


def success(objects) -> bool:
    return _grade_now(objects) >= 1.0 and all(objects[n].resting(max_speed=0.02, max_spin=1.0) for n in CUPS)


def score(objects) -> float:
    return _grade_now(objects)


def distance(objects) -> float:
    """What's left of each lid's travel to shut, plus metres from the cups to the rotor slots, paired off."""
    a, b = (objects[n] for n in CUPS)
    left = sum(max(CLOSED_AT * LID_TRAVEL - c.sliders["lid"], 0.0) for c in (a, b))
    near, far = SLOT_XY
    apart = min(distance_xy(a, near) + distance_xy(b, far), distance_xy(a, far) + distance_xy(b, near))
    return left + apart


# -- the scripted demo: the left arm pushes both lids shut; the right arm then moves cup B, then cup A, to the rotor --
FINGER_X = 0.017  # the closed fingers' half-width along x where the grip block's top edge meets them, 1.2 cm up the TCP
OVERSHOOT = 0.006  # commanded past the stop, so a lid still shuts if the arm falls a little short
HOVER_Z = TOP + 0.16
LID_TOP_Z = STAND_Z + LID_Z + LID[2] / 2
PUSH_Z = LID_TOP_Z + 0.005 + PAD_TIP_PAST_TCP  # pad tips 5 mm above the lid, against the grip block
GRIP_FACE_OPEN_X = tuple(x + LID_OPEN_X - LID[0] / 2 for x, _ in STAND_XY)  # each grip block's near face, lid open
PUSH_START_X = tuple(f - 0.030 - FINGER_X for f in GRIP_FACE_OPEN_X)
PUSH_END_X = tuple(f + LID_TRAVEL + OVERSHOOT - FINGER_X for f in GRIP_FACE_OPEN_X)
PUSH_GAP = 0.005  # the fingers close up to 5 mm short of the grip block, then push from rest
CLOSE_UP_X = tuple(f - PUSH_GAP - FINGER_X for f in GRIP_FACE_OPEN_X)
PUSH_S = 9.0  # s: slow, so the lid's damping (0.15 N at the peak speed) never pushes hard enough to rock a 150 g cup
LEFT_PARK = (0.19, 0.20, HOVER_Z)  # 15 cm or more to the left arm's own side of both lids

GRIP_OPEN = -0.0275  # finger target: pads 55 mm apart, 7.5 mm each side of a cup
GRIP_ABOVE_BOTTOM = 0.055  # the TCP this far above a held cup's bottom: pads on the body from 4.2 to 6 cm up
GRIP_ABOVE_CENTRE = GRIP_ABOVE_BOTTOM - HALF_H
PICK_Z = TRAY_TOP + GRIP_ABOVE_BOTTOM
CARRY_Z = TOP + 0.170  # a carried cup's bottom 3.5 cm over the rotor collars
LIFT_A_Z = TOP + 0.120  # over cup A's stand the arm reaches no higher; the cup's bottom clears the stands by 2.5 cm
APPROACH_A = (STAND_XY[0][0] - 0.05, STAND_XY[0][1], TOP + 0.130)  # in front of cup A, in line with it
OVER_SLOT_Z = SLOT_FLOOR + SLOT_H + 0.015 + GRIP_ABOVE_BOTTOM  # a held cup's bottom 1.5 cm over the collar
PLACE_Z = SLOT_FLOOR + 0.003 + GRIP_ABOVE_BOTTOM  # let go 3 mm over the slot's floor
CLEAR_STAND_Z = PICK_Z + 0.040  # a lifted cup's bottom 1 cm over its stand's collar
# On the way down the arm trails its target: each joint lags by damping / stiffness (0.0625 s at the shoulder, 0.15 s at
# the wrist), which swings the gripper towards the base, in proportion to the speed; on the way up it swings away. A
# cup has only 4 mm round it in a stand and 5 mm in a rotor slot. So it leaves its stand slowly, and it stops 1.5 cm
# over a slot, settles, and goes in at under 3 cm/s.
DOWN_S = 2.5  # s: from the hover to round a cup, the open pads 7.5 mm off its sides
CLEAR_S = 4.0  # s: out of the stand, 4 cm
LOWER_S = 1.0  # s: from the carry height to over a slot
SETTLE_S = 0.4  # s: the lag dies away
SEAT_S = 2.4  # s: into the slot, 4.2 cm
# Going in, the held cup trails 2 to 2.5 mm towards the arm's base and up to 1.5 mm to its right, so the open-loop
# demo aims each seat off by this much (m), which centres the trail in the slot; it plays no part in scoring.
TRIM = {"pick": {0: (0.0, 0.0), 1: (0.0, 0.0)}, "place": {0: (0.001, 0.0005), 1: (0.001, 0.0)}}
# After letting go of cup B the open fingers' tips stand level with its grip block's top, so the arm rises 3.3 cm more
# and moves off it before it heads down for cup A.
RISE_Z = TOP + 0.190
CLEAR_OF_B = (0.245, -0.17, RISE_Z)


def _push(i: int) -> list[Waypoint]:
    x0, x1, y = PUSH_START_X[i], PUSH_END_X[i], STAND_XY[i][1]
    return [
        Waypoint(0.5, (x0, y, PUSH_Z), FINGER_CLOSED, "settle"),
        Waypoint(1.0, (CLOSE_UP_X[i], y, PUSH_Z), FINGER_CLOSED, "up to the grip block"),
        Waypoint(PUSH_S, (x1, y, PUSH_Z), FINGER_CLOSED, "push the lid shut"),
        Waypoint(0.8, (x1, y, PUSH_Z), FINGER_CLOSED, "hold"),
        Waypoint(1.5, (x0, y, PUSH_Z), FINGER_CLOSED, "back off along the lid's line"),
    ]


def _left_plan(lids=(0, 1)) -> list[Waypoint]:
    first, y0 = lids[0], STAND_XY[lids[0]][1]
    plan = [
        Waypoint(1.5, (PUSH_START_X[first], y0, HOVER_Z), FINGER_CLOSED, "over the first lid"),
        Waypoint(1.2, (PUSH_START_X[first], y0, PUSH_Z), FINGER_CLOSED, "down behind its grip block"),
        *_push(first),
    ]
    for i in lids[1:]:
        across = Waypoint(1.5, (PUSH_START_X[i], STAND_XY[i][1], PUSH_Z), FINGER_CLOSED, "across, in front of the cups")
        plan += [across, *_push(i)]
    last = plan[-1].pos
    plan += [
        Waypoint(1.2, (last[0], last[1], HOVER_Z), FINGER_CLOSED, "lift"),
        Waypoint(1.5, LEFT_PARK, FINGER_CLOSED, "park to its own side"),
    ]
    return plan


def _pick(i: int, lift_z: float) -> list[Waypoint]:
    """From over cup ``i`` (or in line with it): down round it, close, lift it out of its stand slowly, then on up."""
    px, py = TRIM["pick"][i]
    x, y = STAND_XY[i][0] + px, STAND_XY[i][1] + py
    return [
        Waypoint(DOWN_S, (x, y, PICK_Z), GRIP_OPEN, "down round the cup"),
        Waypoint(1.0, (x, y, PICK_Z), FINGER_CLOSED, "close"),
        Waypoint(CLEAR_S, (x, y, CLEAR_STAND_Z), FINGER_CLOSED, "lift clear of the stand"),
        Waypoint(1.2, (x, y, lift_z), FINGER_CLOSED, "lift"),
    ]


def _seat(i: int, slot: int, retreat_z: float = CARRY_Z) -> list[Waypoint]:
    """Carry cup ``i`` from its lift to over a rotor slot, set it in slowly, let go and rise to ``retreat_z``."""
    qx, qy = TRIM["place"][slot]
    x, y = SLOT_XY[slot][0] + qx, SLOT_XY[slot][1] + qy
    return [
        Waypoint(2.5, (x, y, CARRY_Z), FINGER_CLOSED, "carry"),
        Waypoint(LOWER_S, (x, y, OVER_SLOT_Z), FINGER_CLOSED, "lower over the slot"),
        Waypoint(SETTLE_S, (x, y, OVER_SLOT_Z), FINGER_CLOSED, "settle"),
        Waypoint(SEAT_S, (x, y, PLACE_Z), FINGER_CLOSED, "seat"),
        Waypoint(0.8, (x, y, PLACE_Z), GRIP_OPEN, "release"),
        Waypoint(1.2, (x, y, retreat_z), GRIP_OPEN, "retreat"),
    ]


def _right_plan(start_after: float) -> list[Waypoint]:
    bx, by = STAND_XY[1]
    pb = TRIM["pick"][1]
    far = SLOT_XY[1]
    return [
        Waypoint(start_after, (*far, CARRY_Z), GRIP_OPEN, "wait over the centrifuge"),
        # Cup B first, into the far slot: it stands in cup A's way out, and the far slot is then never passed over.
        Waypoint(2.0, (bx + pb[0], by + pb[1], CARRY_Z), GRIP_OPEN, "over cup B"),
        *_pick(1, CARRY_Z),
        *_seat(1, 1, RISE_Z),
        Waypoint(1.2, CLEAR_OF_B, GRIP_OPEN, "clear of cup B"),
        # Cup A, into the near slot. Over its stand the arm cannot hover high, so it comes in level with the cup, from
        # in front, and lifts it only 5.5 cm before carrying it off.
        Waypoint(2.5, APPROACH_A, GRIP_OPEN, "in front of cup A"),
        *_pick(0, LIFT_A_Z),
        *_seat(0, 0),
    ]


def demo():
    """Left arm: push cup A's lid shut by its grip block, then cup B's, back off and park to its own side. Right arm:
    hover over the centrifuge until then, lift cup B out of its stand into the far rotor slot, then cup A into the
    near one."""
    left = _left_plan()
    return {"left": left, "right": _right_plan(sum(w.duration for w in left))}
