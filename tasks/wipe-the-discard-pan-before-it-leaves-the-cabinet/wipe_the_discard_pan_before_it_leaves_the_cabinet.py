"""On the station table: the work area of a biosafety cabinet, a steel work tray across the middle of the table with a
dark grille strip along its front edge and a light grey wall standing behind it. On the tray, in a low dark frame on
the midline, sits a covered discard pan: an orange box with a light grey lid. To its left on the tray stands an open
orange biohazard bin. Off the tray, in front of the left arm, a firm light-blue disinfectant pad lies flat in a shallow
grey dish; in front of the right arm is a steel autoclave tray with low walls.

Run it:  inspect-robots run --embodiment yam_arena --scorer task_score ...
         -E task=wipe_the_discard_pan_before_it_leaves_the_cabinet.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from benchkit import ToolState, on_reset, on_step, tool  # noqa: E402
from plantkit import DARK, GREY, STEEL, Clock, in_bin, tilt_deg, tool_points  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = (
    "Take the covered discard pan out to the autoclave tray, but first wipe its whole lid with the disinfectant pad "
    "and drop the pad in the biohazard bin inside the cabinet."
)

PAN_ORANGE = (0.93, 0.48, 0.10)
LID_GREY = (0.80, 0.81, 0.83)
PAD_BLUE = (0.55, 0.76, 0.95)
BIN_ORANGE = (0.90, 0.36, 0.08)
WALL_GREY = (0.86, 0.87, 0.89)

# -- the cabinet: its work tray, the grille strip in front of it and the wall behind it ------------------------------
TRAY_XY = (0.31, 0.04)
TRAY = (0.14, 0.24, 0.01)  # x 24 to 38, y -8 to 16: the spec's 20 cm widened 4 cm to the left, to hold the bin
TRAY_TOP = TOP + TRAY[2]
CABINET_X = TRAY_XY[0] - TRAY[0] / 2  # an object is inside the cabinet with its centre past this line (the grille's
CABINET_Y = (TRAY_XY[1] - TRAY[1] / 2, TRAY_XY[1] + TRAY[1] / 2)  # back edge) and over the tray's width
GRILLE = (0.025, TRAY[1], 0.003)  # x 21.5 to 24
BACK_WALL = (0.01, TRAY[1], 0.20)
BACK_WALL_X = 0.47  # its face at x 46.5: 3 cm or more behind the wrist cameras at the far end of the lid

# -- the pan holder and the discard pan (pan frame: origin at the body's centre, x along its length) ------------------
PAN_XY = (0.31, 0.0)
PAN_BODY = (0.10, 0.055, 0.04)
LID = (0.10, 0.055, 0.005)  # flush with the body, so the fingers close on one flat side
PAN_MASS = 0.3
PAN_Z = TRAY_TOP + PAN_BODY[2] / 2
LID_TOP_LOCAL = PAN_BODY[2] / 2 + LID[2]  # the lid's top face above the pan's origin
LID_TOP = PAN_Z + LID_TOP_LOCAL  # TOP + 0.055 in the holder
HOLDER_GAP = 0.003  # between the pan and the holder's walls, each side
HOLDER_INNER = (PAN_BODY[0] + 2 * HOLDER_GAP, PAN_BODY[1] + 2 * HOLDER_GAP)
HOLDER_WALL = 0.004
HOLDER_H = 0.015

DISCARD_PAN = tool(
    "discard_pan",
    pos=(*PAN_XY, PAN_Z),
    boxes=[
        ((0.0, 0.0, 0.0), PAN_BODY, PAN_ORANGE),
        ((0.0, 0.0, PAN_BODY[2] / 2 + LID[2] / 2), LID, LID_GREY),  # the cover, fixed on: the pan arrives covered
    ],
    mass=PAN_MASS,
)

# -- the pad and its dish, the biohazard bin, the autoclave tray -------------------------------------------------------
PAD = (0.04, 0.03, 0.015)  # gripped across its 3 cm
PAD_MASS = 0.025  # firm: a 5 mm foam face bonded to a rigid 1 cm block (the sim's pad is rigid)
DISH_XY = (0.20, 0.21)
DISH_INNER = (0.06, 0.08)  # 8 cm across the fingers' closing axis: the spec's 6 cm left the opening fingers 0 mm
DISH_WALL = 0.003
DISH_H = 0.015

BIN_XY = (0.33, 0.115)  # on the tray's left part, 4.3 cm clear of the pan's holder
BIN_INNER = 0.07
BIN_WALL = 0.003
BIN_H = 0.06
BIN_RIM = TRAY_TOP + BIN_H

AUTOCLAVE_XY = (0.22, -0.20)
AUTOCLAVE_INNER = (0.13, 0.08)
AUTOCLAVE_WALL = 0.003
AUTOCLAVE_H = 0.02
AUTOCLAVE_FLOOR = 0.002
AUTOCLAVE_FLOOR_TOP = TOP + AUTOCLAVE_FLOOR


def _walls(name: str, xy, inner, height: float, wall: float, color, base_z: float = TOP) -> list[Box]:
    """Four fixed walls around an opening ``inner`` (x, y), standing on ``base_z``."""
    x, y = xy
    ix, iy = inner
    z = base_z + height / 2
    return [
        Box(f"{name}_f", pos=(x + ix / 2 + wall / 2, y, z), size=(wall, iy + 2 * wall, height), color=color),
        Box(f"{name}_b", pos=(x - ix / 2 - wall / 2, y, z), size=(wall, iy + 2 * wall, height), color=color),
        Box(f"{name}_l", pos=(x, y + iy / 2 + wall / 2, z), size=(ix, wall, height), color=color),
        Box(f"{name}_r", pos=(x, y - iy / 2 - wall / 2, z), size=(ix, wall, height), color=color),
    ]


OBJECTS = [
    Box("work_tray", pos=(*TRAY_XY, TOP + TRAY[2] / 2), size=TRAY, color=STEEL),
    Box("grille", pos=(CABINET_X - GRILLE[0] / 2, TRAY_XY[1], TOP + GRILLE[2] / 2), size=GRILLE, color=DARK),
    Box("back_wall", pos=(BACK_WALL_X, TRAY_XY[1], TOP + BACK_WALL[2] / 2), size=BACK_WALL, color=WALL_GREY),
    *_walls("holder", PAN_XY, HOLDER_INNER, HOLDER_H, HOLDER_WALL, DARK, base_z=TRAY_TOP),
    *_walls("bin", BIN_XY, (BIN_INNER, BIN_INNER), BIN_H, BIN_WALL, BIN_ORANGE, base_z=TRAY_TOP),
    *_walls("dish", DISH_XY, DISH_INNER, DISH_H, DISH_WALL, GREY),
    Box("autoclave_floor", pos=(*AUTOCLAVE_XY, TOP + AUTOCLAVE_FLOOR / 2), size=(*AUTOCLAVE_INNER, AUTOCLAVE_FLOOR),
        color=STEEL),
    *_walls("autoclave", AUTOCLAVE_XY, AUTOCLAVE_INNER, AUTOCLAVE_H, AUTOCLAVE_WALL, STEEL),
    DISCARD_PAN,
    Box("pad", pos=(*DISH_XY, TOP + PAD[2] / 2), size=PAD, mass=PAD_MASS, color=PAD_BLUE),
]

# -- the wipe, tracked every step in the pan's own frame -------------------------------------------------------------
# The lid's top face is read as a grid of 2.5 mm cells, 40 along the pan and 22 across it: four quarters of 20 x 11. On
# every step the pad lies flat with its lowest point in the band over the lid, every cell under its footprint (its
# 4 x 3 cm face, turned as it is turned on the lid) is wiped. A quarter is wiped once 90 % of its cells are. A still
# pad, a dab, or a stroke that stops well short of the lid's ends or sides leaves more than that untouched.
WIPE_BAND = 0.006  # m: the pad's lowest point no more than this above the lid's top face ...
WIPE_INTO = 0.003  # ... nor more than this below it (pressed on, as contacts settle in the simulator)
WIPE_TILT_DEG = 10.0  # the pad's face within this of level, either face down: it then rises 5 mm at most across it
CELL = 0.0025  # m: the grid's pitch
QUARTER_DONE = 0.90  # the share of a quarter's cells that wipes it
QUARTERS = ("far left", "far right", "near left", "near right")  # in the pan's frame: far is +x, left is +y
USED_WITHIN = 0.006  # m: the pad is used from the first step its bounding box comes this close to the pan's
PAN_TIP_DEG = 30.0  # the pan leaning more than this, ever: tipped
DROP_FROM = 0.025  # m: the pan landing from more than this above the surface under it: dropped
OFF_SURFACE = 0.002  # m: the pan's lowest point more than this above the surface under its centre: in the air
G = 9.81  # m/s^2

NX, NY = round(LID[0] / CELL), round(LID[1] / CELL)
CELLS = np.array([((i + 0.5) * CELL - LID[0] / 2, (j + 0.5) * CELL - LID[1] / 2) for i in range(NX) for j in range(NY)])
CELL_QUARTER = np.where(CELLS[:, 0] > 0, 0, 2) + np.where(CELLS[:, 1] > 0, 0, 1)  # index into QUARTERS
IN_QUARTER = [CELL_QUARTER == q for q in range(len(QUARTERS))]


def in_cabinet(state) -> bool:
    """The object's centre is past the grille's back edge and over the work tray's width."""
    return bool(state.pos[0] > CABINET_X and CABINET_Y[0] <= state.pos[1] <= CABINET_Y[1])


def rotation_of(state) -> np.ndarray:
    """An object's rotation matrix, from its quaternion (xyzw)."""
    x, y, z, w = np.asarray(state.quat_xyzw, dtype=float) / np.linalg.norm(state.quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def on_lid(pan: ToolState, pad) -> np.ndarray:
    """The pad's centre in the pan's frame, from the centre of the lid's top face (m)."""
    return pan.rotation().T @ (np.asarray(pad.pos, dtype=float) - pan.pos) - np.array([0.0, 0.0, LID_TOP_LOCAL])


def height_over_lid(pan: ToolState, pad) -> float:
    """How far the pad's lowest point is above the plane of the lid's top face, straight below the pad's centre (m).
    Infinite if the pan leans so far that its lid no longer faces up."""
    normal = pan.rotation()[:, 2]
    if normal[2] < 0.5:
        return float("inf")
    face = pan.to_world((0.0, 0.0, LID_TOP_LOCAL))
    x, y = pad.pos[0], pad.pos[1]
    face_z = face[2] - (normal[0] * (x - face[0]) + normal[1] * (y - face[1])) / normal[2]
    return float(pad.low[2] - face_z)


def pad_tilt_deg(pad) -> float:
    """How far the pad's faces lean from level; a pad lying on its other face is level too."""
    tilt = tilt_deg(pad)
    return min(tilt, 180.0 - tilt)


def wiping(pan: ToolState, pad) -> bool:
    """The pad lies flat with its lowest point in the band over the lid's top face."""
    return bool(-WIPE_INTO <= height_over_lid(pan, pad) <= WIPE_BAND and pad_tilt_deg(pad) < WIPE_TILT_DEG)


def under_pad(pan: ToolState, pad) -> np.ndarray:
    """Which of the lid's cells (their centres) lie under the pad's 4 x 3 cm face, turned as it is turned on the lid."""
    turn = pan.rotation().T @ rotation_of(pad)
    u, v = turn[:2, 0], turn[:2, 1]  # the pad's long and short sides, in the lid's plane
    d = CELLS - on_lid(pan, pad)[:2]
    return (np.abs(d @ u) <= np.linalg.norm(u) * PAD[0] / 2) & (np.abs(d @ v) <= np.linalg.norm(v) * PAD[1] / 2)


def box_gap(a, b) -> float:
    """Metres between two objects' bounding boxes, 0 where they overlap."""
    gaps = np.maximum(np.maximum(np.asarray(a.low) - b.high, np.asarray(b.low) - a.high), 0.0)
    return float(np.linalg.norm(gaps))


def off_surface(pan: ToolState) -> float:
    """How far the pan's lowest point is above the surface straight under its centre: the autoclave tray's floor, the
    work tray (the holder's floor), or the table (m)."""
    x, y = pan.pos[0], pan.pos[1]
    if abs(x - AUTOCLAVE_XY[0]) <= AUTOCLAVE_INNER[0] / 2 and abs(y - AUTOCLAVE_XY[1]) <= AUTOCLAVE_INNER[1] / 2:
        floor = AUTOCLAVE_FLOOR_TOP
    elif abs(x - TRAY_XY[0]) <= TRAY[0] / 2 and abs(y - TRAY_XY[1]) <= TRAY[1] / 2:
        floor = TRAY_TOP
    else:
        floor = TOP
    return float(pan.low[2] - floor)


class WipeLedger:
    """What the run has done to the lid and with the pad, and the hazards seen on the way. :meth:`update` takes the
    pan's and the pad's states after each control step."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cells = np.zeros(len(CELLS), dtype=bool)  # the lid's cells wiped so far
        self.wiped_at: list[int | None] = [None] * len(QUARTERS)  # the step each quarter reached 90 %
        self.used_at: int | None = None  # the first step the pad came within USED_WITHIN of the pan
        self.pan_out_at: int | None = None  # the first step the pan's centre was out of the cabinet
        self.binned_first = False  # the pad lay in the bin at that step
        self.pan_out_early_at: int | None = None  # ... with a quarter still unwiped
        self.used_pad_out_at: int | None = None  # the used pad's centre out of the cabinet
        self.tipped_at: int | None = None
        self.dropped_at: int | None = None
        self.fall_from: float | None = None  # in the air: its height over the surface below, plus v^2 / 2g

    def coverage(self, q: int) -> float:
        """The share of quarter ``q``'s cells wiped so far."""
        return float(self.cells[IN_QUARTER[q]].mean())

    @property
    def wiped(self) -> int:
        return sum(at is not None for at in self.wiped_at)

    @property
    def failed(self) -> bool:
        """A hazard happened: it scores 0 whatever follows."""
        return any(at is not None for at in (self.pan_out_early_at, self.used_pad_out_at, self.tipped_at,
                                             self.dropped_at))

    def update(self, pan: ToolState, pad, step: int) -> None:
        wiped_before = self.wiped  # the pan-out check below uses the count before this step's wipe
        if wiping(pan, pad):
            self.cells |= under_pad(pan, pad)
            for q in range(len(QUARTERS)):
                if self.wiped_at[q] is None and self.coverage(q) >= QUARTER_DONE:
                    self.wiped_at[q] = step
        if self.used_at is None and box_gap(pad, pan) <= USED_WITHIN:
            self.used_at = step
        out = not in_cabinet(pan)
        if out and self.pan_out_at is None:
            self.pan_out_at, self.binned_first = step, pad_in_bin(pad)
        if out and self.pan_out_early_at is None and wiped_before < len(QUARTERS):
            self.pan_out_early_at = step
        if self.used_pad_out_at is None and self.used_at is not None and not in_cabinet(pad):
            self.used_pad_out_at = step
        if self.tipped_at is None and pan.tilt_deg() > PAN_TIP_DEG:
            self.tipped_at = step
        # A fall is read from where the pan last was in the air before it lands: its height over the surface plus
        # v^2 / 2g. In free fall that sum stays the height it was let go from, whichever step samples it, so the same
        # drop always counts the same.
        height = off_surface(pan)
        if height > OFF_SURFACE:
            down = max(-float(pan.lin_vel[2]), 0.0)
            self.fall_from = height + down * down / (2 * G)
        elif self.fall_from is not None:
            if self.dropped_at is None and self.fall_from > DROP_FROM:
                self.dropped_at = step
            self.fall_from = None


CLOCK = Clock()
LEDGER = WipeLedger()


@on_reset
def _reset(states) -> None:
    CLOCK.reset()
    LEDGER.reset()


@on_step
def _track(states) -> None:
    CLOCK.tick()
    LEDGER.update(states["discard_pan"], states["pad"], CLOCK.step)


# -- scoring: the end state, plus what the ledger remembers ----------------------------------------------------------
PER_QUARTER = 0.1
ALL_FOUR = PER_QUARTER * len(QUARTERS)  # 0.4: wiped, nothing else done
PAD_BINNED = 0.5  # wiped and the pad in the bin, the pan not yet on the autoclave tray
PAN_OUT = 0.75  # wiped and the pan on the autoclave tray, the pad not in the bin when the pan left the cabinet
ON_TRAY_WITHIN = 0.015  # m: the pan's bottom this close to the autoclave tray's floor
ON_TRAY_UPRIGHT_DEG = 15.0
SET_DOWN_WITHIN = 0.005  # m: for the trial to end, the pan's bottom this close to the tray's floor ...
LET_GO_ABOVE = 0.03  # ... and no gripper over the pan lower than this above its lid: it has been let go
SETTLED = dict(max_speed=0.02, max_spin=1.0)


def pad_in_bin(pad) -> bool:
    return in_bin(pad, BIN_XY, (BIN_INNER, BIN_INNER), TRAY_TOP, BIN_RIM)


def pan_on_tray(pan: ToolState) -> bool:
    """Centre inside the autoclave tray's walls, bottom near its floor, standing within 15 degrees of upright."""
    inside = (abs(pan.pos[0] - AUTOCLAVE_XY[0]) <= AUTOCLAVE_INNER[0] / 2
              and abs(pan.pos[1] - AUTOCLAVE_XY[1]) <= AUTOCLAVE_INNER[1] / 2)
    return bool(inside and pan.low[2] < AUTOCLAVE_FLOOR_TOP + ON_TRAY_WITHIN and pan.tilt_deg() <= ON_TRAY_UPRIGHT_DEG)


def grade(objects, ledger: WipeLedger) -> float:
    """The score as the outcome table states it."""
    pan, pad = objects["discard_pan"], objects["pad"]
    if ledger.failed or pan.tilt_deg() > PAN_TIP_DEG:
        return 0.0
    n = ledger.wiped
    if n < len(QUARTERS):
        return round(PER_QUARTER * n, 6) if in_cabinet(pan) else 0.0  # out with a quarter unwiped: 0
    if not in_cabinet(pad):
        return 0.0  # the pad wiped, so it is used: carried out of the cabinet
    on_tray, binned = pan_on_tray(pan), pad_in_bin(pad)
    if on_tray:
        return 1.0 if binned and ledger.binned_first else PAN_OUT  # the pad binned first, as instructed
    return PAD_BINNED if binned else round(ALL_FOUR, 6)


def let_go(pan: ToolState) -> bool:
    """No gripper's TCP is over the pan lower than 3 cm above its lid. Outside the simulator there are no grippers to
    read, and it holds."""
    lid_z = float(pan.to_world((0.0, 0.0, LID_TOP_LOCAL))[2])
    for tcp in tool_points():
        over = all(abs(tcp[k] - pan.pos[k]) <= PAN_BODY[k] / 2 + 0.03 for k in (0, 1))
        if over and tcp[2] < lid_z + LET_GO_ABOVE:
            return False
    return True


def success(objects) -> bool:
    """The full score, with the pan set down on the tray's floor and let go, and the pan and the pad at rest."""
    pan, pad = objects["discard_pan"], objects["pad"]
    return (grade(objects, LEDGER) >= 1.0 and pan.low[2] < AUTOCLAVE_FLOOR_TOP + SET_DOWN_WITHIN and let_go(pan)
            and pan.resting(**SETTLED) and pad.resting(**SETTLED))


def score(objects) -> float:
    return grade(objects, LEDGER)


def distance(objects) -> float:
    """Metres from the pad to the nearest cell still to wipe in a quarter not yet wiped, then from the pad to the bin,
    then from the pan to the autoclave tray."""
    pan, pad = objects["discard_pan"], objects["pad"]
    left = [q for q, at in enumerate(LEDGER.wiped_at) if at is None]
    if left:
        todo = CELLS[~LEDGER.cells & np.isin(CELL_QUARTER, left)]
        points = pan.pos + np.c_[todo, np.full(len(todo), LID_TOP_LOCAL)] @ pan.rotation().T
        return float(np.linalg.norm(points - np.asarray(pad.pos), axis=1).min())
    if not pad_in_bin(pad):
        return float(np.linalg.norm(np.asarray(pad.pos[:2]) - np.array(BIN_XY)))
    if not pan_on_tray(pan):
        return float(np.linalg.norm(np.asarray(pan.pos[:2]) - np.array(AUTOCLAVE_XY)))
    return 0.0


# -- the scripted demo: left arm wipes the lid in two strokes and bins the pad; right arm then carries the pan out ----
SPEED = 0.7  # the exemplars' pick-and-place pace
PAD_OPEN = -0.025  # finger target: pads 50 mm apart, 1 cm clear of the pad each side, 1 cm inside the dish's walls
TIP_ABOVE_PAD_BOTTOM = 0.005  # the fingers grip the pad's rigid block: from 5 mm above its bottom face to its top
GRIP_ABOVE_PAD_BOTTOM = TIP_ABOVE_PAD_BOTTOM + PAD_TIP_PAST_TCP  # the TCP above the pad's bottom while held
PAD_GRIP_Z = TOP + GRIP_ABOVE_PAD_BOTTOM
OVER_DISH = 0.025  # the descent pauses with the finger tips 1.5 cm over the dish's rim, so the arm's lag dies away
LEFT_TRAVEL_Z = TOP + 0.15
WIPE_GAP = -0.001  # the pad's bottom commanded 1 mm into the lid: a light press, 7 mm under the band's top
WIPE_Z = LID_TOP + WIPE_GAP + GRIP_ABOVE_PAD_BOTTOM  # the finger tips 4 mm over the lid (5 mm once the pad rests on it)
OVER_LID_Z = WIPE_Z + 0.02  # 2 cm up between strokes: at the lid's far end a straight-down grip fails 6 cm up
STROKE_Y = LID[1] / 4  # each stroke along the middle of one half: the 3 cm pad covers the 2.75 cm half and 1.25 mm more
STROKE_X = (PAN_XY[0] - LID[0] / 2 + 0.02, PAN_XY[0] + LID[0] / 2 - 0.02)  # 2 cm inside each end of the lid
STROKE_S = 4.0  # s per 6 cm stroke: 2.25 cm/s at its fastest
APPROACH_S = 1.5  # the last 2 cm onto the lid
SETTLE_S = 0.5
PAD_DROP_Z = BIN_RIM + 0.015 + GRIP_ABOVE_PAD_BOTTOM  # let go with the pad's bottom 1.5 cm over the bin's rim
LEFT_PARK = (0.25, 0.265)  # 15 cm to the left of the bin

PAN_OPEN = -0.036  # finger target: pads 72 mm apart, 8.5 mm clear of the pan each side
PAN_GRIP_Z = TRAY_TOP + HOLDER_H + 0.008 + PAD_TIP_PAST_TCP  # the finger tips 8 mm above the holder's walls
PAN_GRIP_ABOVE_BOTTOM = PAN_GRIP_Z - TRAY_TOP  # 3.6 cm: 2.2 cm of pad on the pan's sides, the lid's edge included
RIGHT_TRAVEL_Z = TOP + 0.15  # the highest a straight-down grip reaches over the pan
OVER_PAN = 0.03  # the descent pauses with the finger tips 8 mm over the lid, so the arm's lag dies away
CLEAR_HOLDER = 0.03  # the first, slow part of the lift: the pan's bottom 3 cm up, twice the holder's walls
LIFT_S = 2.5  # s for those 3 cm: the arm's lag swings the pan's bottom about 1 mm, against 3 mm of clearance
PLACE_GAP = 0.003  # let go with the pan's bottom this far above the autoclave tray's floor
PAN_PLACE_Z = AUTOCLAVE_FLOOR_TOP + PLACE_GAP + PAN_GRIP_ABOVE_BOTTOM
OVER_TRAY = 0.02  # the set-down pauses with the pan this far above where it is let go, over the tray's walls
RIGHT_HOVER = (0.25, -0.24)  # where the right arm waits, to its own side of the cabinet


def _left_plan(strokes=(1, 2)) -> list[Waypoint]:
    """The pad from its dish, two strokes along the lid (one over each half), the pad into the bin, then park.
    ``strokes`` picks which strokes to make, so a test can rebuild a plan that skips one."""
    k = 1.0 / SPEED
    near, far = STROKE_X
    (dx, dy), (bx, by) = DISH_XY, BIN_XY
    py = PAN_XY[1]
    plan = [
        Waypoint(1.5 * k, (dx, dy, LEFT_TRAVEL_Z), PAD_OPEN, "over the pad"),
        Waypoint(1.5, (dx, dy, PAD_GRIP_Z + OVER_DISH), PAD_OPEN, "lower over the pad"),
        Waypoint(0.4, (dx, dy, PAD_GRIP_Z + OVER_DISH), PAD_OPEN, "settle"),
        Waypoint(1.2, (dx, dy, PAD_GRIP_Z), PAD_OPEN, "down to the pad"),
        Waypoint(0.8 * k, (dx, dy, PAD_GRIP_Z), FINGER_CLOSED, "grip the pad"),
        Waypoint(1.2 * k, (dx, dy, LEFT_TRAVEL_Z), FINGER_CLOSED, "lift the pad"),
        Waypoint(2.0 * k, (near, py + STROKE_Y, LEFT_TRAVEL_Z), FINGER_CLOSED, "over the lid"),
    ]
    if 1 in strokes:
        plan += [
            Waypoint(1.2, (near, py + STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "lower over the lid"),
            Waypoint(SETTLE_S, (near, py + STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "settle"),
            Waypoint(APPROACH_S, (near, py + STROKE_Y, WIPE_Z), FINGER_CLOSED, "onto the lid"),
            Waypoint(STROKE_S, (far, py + STROKE_Y, WIPE_Z), FINGER_CLOSED, "first stroke"),
            Waypoint(1.0, (far, py + STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "lift off the lid"),
        ]
    else:
        plan.append(Waypoint(2.0, (far, py + STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "to the far end"))
    if 2 in strokes:
        plan += [
            Waypoint(1.5, (far, py - STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "across"),
            Waypoint(SETTLE_S, (far, py - STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "settle"),
            Waypoint(APPROACH_S, (far, py - STROKE_Y, WIPE_Z), FINGER_CLOSED, "onto the lid"),
            Waypoint(STROKE_S, (near, py - STROKE_Y, WIPE_Z), FINGER_CLOSED, "second stroke"),
            Waypoint(1.0, (near, py - STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "lift off the lid"),
        ]
    else:
        plan.append(Waypoint(2.0, (near, py - STROKE_Y, OVER_LID_Z), FINGER_CLOSED, "back along the lid"))
    plan += [
        Waypoint(1.2, (near, py - STROKE_Y, LEFT_TRAVEL_Z), FINGER_CLOSED, "up"),
        Waypoint(2.0, (bx, by, LEFT_TRAVEL_Z), FINGER_CLOSED, "over the bin"),
        Waypoint(1.2, (bx, by, PAD_DROP_Z), FINGER_CLOSED, "lower over the bin"),
        Waypoint(0.8, (bx, by, PAD_DROP_Z), PAD_OPEN, "let go of the pad"),
        Waypoint(1.0, (bx, by, LEFT_TRAVEL_Z), PAD_OPEN, "up"),
        Waypoint(1.5, (*LEFT_PARK, LEFT_TRAVEL_Z), PAD_OPEN, "park to the side"),
    ]
    return plan


def _right_plan(start_after: float) -> list[Waypoint]:
    """Hover to the side until ``start_after`` s, then lift the pan out of its holder and set it on the autoclave
    tray. The descent into the holder, the lift out of it and the set-down go slowly, so the arm's lag cannot put the
    pan on a wall."""
    k = 1.0 / SPEED
    (px, py), (ax, ay) = PAN_XY, AUTOCLAVE_XY
    return [
        Waypoint(start_after, (*RIGHT_HOVER, RIGHT_TRAVEL_Z), PAN_OPEN, "wait for the pad to be in the bin"),
        Waypoint(1.5 * k, (px, py, RIGHT_TRAVEL_Z), PAN_OPEN, "over the pan"),
        Waypoint(1.5, (px, py, PAN_GRIP_Z + OVER_PAN), PAN_OPEN, "lower over the pan"),
        Waypoint(0.4, (px, py, PAN_GRIP_Z + OVER_PAN), PAN_OPEN, "settle"),
        Waypoint(1.2, (px, py, PAN_GRIP_Z), PAN_OPEN, "down to the pan"),
        Waypoint(0.8 * k, (px, py, PAN_GRIP_Z), FINGER_CLOSED, "grip the pan"),
        Waypoint(LIFT_S, (px, py, PAN_GRIP_Z + CLEAR_HOLDER), FINGER_CLOSED, "lift out of the holder"),
        Waypoint(1.2, (px, py, RIGHT_TRAVEL_Z), FINGER_CLOSED, "up"),
        Waypoint(1.8 * k, (ax, ay, RIGHT_TRAVEL_Z), FINGER_CLOSED, "carry out of the cabinet"),
        Waypoint(1.2, (ax, ay, PAN_PLACE_Z + OVER_TRAY), FINGER_CLOSED, "lower over the tray"),
        Waypoint(0.4, (ax, ay, PAN_PLACE_Z + OVER_TRAY), FINGER_CLOSED, "settle"),
        Waypoint(1.5, (ax, ay, PAN_PLACE_Z), FINGER_CLOSED, "set down"),
        Waypoint(0.8, (ax, ay, PAN_PLACE_Z), PAN_OPEN, "let go of the pan"),
        Waypoint(1.0, (ax, ay, RIGHT_TRAVEL_Z), PAN_OPEN, "up"),
    ]


def demo():
    """Left arm: take the pad from its dish, wipe the lid in two strokes along it (one over each half), drop the pad
    in the bin and park to the side. Right arm: hover to its side until then, lift the pan out of its holder and set
    it down on the autoclave tray."""
    left = _left_plan()
    return {"left": left, "right": _right_plan(sum(w.duration for w in left))}
