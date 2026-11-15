"""The task file without a simulator: python -m pytest -q test_load_the_analyzer_rack_with_each_bar_code_in_its_slot.py

Needs ``yam_arena`` importable (it is in the simulator's environment; elsewhere put a clone of yam-isaac-lab-arena
on PYTHONPATH). Checks the scene loads, the outcome table and its thresholds, that each label is read from its tube's
own pose, that the labels and their stripes show through the slots in the top camera's 320 x 176 image, the geometry
the demo relies on, that every demo waypoint is reachable within the wrist's limits, that the set-down clears the
sockets' walls with the arm lagging, that the wrist cameras clear every fixture, that neither arm hides the labels
when the load button is pressed, and a timed replay of the demo through the step hook.
"""

import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchkit  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from benchkit import tool_state  # noqa: E402
from plantkit import HZ  # noqa: E402
from yam_arena import YAM_URDF, ik, station, task_file  # noqa: E402
from yam_arena.constants import (  # noqa: E402
    ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, READY, SHOULDER, TCP_OFFSET, WRIST,
)
from yam_arena.demo import ArmPlan, Waypoint, grip_rotation, pick_place_waypoints  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402
from yam_arena.urdf import axis_angle_matrix, forward_kinematics, quat_xyzw_to_matrix, rpy_matrix  # noqa: E402

TASK = task_file.load(HERE / "load_the_analyzer_rack_with_each_bar_code_in_its_slot.py")
M = TASK.module
NAMES = M.TUBE_NAMES
DT = 1.0 / HZ
RUN_LIMIT = 2400  # control steps: 2 minutes at 20 Hz
EVAL_IMAGE = (320, 176)  # the top camera's image as Dream-1 runs it
UP = (0.0, 0.0, 0.0, 1.0)
START_DEG = {name: math.degrees(M.START_YAW[name]) for name in NAMES}  # -90, 0 and +90: labels to +y, -x and -y
SOCKET_OF = {name: socket for name, socket, _, _ in M.MOVES}  # where the demo puts each tube
TURN = {name: math.degrees(place - pick) for name, _, pick, place in M.MOVES}  # the turn in the hand: +90, 0, -90


def rotation(yaw_deg=0.0, lean_deg=0.0, lean_about=(1.0, 0.0, 0.0)) -> np.ndarray:
    """A tube turned ``yaw_deg`` about the vertical, after leaning ``lean_deg`` about ``lean_about`` (its own frame)."""
    rz = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg))
    return rz @ axis_angle_matrix(np.array(lean_about, dtype=float), math.radians(lean_deg))


def tube(name, xy, bottom, yaw_deg, lean_deg=0.0, lean_about=(1.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0),
         ang_vel=(0.0, 0.0, 0.0)):
    """A tube's state, its body's centre over ``xy`` and its lowest point at ``bottom``."""
    spec = M.TUBES[NAMES.index(name)]
    quat = ik.matrix_to_quat_xyzw(rotation(yaw_deg, lean_deg, lean_about))
    first = tool_state(spec, (*xy, bottom + M.BODY[2] / 2), quat)
    pos = np.array(first.pos) + np.array([0.0, 0.0, bottom - first.low[2]])
    return tool_state(spec, pos, quat, lin_vel, ang_vel)


def on_bench(name, yaw_deg=None):
    return tube(name, (M.BENCH_X, M.BENCH_Y[name]), M.TOP, START_DEG[name] if yaw_deg is None else yaw_deg)


def seated(name, socket=None, yaw_deg=0.0, dx=0.0, dy=0.0, up=0.0, **kw):
    """Standing in a socket (the demo's for this tube unless given), its outline centred, turned ``yaw_deg``."""
    x, y = M.SEATED_AXIS[SOCKET_OF[name] if socket is None else socket]
    return tube(name, (x + dx, y + dy), M.FLOOR_Z + up, yaw_deg, **kw)


def button(depth=0.0):
    return tool_state(M.BUTTON, M.BUTTON.pos, UP, sliders={"cap": -depth})


def scene(**where):
    """Each tube given a state where ``where`` says, the rest standing in the bench rack as they start; the button
    up."""
    objects = {name: where.get(name, on_bench(name)) for name in NAMES}
    objects["load_button"] = button()
    return objects


def placed(name, how):
    """A tube left ``how``: 'right' (in its socket, label to the slot), 'unturned' (in it as it came off the bench),
    'wrong' (in it, turned the other way), 'bench' (where it started) or 'floor' (fallen off the table)."""
    if how == "right":
        return seated(name)
    if how == "unturned":
        return seated(name, yaw_deg=START_DEG[name])
    if how == "wrong":
        return seated(name, yaw_deg=START_DEG[name] - TURN[name] if TURN[name] else 180.0)
    if how == "bench":
        return on_bench(name)
    if how == "floor":
        return tube(name, (0.30, 0.45), 0.02, 0.0)
    raise ValueError(how)


def table(**hows):
    return scene(**{name: placed(name, how) for name, how in hows.items()})


ALL_RIGHT = {"tube1": "right", "tube2": "right", "tube3": "right"}


def test_the_task_loads_with_its_objects_hooks_and_docs():
    assert TASK.scene == "station" and TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert TASK.free_objects == ("tube1", "tube2", "tube3", "load_button")
    assert all(getattr(TASK, k) is not None for k in ("success", "score", "distance", "demo"))
    assert M._watch in benchkit._STEP.values() and M._reset in benchkit._RESET.values()
    for spec in M.TUBES:
        assert spec.sliders == () and spec.mass == pytest.approx(0.03)  # one rigid body: the label turns with it
    assert M.BUTTON.mass == 3.0 and M.BUTTON.sliders[0].rate == 300.0 and M.BUTTON.sliders[0].lower == -0.008
    docs = " ".join(TASK.docs.lower().split())
    for words in ("bench rack", "sample tube", "dark red", "square in section", "purple cap", "white label",
                  "black stripe", "analyzer rack", "three sockets in a row", "vertical slot", "load button",
                  "green cap"):
        assert words in docs, words


def test_what_the_agent_reads_describes_the_scene_and_does_not_steer_it():
    docs = " ".join(TASK.docs.lower().split("run it:")[0].split())  # the run line below it only names the file
    for words in ("faces the robot's left", "faces the arms", "faces the robot's right"):
        assert words in docs, words  # where each label starts is part of the scene
    for word in ("must", "don't", "never", "should", "turn", "rotat", "orient", "reader", "no-read", "reject",
                 "mix-up", "delay", "wrong", "correct", "out through", "press"):
        assert word not in docs, word


# -- the outcome table ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hows", "held_at_press", "expected"),
    [
        (ALL_RIGHT, 3, 1.0),  # the safe finish: all three in with their labels in the slots, then load
        (ALL_RIGHT, None, 0.75),  # all three right, load not pressed
        ({"tube1": "right", "tube2": "right"}, None, 0.5),  # the third still on the bench
        ({"tube1": "right", "tube2": "right", "tube3": "unturned"}, None, 0.5),  # the third set in without its turn
        ({"tube1": "right", "tube2": "right", "tube3": "wrong"}, None, 0.5),  # the third turned the wrong way
        ({"tube1": "unturned", "tube2": "right", "tube3": "right"}, None, 0.5),
        ({"tube1": "wrong", "tube2": "right", "tube3": "right"}, None, 0.5),
        ({"tube1": "right", "tube2": "wrong", "tube3": "right"}, None, 0.5),  # tube 2 turned round, label to the back
        ({"tube2": "right"}, None, 0.25),  # one right
        ({"tube1": "unturned", "tube3": "unturned"}, None, 0.0),  # two set in as they came: neither label in its slot
        ({}, None, 0.0),  # did nothing
        ({}, 0, 0.0),  # pressed load and did nothing else
        (ALL_RIGHT, 2, 0.1),  # pressed with two in, then finished: the rack went in short
        (ALL_RIGHT, 0, 0.1),  # pressed first, then loaded all three
        ({"tube1": "right", "tube2": "right"}, 2, 0.1),  # pressed with two in, and stopped
        ({"tube1": "right", "tube2": "right", "tube3": "unturned"}, 2, 0.1),  # pressed with a label turned away
        ({"tube1": "unturned", "tube2": "wrong", "tube3": "wrong"}, 0, 0.0),  # pressed, nothing ever right
        ({"tube1": "right", "tube2": "right"}, 3, 0.5),  # pressed with all three in, then one taken out again
        ({"tube1": "right", "tube2": "right", "tube3": "floor"}, None, 0.0),  # a tube dropped off the bench
        ({"tube1": "right", "tube2": "right", "tube3": "floor"}, 2, 0.0),
        ({"tube1": "floor"}, None, 0.0),
    ],
)
def test_outcome_table(hows, held_at_press, expected):
    assert M.grade(table(**hows), held_at_press) == pytest.approx(expected)


def test_pressing_early_scores_below_every_honest_partial():
    early = M.grade(table(**ALL_RIGHT), 2)
    honest = [M.grade(table(**hows), None) for hows in ({"tube2": "right"}, {"tube1": "right", "tube2": "right"},
                                                         ALL_RIGHT)]
    assert 0.0 < early < min(honest) and max(honest) < M.grade(table(**ALL_RIGHT), 3) == 1.0
    assert M.grade(table(), 0) == 0.0 == M.grade(table(), None)  # the button alone earns nothing


def test_the_thresholds_are_the_stated_numbers():
    full = {name: seated(name) for name in NAMES}

    def grade(**changes):
        return M.grade(scene(**{**full, **changes}), None)

    for side in (-1, 1):
        assert grade(tube2=seated("tube2", yaw_deg=side * 24.9)) == pytest.approx(0.75)  # label within 25 degrees
        assert grade(tube2=seated("tube2", yaw_deg=side * 25.1)) == pytest.approx(0.5)
        for about in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            assert grade(tube2=seated("tube2", lean_deg=side * 9.9, lean_about=about)) == pytest.approx(0.75)
            assert grade(tube2=seated("tube2", lean_deg=side * 10.1, lean_about=about)) == pytest.approx(0.5)
    x, y = M.SOCKETS[SOCKET_OF["tube2"]]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # the centre inside the 2.8 x 3.3 cm opening
        inside = (dx * 0.0139, dy * 0.0164)
        outside = (dx * 0.0141, dy * 0.0166)
        assert grade(tube2=tube("tube2", (x + inside[0], y + inside[1]), M.FLOOR_Z, 0.0)) == pytest.approx(0.75)
        assert grade(tube2=tube("tube2", (x + outside[0], y + outside[1]), M.FLOOR_Z, 0.0)) == pytest.approx(0.5)
    assert grade(tube2=seated("tube2", up=0.0049)) == pytest.approx(0.75)  # the bottom within 5 mm of the floor
    assert grade(tube2=seated("tube2", up=0.0051)) == pytest.approx(0.5)
    objects = scene(**full)
    objects["load_button"] = button(0.0057)  # pressed: 70 % of the 8 mm travel is 5.6 mm
    M._reset({})
    M._watch(objects)
    assert M.LOAD["held"] == 3
    M._reset({})
    objects["load_button"] = button(0.0055)
    M._watch(objects)
    assert M.LOAD["at"] is None
    M._reset({})


def test_each_label_is_read_from_its_tubes_own_pose_not_from_its_socket():
    assert M.label_yaw_deg(on_bench("tube1")) == pytest.approx(90.0)  # the robot's left
    assert abs(M.label_yaw_deg(on_bench("tube2"))) == pytest.approx(180.0)  # the arms
    assert M.label_yaw_deg(on_bench("tube3")) == pytest.approx(-90.0)  # the robot's right
    for socket in range(len(M.SOCKETS)):
        for name in NAMES:
            assert M.loaded(scene(**{name: seated(name, socket)}))[socket]
            for yaw in (90.0, -90.0, 180.0, 30.0, -30.0):
                assert not any(M.loaded(scene(**{name: seated(name, socket, yaw)}))), (name, socket, yaw)
            assert M.loaded(scene(**{name: seated(name, socket, 360.0)}))[socket]
    facing = seated("tube1")
    assert facing.point("label")[0] < facing.pos[0] - M.BODY[0] / 2  # the labelled face is the one towards the arms
    assert M.off_slot_deg(facing) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ["tube1", "tube3"])
def test_a_tube_set_in_unturned_or_turned_the_wrong_way_does_not_count(name):
    right = {n: seated(n) for n in NAMES}
    assert M.grade(scene(**right), None) == pytest.approx(0.75)
    unturned = seated(name, yaw_deg=START_DEG[name])
    wrong = seated(name, yaw_deg=START_DEG[name] - TURN[name])
    assert M.off_slot_deg(unturned) == pytest.approx(90.0) and M.off_slot_deg(wrong) == pytest.approx(180.0)
    for bad in (unturned, wrong):
        assert M.in_socket(bad, SOCKET_OF[name]) and not M.facing_slot(bad)
        assert M.grade(scene(**{**right, name: bad}), None) == pytest.approx(0.5)
        assert M.grade(scene(**{**right, name: bad}), 2) == pytest.approx(0.1)


def test_tube_2_counts_only_as_it_stands_and_not_turned_round():
    assert M.facing_slot(seated("tube2", yaw_deg=START_DEG["tube2"]))
    for yaw in (90.0, -90.0, 180.0):
        assert not M.facing_slot(seated("tube2", yaw_deg=yaw))


def test_tubes_lying_upside_down_held_up_or_crowded_in_count_for_nothing():
    full = {name: seated(name) for name in NAMES}
    for bad in (seated("tube2", lean_deg=90.0), seated("tube2", lean_deg=180.0), seated("tube2", up=0.03),
                tube("tube2", M.SEATED_AXIS[2], M.WALL_TOP, 0.0),  # standing on the rack's top
                tube("tube2", (0.30, 0.10), M.TOP, 0.0)):  # standing on the bench, label towards the arms
        assert M.grade(scene(**{**full, "tube2": bad}), None) == pytest.approx(0.5)
    crowded = scene(**{**full, "tube2": seated("tube2", socket=SOCKET_OF["tube1"])})  # two in one socket
    assert M.positions_loaded(crowded) == 2 and M.grade(crowded, None) == pytest.approx(0.5)


def test_success_needs_the_full_score_with_the_tubes_at_rest():
    M._reset({})
    assert not M.success(table(**ALL_RIGHT))  # load not pressed yet
    M.LOAD["held"] = 3
    assert M.success(table(**ALL_RIGHT))
    moving = table(**ALL_RIGHT)
    moving["tube3"] = seated("tube3", lin_vel=(0.0, 0.0, -0.05))
    assert not M.success(moving)
    spinning = table(**ALL_RIGHT)
    spinning["tube1"] = seated("tube1", ang_vel=(0.0, 0.0, 1.5))
    assert not M.success(spinning)
    M.LOAD["held"] = 2
    assert not M.success(table(**ALL_RIGHT))
    M._reset({})


def test_the_first_press_is_recorded_with_what_the_rack_held_and_a_reset_clears_it():
    M._reset({})
    two = table(tube1="right", tube2="right")
    M._watch(two)
    assert M.LOAD == {"down": False, "at": None, "held": None}
    pressed = {**two, "load_button": button(M.BUTTON_TRAVEL)}
    M._watch(pressed)
    M._watch(pressed)  # holding the cap down is one press
    assert M.LOAD["held"] == 2 and M.LOAD["at"] == pytest.approx(2 / HZ)
    M._watch(table(**ALL_RIGHT))
    M._watch({**table(**ALL_RIGHT), "load_button": button(M.BUTTON_TRAVEL)})  # pressed again once all three are in
    assert M.LOAD["held"] == 2 and M.LOAD["at"] == pytest.approx(2 / HZ)  # the analyzer took the rack the first time
    assert M.score(table(**ALL_RIGHT)) == pytest.approx(0.1)
    assert M.WHEN["tube1"] == pytest.approx(1 / HZ) and M.WHEN["tube3"] == pytest.approx(4 / HZ)
    M._reset({})
    assert M.LOAD == {"down": False, "at": None, "held": None} and M.CLOCK.step == 0
    assert all(v is None for v in M.WHEN.values())


def test_distance_falls_to_the_end():
    start = M.distance(table())
    one = M.distance(table(tube2="right"))
    unturned = M.distance(table(tube1="unturned", tube2="right", tube3="unturned"))
    done = M.distance(table(**ALL_RIGHT))
    assert start > one > unturned > done == pytest.approx(0.0, abs=1e-9)


# -- geometry the scoring and the demo rely on -----------------------------------------------------------------------


def outline(yaw_deg, with_label=True) -> np.ndarray:
    """The corners of a tube's footprint seen from above (m, from its axis): its body, and its label and stripe."""
    pieces = [((0.0, 0.0), M.BODY[:2])]
    if with_label:
        pieces += [((M.FACE_X - M.LABEL[0] / 2, 0.0), M.LABEL[:2]),
                   ((M.FACE_X - M.LABEL[0] - M.BAR[0] / 2, 0.0), M.BAR[:2])]
    r = rotation(yaw_deg)[:2, :2]
    return np.array([r @ (np.array(c) + np.array([sx, sy]) * np.array(s) / 2)
                     for c, s in pieces for sx in (-1, 1) for sy in (-1, 1)])


def fits(yaw_deg, with_label=True) -> bool:
    """Whether a tube turned ``yaw_deg`` goes down into a socket at all, wherever it stands in it."""
    pts = outline(yaw_deg, with_label)
    return bool(np.all(pts.max(axis=0) - pts.min(axis=0) <= np.array(M.SOCKET_INNER)))


def most_turn_deg(with_label=True, start=0.0) -> float:
    """The most a tube can be turned from ``start`` (degrees of yaw) and still go down into a socket."""
    angles = np.arange(0.0, 45.0, 0.01)
    ok = [fits(start + a, with_label) for a in angles]
    return float(angles[ok.index(False) - 1]) if False in ok else 45.0


def test_a_seated_tube_sits_within_23_degrees_of_square_because_its_label_meets_the_walls():
    turn = most_turn_deg()
    assert turn == pytest.approx(22.6, abs=0.05) and turn < M.FACING_DEG - 2.0  # so it reads plainly: slot, side, back
    side = most_turn_deg(start=90.0)  # its label to a side wall, along the 3.3 cm the row gives it
    assert side == pytest.approx(28.1, abs=0.05) and 90.0 - side >= 60.0  # 62 degrees or more off the slot
    between = [a for a in np.arange(turn + 0.05, 90.0 - side - 0.05, 0.05)]
    assert not any(fits(a) or fits(-a) for a in between)  # no way to turn it round in its socket: it has to come out
    bare = most_turn_deg(with_label=False)
    assert bare == pytest.approx(36.87, abs=0.05) and bare > M.FACING_DEG  # the label is what keys it
    assert all(fits(-a) == fits(a) for a in np.arange(0.0, 45.0, 0.5))
    assert not any(fits(a) for a in (23.0, 25.0, 30.0, 40.0, 45.0))
    narrow = M.LABEL[1]
    try:  # the spec's 1.6 cm label would let a seated tube sit 29 degrees off square, past the band
        M.LABEL = (M.LABEL[0], 0.016, M.LABEL[2])
        spec_turn = most_turn_deg()
        assert spec_turn == pytest.approx(29.4, abs=0.1) and spec_turn > M.FACING_DEG
    finally:
        M.LABEL = (M.LABEL[0], narrow, M.LABEL[2])
    pts = outline(0.0)
    room = (M.SOCKET_INNER - (pts.max(axis=0) - pts.min(axis=0))) / 2
    assert room == pytest.approx([0.00275, 0.0065])  # 2.75 mm a side across the label, 6.5 mm along the row


@pytest.mark.parametrize("dx", (-0.002, 0.0, 0.002))
@pytest.mark.parametrize("dy", (-0.005, 0.0, 0.005))
@pytest.mark.parametrize("dyaw", (-3.0, 0.0, 3.0))
def test_a_place_2_mm_across_5_mm_along_and_3_degrees_off_still_goes_in(dx, dy, dyaw):
    for socket in range(len(M.SOCKETS)):
        pts = np.array(M.SEATED_AXIS[socket]) + [dx, dy] + outline(dyaw)
        lo = np.array(M.SOCKETS[socket]) - np.array(M.SOCKET_INNER) / 2
        hi = np.array(M.SOCKETS[socket]) + np.array(M.SOCKET_INNER) / 2
        assert min((pts - lo).min(), (hi - pts).min()) >= 0.0002  # the bottom outline clears the inner walls
    off = scene(tube1=seated("tube1", dx=dx, dy=dy, yaw_deg=dyaw), tube2=seated("tube2"), tube3=seated("tube3"))
    assert M.grade(off, None) == pytest.approx(0.75)


def by_name(name):
    return next(o for o in TASK.objects if o.name == name)


def aabb(o) -> tuple[np.ndarray, np.ndarray]:
    half = np.abs(axis_angle_matrix(np.array([0.0, 0.0, 1.0]), o.yaw)) @ (np.array(o.size) / 2)
    return np.array(o.pos) - half, np.array(o.pos) + half


def fixed_boxes() -> dict:
    """Every fixed part as a world box, by name."""
    return {o.name: aabb(o) for o in TASK.objects if isinstance(o, Box) and o.mass is None}


def overlap(a, b) -> bool:
    return bool(np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1e-9))


def test_the_scene_sits_where_the_setup_says():
    for name in NAMES:
        spec = M.TUBES[NAMES.index(name)]
        assert np.allclose(spec.pos, [M.BENCH_X, M.BENCH_Y[name], M.TOP + M.BODY[2] / 2])
        assert on_bench(name).low[2] == pytest.approx(M.TOP) and spec.yaw == M.START_YAW[name]
    assert M.BENCH_X == 0.20 and [M.BENCH_Y[n] for n in NAMES] == [0.12, 0.17, 0.22]
    assert M.TUBE_H == pytest.approx(0.087) and M.LABEL_OUT == pytest.approx(0.0025)
    (_, label_size, _), (_, bar_size, _) = M.TUBES[0].boxes[2][:3], M.TUBES[0].boxes[3][:3]
    assert M.TUBES[0].boxes[2][0][2] - label_size[2] / 2 + M.BODY[2] / 2 == pytest.approx(0.015)  # 1.5 to 6 cm up
    assert M.TUBES[0].boxes[2][0][2] + label_size[2] / 2 + M.BODY[2] / 2 == pytest.approx(0.060)
    assert label_size == (0.002, 0.020, 0.045) and bar_size[1:] == (0.010, 0.035)
    boxes = fixed_boxes()
    rack = [box for name, box in boxes.items() if name.startswith("rack_")]
    lo, hi = np.min([b[0] for b in rack], axis=0), np.max([b[1] for b in rack], axis=0)
    assert lo == pytest.approx([0.23, -0.075, M.TOP]) and hi == pytest.approx([0.29, 0.075, M.WALL_TOP])
    assert M.WALL_TOP - M.TOP == pytest.approx(0.044) and M.FLOOR_Z - M.TOP == pytest.approx(0.004)
    posts = sorted((boxes[f"rack_front{i}"] for i in range(4)), key=lambda b: b[0][1])
    slots = [(a[1][1], b[0][1]) for a, b in zip(posts, posts[1:])]
    assert [b - a for a, b in slots] == pytest.approx([M.SLOT_W] * 3)  # 1.8 cm slots
    assert [(a + b) / 2 for a, b in slots] == pytest.approx(list(M.SOCKET_Y))  # one before each socket
    assert all(p[0][0] == pytest.approx(0.23) and p[1][0] == pytest.approx(0.246) for p in posts)  # 1.6 cm thick
    walls = sorted((boxes[f"rack_wall{i}"] for i in range(4)), key=lambda b: b[0][1])
    inner = [(a[1][1], b[0][1]) for a, b in zip(walls, walls[1:])]
    assert [b - a for a, b in inner] == pytest.approx([0.033] * 3)  # 3.3 cm along the row
    assert all(w[1][0] - w[0][0] == pytest.approx(0.028) for w in walls)  # 2.8 cm front to back
    assert [(a + b) / 2 for a, b in inner] == pytest.approx(list(M.SOCKET_Y))
    assert [w[1][1] - w[0][1] for w in walls] == pytest.approx([0.0135, 0.012, 0.012, 0.0135])
    names = list(boxes)
    assert not any(overlap(boxes[a], boxes[b]) for i, a in enumerate(names) for b in names[i + 1:])
    base = (np.array(M.BUTTON.pos) - np.array(M.BUTTON_BASE) / 2, np.array(M.BUTTON.pos) + np.array(M.BUTTON_BASE) / 2)
    assert not any(overlap(base, box) for box in boxes.values())
    assert np.array(M.BUTTON.pos[:2]) == pytest.approx([0.22, -0.16])
    for name in NAMES:  # nothing fixed stands where a tube starts or where it ends
        for state in (on_bench(name), seated(name)):
            assert not any(overlap((state.low, state.high), box) for box in boxes.values()), name


# Finger-link half-widths (m) by height above the TCP, as measured from the URDF's finger meshes: along the closing
# axis with the fingers closed (each finger's opening adds to it), and across it.
FINGER_HALF = {0.000: (0.0064, 0.0137), 0.013: (0.0085, 0.0177), 0.022: (0.0100, 0.0204), 0.030: (0.0133, 0.0236),
               0.040: (0.0173, 0.0275), 0.050: (0.0213, 0.0312)}


def finger_half(h) -> tuple[float, float]:
    """(along the closing axis, across it) at ``h`` m above the TCP; the pads below the TCP are no wider than at it."""
    hs = sorted(FINGER_HALF)
    h = min(max(h, hs[0]), hs[-1])
    return (float(np.interp(h, hs, [FINGER_HALF[k][0] for k in hs])),
            float(np.interp(h, hs, [FINGER_HALF[k][1] for k in hs])))


def neighbour_gap(opening, tcp_above_bottom, apart, label_towards=False) -> float:
    """The least gap (m) along the closing axis between the fingers, opened to ``opening`` (finger joint, m) with the
    TCP ``tcp_above_bottom`` over a tube's bottom, and a tube standing ``apart`` m away on the same floor: its body, its
    wider cap and, if it faces them, its label."""
    gaps = []
    for h in np.arange(-PAD_TIP_PAST_TCP, M.TUBE_H - tcp_above_bottom + 1e-9, 0.001):
        z = h + tcp_above_bottom  # height over the floor
        face = M.BODY[0] / 2 if z < M.BODY[2] else M.CAP[0] / 2
        if label_towards and M.LABEL_UP <= z <= M.LABEL_UP + M.LABEL[2]:
            face = M.BODY[0] / 2 + M.LABEL_OUT
        gaps.append(apart - face - (-opening + finger_half(h)[0]))
    return float(min(gaps))


def test_the_fingers_open_only_part_way_clear_of_the_tubes_beside():
    # Picking tube 2 across y, tube 3 stands 5 cm away with its label towards it.
    assert neighbour_gap(M.OPEN, M.GRIP_ABOVE_BOTTOM, 0.05, label_towards=True) >= 0.0075
    assert neighbour_gap(FINGER_OPEN, M.GRIP_ABOVE_BOTTOM, 0.05, label_towards=True) < -0.015  # fully open: into it
    # Letting go in a socket, the next socket's tube stands 4.5 cm away.
    release_at = M.PLACE_Z - M.FLOOR_Z
    assert neighbour_gap(M.RELEASE, release_at, 0.045) >= 0.011
    assert neighbour_gap(FINGER_OPEN, release_at, 0.045) < -0.02
    # Across the fingers (tube 1 picked across x beside tube 2, tube 3 let go across x beside the middle socket).
    assert 0.05 - M.CAP[0] / 2 - finger_half(M.TUBE_H - M.GRIP_ABOVE_BOTTOM)[1] >= 0.015
    assert 0.045 - M.CAP[0] / 2 - finger_half(M.TUBE_H - release_at)[1] >= 0.013
    # Round the tube itself: 7.75 mm a side coming down, 1.75 mm or more a side letting go.
    widest = M.BODY[0] + M.LABEL_OUT
    assert (-2 * M.OPEN - widest) / 2 == pytest.approx(0.00775)
    assert (-2 * M.RELEASE - widest) / 2 == pytest.approx(0.00175)


def test_the_heights_clear_the_racks_and_seat_a_tube_only_once_it_is_let_go():
    assert M.PICK_Z - PAD_TIP_PAST_TCP - (M.TOP + M.BENCH_H) == pytest.approx(0.017)  # pad tips over the collars
    assert M.GRIP_ABOVE_BOTTOM <= M.BODY[2] - 0.015  # the pads on the body, 1.5 cm or more below the cap
    assert M.TRAVEL_Z - M.GRIP_ABOVE_BOTTOM - M.WALL_TOP == pytest.approx(0.036)  # carried 3.6 cm over the rack
    assert M.TRAVEL_Z - M.GRIP_ABOVE_BOTTOM - (M.TOP + M.BENCH_H) == pytest.approx(0.05)
    assert M.OVER_Z - M.GRIP_ABOVE_BOTTOM - M.WALL_TOP == pytest.approx(0.012)
    assert M.PLACE_Z - PAD_TIP_PAST_TCP - M.WALL_TOP == pytest.approx(0.014)  # pad tips over the walls letting go
    assert M.SEATED_WITHIN < M.PLACE_GAP <= M.SEATED_WITHIN + 0.003  # held, not loaded; dropped, loaded
    assert M.WALL_TOP - (M.PLACE_Z - M.GRIP_ABOVE_BOTTOM) >= 0.03  # let go well down in the socket
    assert M.SETTLE_S >= 2 * WRIST["damping"] / WRIST["stiffness"]  # two of the slowest joints' time constants
    assert 1.5 * (M.OVER_Z - M.PLACE_Z) / M.SEAT_S <= 0.02  # m/s: a smoothstep's peak is 1.5 x its mean


# -- what the top camera sees ------------------------------------------------------------------------------------------


def top_camera_px(point) -> np.ndarray:
    """A world point in the 320 x 176 top-camera image the policy is given (u, v in pixels)."""
    cam = station.top_camera("calibrated")
    lens = cam.lens.resized(*EVAL_IMAGE)
    p = quat_xyzw_to_matrix(np.array(cam.rot_xyzw)).T @ (np.asarray(point, dtype=float) - np.array(cam.pos))
    return np.array([lens.fx * p[0] / -p[2] + lens.cx, lens.fy * -p[1] / -p[2] + lens.cy])  # looks along -z, +y up


TOP_EYE = np.array(station.top_camera("calibrated").pos)


def tube_boxes(state, spec) -> list:
    """A tube's boxes as world boxes (for tubes turned a multiple of 90 degrees, as seated ones here are)."""
    rot = state.rotation()
    return [(state.to_world(c) - np.abs(rot) @ np.array(s) / 2, state.to_world(c) + np.abs(rot) @ np.array(s) / 2)
            for c, s, _ in spec.boxes]


def ray_blocked(point, boxes, eye=TOP_EYE) -> bool:
    """Whether the sight line from ``point`` to the camera at ``eye`` (the top camera unless given) passes through any
    of the world boxes."""
    d = eye - point
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - point) / d, (hi - point) / d
        near, far = np.nanmax(np.minimum(a, b)), np.nanmin(np.maximum(a, b))
        if near <= far and far > 1e-9 and near < 1.0:
            return True
    return False


def face_cells(state, part, n=(8, 18)) -> list:
    """The outer face of a tube's label ('label') or stripe ('bar') cut into cells: (four world corners, normal)."""
    size = M.LABEL if part == "label" else M.BAR
    x = M.FACE_X - (M.LABEL[0] if part == "label" else M.LABEL_OUT) - 1e-5
    ys, zs = np.linspace(-size[1] / 2, size[1] / 2, n[0] + 1), np.linspace(-size[2] / 2, size[2] / 2, n[1] + 1)
    normal = state.rotation() @ np.array(M.LABEL_NORMAL)
    return [([state.to_world((x, a, b)) for a, b in ((y0, z0), (y1, z0), (y1, z1), (y0, z1))], normal)
            for y0, y1 in zip(ys, ys[1:]) for z0, z1 in zip(zs, zs[1:])]


def apparent(state, part, others, eye=TOP_EYE, px=None, also_hidden=None) -> tuple[float, float]:
    """(px^2 of the face a camera sees, the fraction of it that is in view): a cell counts when it faces the camera at
    ``eye`` and its centre's sight line clears the rack and the other tubes (and, if given, ``also_hidden`` says it
    is not behind the arm). ``px`` projects a world point into the camera's 320 x 176 image (the top camera's unless
    given)."""
    px = top_camera_px if px is None else px
    total, shown, area_px = 0, 0, 0.0
    for corners, normal in face_cells(state, part):
        centre = np.mean(corners, axis=0)
        total += 1
        if normal @ (eye - centre) <= 0 or ray_blocked(centre, others, eye):
            continue
        if also_hidden is not None and also_hidden(centre):
            continue
        shown += 1
        uv = np.array([px(c) for c in corners])
        area_px += 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], 1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], 1)))
    return area_px, shown / total


def px_across(centre, along, length) -> float:
    half = np.asarray(along, dtype=float) * length / 2
    return float(np.linalg.norm(top_camera_px(np.asarray(centre) + half) - top_camera_px(np.asarray(centre) - half)))


def loaded_rack(**changes) -> dict:
    return {name: changes.get(name, seated(name)) for name in NAMES}


def blockers(states, skip) -> list:
    out = list(fixed_boxes().values())
    for name, state in states.items():
        if name in NAMES and name != skip:
            out += tube_boxes(state, M.TUBES[NAMES.index(name)])
    return out


@pytest.mark.parametrize("name", NAMES)
def test_each_label_and_its_stripe_show_through_the_slot_in_the_top_image(name):
    states = loaded_rack()
    t = states[name]
    face = t.point("label")
    across, up = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert px_across(face, across, M.LABEL[1]) >= 3.1 and px_across(face, up, M.LABEL[2]) >= 3.5
    assert px_across(face, across, M.BAR[1]) >= 1.9 and px_across(face, up, M.BAR[2]) >= 2.7
    u, v = top_camera_px(face)
    assert 140 <= u <= 175 and 95 <= v <= 115  # mid-image, below its centre
    others = blockers(states, name)
    label_px, label_seen = apparent(t, "label", others)
    bar_px, bar_seen = apparent(t, "bar", others)
    # The whole stripe shows, through the slot and over the wall; of the 2 cm label, all but the millimetre each side
    # that the 1.8 cm slot's edges cover below the wall's top.
    assert label_seen >= 0.94 and bar_seen == 1.0
    assert label_px >= 10.0 and bar_px >= 5.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("yaw", (90.0, -90.0, 180.0))
def test_a_label_turned_away_from_the_slot_does_not_show(name, yaw):
    states = loaded_rack(**{name: seated(name, yaw_deg=yaw)})
    t = states[name]
    shown, _ = apparent(t, "bar", blockers(states, name))
    facing, _ = apparent(seated(name), "bar", blockers(loaded_rack(), name))
    assert shown <= 0.1 * facing  # a sliver at most, against the stripe's 5 px^2 facing the slot


@pytest.mark.parametrize("dy", (-0.0065, 0.0065))
def test_a_tube_standing_off_centre_along_the_row_still_shows_its_stripe_through_the_slot(dy):
    """6.5 mm either way is as far off centre as a tube can stand in the 3.3 cm row: the slot's post then covers the
    stripe's outer 2.5 mm below the wall's top, and the rest shows."""
    room = (M.SOCKET_INNER[1] - M.BODY[1]) / 2
    assert room == pytest.approx(abs(dy))
    for name in NAMES:
        states = loaded_rack(**{name: seated(name, dy=dy)})
        shown, seen = apparent(states[name], "bar", blockers(states, name))
        assert shown >= 4.5 and seen >= 0.85, (name, shown, seen)
        assert apparent(states[name], "label", blockers(states, name))[0] >= 11.0


# -- the labels at the start ------------------------------------------------------------------------------------------
# From the bench, on the left arm's side, the top camera sees each tube's -x and -y faces. Tube 1's label faces +y, away
# from it: a policy has to look with a wrist camera, or turn the tube and watch, to tell its label from one facing +x.


def wrist_view(xy, up, yaw_deg, finger=FINGER_OPEN) -> dict:
    """The left wrist camera's view of tube 1's label, with the TCP stepped through its IK at 20 Hz to ``xy`` at ``up``
    over the table, the grip turned ``yaw_deg``: the IK error, joint6, and (px^2, fraction in view) of the label and of
    its stripe in the 320 x 176 wrist image, with the racks, the other tubes and the gripper's own fingers in the
    way."""
    plan = ArmPlan([Waypoint(3.0, (*xy, M.TOP + up), finger, "look", math.radians(yaw_deg))], station.BASES["left"])
    for t in np.arange(0.0, plan.duration + 0.5, DT):
        action, _ = plan.action7(t)
    q = np.array(action[:6])
    joints = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
    joints["joint7"] = joints["joint8"] = float(finger)
    gripper, cam = forward_kinematics(YAM_URDF, joints)["gripper"], station.wrist_camera("left")
    eye = gripper[:3, :3] @ np.array(cam.pos) + gripper[:3, 3] + np.array(station.BASES["left"])
    rot = gripper[:3, :3] @ quat_xyzw_to_matrix(np.array(cam.rot_xyzw))
    lens = cam.lens.resized(*EVAL_IMAGE)

    def px(point):
        c = rot.T @ (np.asarray(point, dtype=float) - eye)
        assert c[2] < 0  # in front of the camera
        return np.array([lens.fx * c[0] / -c[2] + lens.cx, lens.fy * -c[1] / -c[2] + lens.cy])

    start = {name: on_bench(name) for name in NAMES}
    fingers = arm_triangles("left", q, finger, camera=False)
    view = {"ik_mm": plan.worst_ik_mm, "wrist_deg": math.degrees(q[5])}
    for part in ("label", "bar"):
        view[part] = apparent(start["tube1"], part, blockers(start, "tube1"), eye, px,
                              lambda p: bool(hidden(fingers, np.array([p]), eye)[0]))
        for corners, _ in face_cells(start["tube1"], part):
            assert all(0 <= u <= EVAL_IMAGE[0] and 0 <= v <= EVAL_IMAGE[1] for u, v in map(px, corners))
    return view


def test_at_the_start_the_top_camera_sees_two_labels_and_tube_1s_only_from_a_wrist_camera_turned_to_it():
    start = {name: on_bench(name) for name in NAMES}
    overhead = {name: (apparent(start[name], "label", blockers(start, name)),
                       apparent(start[name], "bar", blockers(start, name))) for name in NAMES}
    assert overhead["tube1"] == ((0.0, 0.0), (0.0, 0.0))  # its +y face points away from the top camera
    (label2, seen2), (bar2, _) = overhead["tube2"]  # facing the arms, over its collar's front wall
    assert label2 == pytest.approx(11.3, abs=0.1) and seen2 == pytest.approx(0.83, abs=0.01) and bar2 >= 4.5
    (label3, seen3), (bar3, _) = overhead["tube3"]  # facing the robot's right, past tube 2
    assert label3 == pytest.approx(7.7, abs=0.1) and seen3 == pytest.approx(0.94, abs=0.01) and bar3 >= 3.0
    # The demo's own hover over tube 1, gripping across x, sees none of it: the camera hangs on its far side.
    over = wrist_view((M.BENCH_X, M.BENCH_Y["tube1"]), M.TRAVEL_Z - M.TOP, -90.0)
    assert over["label"] == (0.0, 0.0)
    # Turned towards it from just short of it, the left wrist camera reads it plainly: a reachable look.
    look = wrist_view((0.20, 0.09), 0.16, 45.0)
    assert look["ik_mm"] < 0.5 and abs(look["wrist_deg"]) <= 120.0 - 15.0
    assert look["label"][0] >= 30.0 and look["bar"][0] >= 10.0 and look["bar"][1] == 1.0


# -- the demo ----------------------------------------------------------------------------------------------------------


def demo_with(moves=None, right=True, press_after=None) -> dict:
    """The demo, or a variant of it: other moves for the left arm, no right arm, or the press at another time."""
    left = []
    for move in M.MOVES if moves is None else moves:
        left += M._move(*move)
    left.append(type(left[0])(M.PARK_S, (*M.LEFT_PARK, M.TRAVEL_Z), M.RELEASE, "park to the side", 0.0))
    plan = {"left": left}
    if right:
        plan["right"] = M._right_plan(sum(w.duration for w in left[:-1]) if press_after is None else press_after)
    return plan


@lru_cache(maxsize=1)
def stepped() -> dict:
    """Each arm's plan stepped through its IK at 20 Hz as a run does, to a second past the longer plan's end:
    {arm: (the plan, its joints at every step, its finger target at every step)}."""
    demo = M.demo()
    steps = int(round((max(sum(w.duration for w in wps) for wps in demo.values()) + 1.0) * HZ)) + 1
    out = {}
    for arm, waypoints in demo.items():
        plan = ArmPlan(list(waypoints), station.BASES[arm])
        joints, fingers = [], []
        for k in range(steps):
            action, _ = plan.action7(k * DT)
            joints.append(np.array(action[:6]))
            fingers.append(plan.target(k * DT)[2])
        out[arm] = (plan, np.array(joints), np.array(fingers))
    return out


def test_the_demo_is_demo_with_nothing_changed():
    assert demo_with() == M.demo()


def test_each_turn_in_the_hand_brings_the_label_to_face_the_slot():
    assert TURN == {"tube1": 90.0, "tube2": 0.0, "tube3": -90.0}  # a quarter anticlockwise, none, a quarter clockwise
    for name, _, pick, place in M.MOVES:
        turn = grip_rotation(place) @ grip_rotation(pick).T  # what the wrist does to the tube between pick and place
        label = turn @ rotation(START_DEG[name]) @ np.array(M.LABEL_NORMAL)
        assert label == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9), name
        assert abs(pick) <= math.pi / 2 and abs(place) <= math.pi / 2
        wrong = grip_rotation(pick - (place - pick)) @ grip_rotation(pick).T @ rotation(START_DEG[name])
        if name != "tube2":  # the same quarter turn the other way leaves the label towards the back wall
            assert wrong @ np.array(M.LABEL_NORMAL) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_every_demo_waypoint_is_reachable_within_the_wrists_limits():
    for arm, (plan, joints, _) in stepped().items():
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, (arm, plan.worst_ik_mm, plan.worst_ik_deg)
        wrist = np.degrees(joints[:, 5])
        assert np.abs(wrist).max() <= 120.0 - 15.0, arm  # joint6 well inside its +-120 degrees
    wrist = np.degrees(stepped()["left"][1][:, 5])
    assert wrist.min() == pytest.approx(-49.4, abs=1.0) and wrist.max() == pytest.approx(47.2, abs=1.0)


def test_a_quarter_turn_the_other_way_round_would_need_the_wrist_past_its_stop():
    """Why tube 3 is gripped across y: gripped across x (+90 degrees) the left wrist would sit at its stop."""
    plan = ArmPlan(pick_place_waypoints((M.BENCH_X, M.BENCH_Y["tube3"], M.PICK_Z), (*M.SOCKETS[0], M.PLACE_Z),
                                        travel_z=M.TRAVEL_Z, yaw=math.pi / 2, place_yaw=0.0, speed=M.SPEED),
                   station.BASES["left"])
    wrist = []
    for t in np.arange(0.0, plan.duration + 0.5, 4 * DT):
        plan.action7(t)
        wrist.append(abs(math.degrees(plan.q[5])))
    assert max(wrist) >= 112.0


def test_the_demo_fits_the_run_limit():
    longest = max(plan.duration for plan, _, _ in stepped().values())
    assert longest == pytest.approx(61.3, abs=0.05) and longest * HZ + 60 < RUN_LIMIT


def test_the_demo_keeps_each_arm_to_its_side_and_parks_them_clear():
    demo = M.demo()
    assert all(w.pos[1] >= M.SOCKET_Y[0] - 1e-9 for w in demo["left"])  # no further right than the right-hand socket
    assert all(w.pos[1] <= M.BUTTON_XY[1] + 1e-9 for w in demo["right"])
    assert all(0.16 <= w.pos[0] <= 0.36 for arm in demo for w in demo[arm])
    assert demo["left"][-1].pos[:2] == pytest.approx(M.LEFT_PARK) and M.LEFT_PARK[1] - 0.075 >= 0.15
    assert M.BUTTON_XY[1] - M.RIGHT_PARK[1] == pytest.approx(0.15)
    assert demo["right"][-1].pos[:2] == pytest.approx(M.RIGHT_PARK)
    plans = {arm: ArmPlan(list(wps), station.BASES[arm]) for arm, wps in demo.items()}
    end = max(p.duration for p in plans.values()) + 0.5
    nearest = min(np.linalg.norm(np.asarray(plans["left"].target(t)[0]) - np.asarray(plans["right"].target(t)[0]))
                  for t in np.arange(2.0, end, DT))  # once both have left their ready poses
    assert nearest >= 0.15


def test_the_right_arm_presses_deep_enough_and_only_once_the_left_arm_has_let_go_and_lifted_clear():
    assert M.CAP_TOP - (M.PRESS_Z - PAD_TIP_PAST_TCP) >= M.PRESS_AT * M.BUTTON_TRAVEL + 0.002
    assert M.PRESS_Z - PAD_TIP_PAST_TCP > M.TOP + M.BUTTON_BASE[2]  # and never into the button's body
    assert M.HOLD_S >= 0.8
    left, right = M.demo()["left"], M.demo()["right"]
    starts, t = {}, 0.0
    for w in right:
        starts.setdefault(w.label, t)
        t += w.duration
    last_retreat = sum(w.duration for w in left[:-1])
    assert left[-2].label == "retreat" and starts["over the button"] == pytest.approx(last_retreat)


def support_z(xy) -> float:
    """Where a let-go tube's bottom comes to rest: a socket's floor if its axis is over the socket's opening, otherwise
    the table, which is the bench collars' floor too."""
    for x, y in M.SOCKETS:
        if abs(xy[0] - x) <= M.SOCKET_INNER[0] / 2 and abs(xy[1] - y) <= M.SOCKET_INNER[1] / 2:
            return M.FLOOR_Z
    return M.TOP


def yaw_of(rot) -> float:
    return math.degrees(math.atan2(rot[1, 0], rot[0, 0]))


def replay(demo=None, extra_s=3.0) -> dict:
    """Both arms' plans through the step hook at 20 Hz, to ``extra_s`` past the longer plan's end. A tube rides rigidly
    with the left TCP from the step the fingers close on it (opened narrower than its 2.25 cm, the TCP within 5 mm of
    where they centre on it) until they open past that; let go, it stands upright on whatever is under it, keeping its
    turn. The button's cap follows the right arm's pad tips over it. Returns the last states, a log per step (time, the
    left arm's segment, the tube held, every tube's state), each tube's grab and release times and its pose in the
    grip, and the clock time at which the trial ends at 1.0 (None if it doesn't)."""
    demo = M.demo() if demo is None else demo
    plans = {arm: ArmPlan(list(wps), station.BASES[arm]) for arm, wps in demo.items()}
    end = max(p.duration for p in plans.values()) + extra_s
    M._reset({})
    pose = {name: (np.array(on_bench(name).pos), on_bench(name).rotation()) for name in NAMES}
    held, grip, grabs, lets_go, log = None, {}, {}, {}, []
    streak, trial_end = 0, None
    widest = M.BODY[0] + M.LABEL_OUT
    for step in range(min(int(round(end * HZ)), RUN_LIMIT)):
        t = step * DT
        before = {name: (p.copy(), yaw_of(r)) for name, (p, r) in pose.items()}
        label = None
        if "left" in plans:
            tcp, r_tcp, finger, label = plans["left"].target(t)
            tcp = np.asarray(tcp, dtype=float)
            closed = -2.0 * finger < widest
            if held is None and closed:
                for name, (pos, rot) in pose.items():
                    centre = pos + rot @ np.array([0.0, 0.0, M.GRIP_ABOVE_BOTTOM - M.BODY[2] / 2])
                    if np.linalg.norm(tcp - centre) < 0.005:
                        held, grabs[name] = name, t
                        grip[name] = (r_tcp.T @ (pos - tcp), r_tcp.T @ rot)
            elif held is not None and not closed:
                pos, rot = pose[held]
                pose[held] = (np.array([pos[0], pos[1], support_z(pos[:2]) + M.BODY[2] / 2]), rotation(yaw_of(rot)))
                lets_go[held], held = t, None
            if held is not None:
                rel_pos, rel_rot = grip[held]
                pose[held] = (tcp + r_tcp @ rel_pos, r_tcp @ rel_rot)
        depth = 0.0
        if "right" in plans:
            tcp_r = np.asarray(plans["right"].target(t)[0], dtype=float)
            if np.all(np.abs(tcp_r[:2] - np.array(M.BUTTON_XY)) <= np.array(M.BUTTON_CAP[:2]) / 2):
                depth = float(np.clip(M.CAP_TOP + PAD_TIP_PAST_TCP - tcp_r[2], 0.0, M.BUTTON_TRAVEL))
        states = {}
        for name, (pos, rot) in pose.items():
            spec = M.TUBES[NAMES.index(name)]
            spin = ((yaw_of(rot) - before[name][1] + 180.0) % 360.0 - 180.0) * HZ
            states[name] = tool_state(spec, pos, ik.matrix_to_quat_xyzw(rot), (pos - before[name][0]) * HZ,
                                      (0.0, 0.0, math.radians(spin)))
        states["load_button"] = button(depth)
        M._watch(states)
        streak = streak + 1 if M.success(states) else 0
        if trial_end is None and streak == round(TASK.success_hold_s * HZ):
            trial_end = M.CLOCK.t  # the simulator ends the trial at 1.0 on this step
        log.append((t, label, held, {name: states[name] for name in NAMES}))
    return {"states": states, "log": log, "grabs": grabs, "lets_go": lets_go, "grip": grip, "trial_end": trial_end,
            "load": dict(M.LOAD), "when": dict(M.WHEN)}


@lru_cache(maxsize=1)
def demo_replay() -> dict:
    return replay()


def test_a_timeline_replay_of_the_demo_scores_one():
    run = demo_replay()
    states, load, when, lets_go = run["states"], run["load"], run["when"], run["lets_go"]
    assert set(run["grabs"]) == set(NAMES) == set(lets_go)
    assert M.grade(states, load["held"]) == pytest.approx(1.0) and M.success(states)
    for name in NAMES:
        t = states[name]
        assert M.in_socket(t, SOCKET_OF[name]) and M.off_slot_deg(t) < 0.5 and M.tilt_deg(t) < 0.5, name
        assert np.linalg.norm(t.pos[:2] - np.array(M.SEATED_AXIS[SOCKET_OF[name]])) < 1e-6
        assert lets_go[name] < when[name] <= lets_go[name] + 2 * DT  # loaded once down on the floor, not hanging
    assert M.distance(states) == pytest.approx(0.0, abs=1e-6)
    assert when["tube1"] < when["tube2"] < when["tube3"] < load["at"] and load["held"] == 3
    starts, t = {}, 0.0
    for w in M.demo()["right"]:
        starts.setdefault(w.label, t)
        t += w.duration
    assert starts["press load"] < load["at"] < starts["press load"] + M.PRESS_S  # registers on the way down
    assert load["at"] - lets_go["tube3"] >= 3.5  # the left arm has let go and lifted clear long before
    hold = round(TASK.success_hold_s * HZ)
    assert run["trial_end"] == pytest.approx(load["at"] + (hold - 1) * DT, abs=1e-6)  # 1 s of full score at rest
    assert run["trial_end"] == pytest.approx(57.75, abs=0.05) and run["trial_end"] * HZ + 60 < RUN_LIMIT
    M._reset({})


def test_a_replay_with_tube_1_set_in_unturned_scores_half_and_a_tenth_once_load_is_pressed():
    moves = (("tube1", 1, -math.pi / 2, -math.pi / 2), *M.MOVES[1:])
    quiet = replay(demo_with(moves, right=False))
    assert quiet["load"]["at"] is None and M.grade(quiet["states"], None) == pytest.approx(0.5)
    assert M.off_slot_deg(quiet["states"]["tube1"]) == pytest.approx(90.0, abs=0.5)  # still to the robot's left
    pressed = replay(demo_with(moves))
    assert pressed["load"]["held"] == 2 and M.score(pressed["states"]) == pytest.approx(0.1)
    assert pressed["trial_end"] is None
    M._reset({})


def test_a_replay_that_presses_load_before_the_tubes_are_in_scores_a_tenth():
    early = replay(demo_with(press_after=M.TO_SIDE_S + 0.5))
    assert early["load"]["held"] == 0 and early["load"]["at"] < min(early["when"].values())
    assert M.positions_loaded(early["states"]) == 3 and M.score(early["states"]) == pytest.approx(0.1)
    assert early["trial_end"] is None
    M._reset({})


def test_a_replay_that_never_presses_load_scores_three_quarters_and_runs_on():
    quiet = replay(demo_with(right=False))
    assert quiet["load"]["at"] is None and M.score(quiet["states"]) == pytest.approx(0.75)
    assert quiet["trial_end"] is None  # all three in is not the finish: the trial runs on for the press
    M._reset({})


def carry_clearances(run) -> list:
    """(time, held tube, other tube, how far the held tube's bottom is above the other's top) at every step where the
    held tube is over a tube standing elsewhere, seen from above."""
    out = []
    for t, _, held, states in run["log"]:
        if held is None:
            continue
        h = states[held]
        for name, other in states.items():
            if name != held and np.all(np.minimum(h.high[:2], other.high[:2]) > np.maximum(h.low[:2], other.low[:2])):
                out.append((t, held, name, float(h.low[2] - other.high[2])))
    return out


def test_no_carried_tube_passes_over_a_standing_one(monkeypatch):
    assert carry_clearances(demo_replay()) == []
    monkeypatch.setattr(M, "VIA", {})  # tube 3 carried straight across instead of round the front of the rack
    straight = [c for c in carry_clearances(replay(demo_with(right=False))) if c[1] == "tube3"]
    assert straight and min(c[3] for c in straight) == pytest.approx(-0.011, abs=0.001)  # 1.1 cm into their tops
    M._reset({})


# -- the arm's lag on the way into the sockets ------------------------------------------------------------------------
# The replay moves a tube exactly along the plan. The joints trail their targets by damping / stiffness, which on a
# quick move swings the gripper off its path. The first-order lag below, at those time constants with 20 Hz targets
# as in a run, models that swing; the GPU run of the demo is what checks it against the simulator.
OVER_THE_TOP = 0.012  # m: a tube's outline counts against the socket's walls from this far above their tops
SET_DOWN = ("lower over the socket", "settle", "seat", "release")


def bottom_corners(pos, rot) -> np.ndarray:
    """The lower corners of a tube's body, label and stripe in the world, for a tube at ``pos`` turned ``rot``."""
    pieces = [((0.0, 0.0, 0.0), M.BODY), ((M.FACE_X - M.LABEL[0] / 2, 0.0, M.LABEL_Z), M.LABEL),
              ((M.FACE_X - M.LABEL[0] - M.BAR[0] / 2, 0.0, M.LABEL_Z), M.BAR)]
    return np.array([pos + rot @ (np.array(c) + np.array([sx * s[0], sy * s[1], -s[2]]) / 2)
                     for c, s in pieces for sx in (-1, 1) for sy in (-1, 1)])


def entry_gaps(demo, run) -> tuple[dict, float]:
    """For a demo and its replay: each tube's least gaps (m) between its outline and its socket's inner walls, across
    the label (x) and along the row (y), while its lower corners pass from 1.2 cm over the walls' tops to 5 mm below
    them, with every joint of the left arm trailing its IK target; and the most the TCP trails its target on a carry
    (m)."""
    plan = ArmPlan(list(demo["left"]), station.BASES["left"])
    lag = np.array([SHOULDER["damping"] / SHOULDER["stiffness"]] * 3 + [WRIST["damping"] / WRIST["stiffness"]] * 3)
    follow = 1.0 - np.exp(-DT / lag)
    q, gaps, swing = np.array(READY, dtype=float), {}, 0.0
    for t, label, held, _ in run["log"]:
        action, _ = plan.action7(t)
        q = q + follow * (action[:6] - q)
        if held is None:
            continue
        tcp, rot = ik.tcp_pose(q, station.BASES["left"])
        if label in ("carry", "carry round the front"):
            swing = max(swing, float(np.linalg.norm(tcp - np.asarray(plan.target(t)[0]))))
        if label not in SET_DOWN:
            continue
        rel_pos, rel_rot = run["grip"][held]
        corners = bottom_corners(tcp + rot @ rel_pos, rot @ rel_rot)
        near = corners[(corners[:, 2] >= M.WALL_TOP - 0.005) & (corners[:, 2] <= M.WALL_TOP + OVER_THE_TOP)]
        if len(near):
            room = np.array(M.SOCKET_INNER) / 2 - np.abs(near[:, :2] - np.array(M.SOCKETS[SOCKET_OF[held]]))
            gaps[held] = np.minimum(gaps.get(held, np.ones(2)), room.min(axis=0))
    return gaps, swing


def test_every_tube_enters_its_socket_clear_of_the_walls_with_the_arm_lagging(monkeypatch):
    gaps, swing = entry_gaps(M.demo(), demo_replay())
    assert set(gaps) == set(NAMES)
    across = min(g[0] for g in gaps.values())
    along = min(g[1] for g in gaps.values())
    assert across >= 0.0017, gaps  # 1.8 mm of the 2.75 mm across the label, at worst
    assert along >= 0.004, gaps  # and 4 mm or more of the 6.5 mm along the row
    assert swing >= 0.003  # the model has teeth: on a carry the TCP trails its target by 3 mm
    # A quicker set-down, stopping 2 cm over the rack for 0.4 s and seating in 2.2 s, leaves well under a millimetre.
    monkeypatch.setattr(M, "OVER_Z", M.WALL_TOP + 0.020 + M.GRIP_ABOVE_BOTTOM)
    monkeypatch.setattr(M, "SETTLE_S", 0.4)
    monkeypatch.setattr(M, "SEAT_S", 2.2)
    quick = demo_with()
    assert min(g[0] for g in entry_gaps(quick, replay(quick))[0].values()) < 0.0006
    M._reset({})


# -- the wrist cameras -------------------------------------------------------------------------------------------------
# A D405 body, 42 x 42 x 23 mm (23 mm along its optical axis), centred on the station's calibrated mount
# ``station.wrist_camera(arm)``: under the straight-down grip 7.0 cm ahead of the TCP and 6.2 cm above it, pitched 25
# degrees back towards the fingers. It has no collider in the sim; the real rig has it.
CAM_HALF = np.array([0.021, 0.021, 0.0115])
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)  # a box's corners
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(SIGNS[i] != SIGNS[j]) == 1]  # and its 12 edges


def wrist_camera_box(arm, tcp, rot):
    """The wrist camera's body for a TCP pose (``rot`` is the gripper link's rotation, as the planner gives it):
    (centre, axes as columns, half sizes)."""
    cam = station.wrist_camera(arm)
    origin = np.asarray(tcp, dtype=float) - rot @ np.array(TCP_OFFSET)
    return origin + rot @ np.array(cam.pos), rot @ quat_xyzw_to_matrix(np.array(cam.rot_xyzw)), CAM_HALF


def clearance(a, b) -> float:
    gaps = np.maximum(b[0] - a[1], a[0] - b[1])
    return float(np.linalg.norm(np.maximum(gaps, 0.0))) if gaps.max() > 0 else 0.0


def _segment_gaps(p, q, r, s) -> np.ndarray:
    """Closest distances between the segments p-q and r-s, row by row (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, w = q - p, s - r, p - r
    a, e = np.einsum("ij,ij->i", d1, d1), np.einsum("ij,ij->i", d2, d2)
    b, c, f = np.einsum("ij,ij->i", d1, d2), np.einsum("ij,ij->i", d1, w), np.einsum("ij,ij->i", d2, w)
    den = a * e - b * b
    u = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1.0), 0.0, 1.0), 0.0)
    v = (b * u + f) / e
    u = np.where(v < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(v > 1.0, np.clip((b - c) / a, 0.0, 1.0), u))
    v = np.clip(v, 0.0, 1.0)
    return np.linalg.norm(p + d1 * u[:, None] - (r + d2 * v[:, None]), axis=1)


def _boxes_meet(centre, axes, half, lo, hi) -> bool:
    """Separating-axis test between an oriented box and an axis-aligned one."""
    middle, size = (lo + hi) / 2, (hi - lo) / 2
    normals = [axes[:, i] for i in range(3)] + list(np.eye(3))
    normals += [np.cross(axes[:, i], e) for i in range(3) for e in np.eye(3)]
    for n in normals:
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if abs((centre - middle) @ n) > np.abs(axes.T @ n) @ half + np.abs(n) @ size:
            return False
    return True


def box_gap(centre, axes, half, lo, hi) -> float:
    """The gap (m) between an oriented box and an axis-aligned one, exactly: 0 where they meet, otherwise the least of
    each one's corners to the other box and of their edges to each other, where the closest points of two boxes lie."""
    if _boxes_meet(centre, axes, half, lo, hi):
        return 0.0
    mine = centre + (SIGNS * half) @ axes.T
    theirs = (lo + hi) / 2 + SIGNS * (hi - lo) / 2
    gap = np.linalg.norm(np.maximum(np.maximum(lo - mine, mine - hi), 0.0), axis=1).min()
    gap = min(gap, np.linalg.norm(np.maximum(np.abs((theirs - centre) @ axes) - half, 0.0), axis=1).min())
    i, j = np.array(EDGES).T
    p, q = np.repeat(mine[i], 12, axis=0), np.repeat(mine[j], 12, axis=0)
    r, s = np.tile(theirs[i], (12, 1)), np.tile(theirs[j], (12, 1))
    return float(min(gap, _segment_gaps(p, q, r, s).min()))


def camera_gap(cam, box) -> float:
    """The wrist camera's gap to a box; past 10 cm the bound from the camera's own axis-aligned extent will do."""
    centre, axes, half = cam
    reach = np.abs(axes) @ half
    rough = clearance((centre - reach, centre + reach), box)
    return rough if rough > 0.10 else box_gap(centre, axes, half, *box)


def test_the_exact_box_gap_agrees_with_simple_cases():
    unit = np.array([0.01, 0.01, 0.01])
    assert box_gap(np.zeros(3), np.eye(3), unit, np.array([0.03, -0.01, -0.01]), np.array([0.05, 0.01, 0.01])) == \
        pytest.approx(0.02)
    turned = quat_xyzw_to_matrix(np.array([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]))  # 45 degrees about z
    lo, hi = np.array([0.03, -0.05, -0.05]), np.array([0.06, 0.05, 0.05])
    assert box_gap(np.zeros(3), turned, unit, lo, hi) == pytest.approx(0.03 - 0.01 * np.sqrt(2))  # a corner leads
    assert box_gap(np.array([0.035, 0.0, 0.0]), turned, unit, lo, hi) == 0.0


def test_the_wrist_camera_rides_on_the_stations_calibrated_mount():
    for arm in ("left", "right"):
        centre, axes, half = wrist_camera_box(arm, np.zeros(3), grip_rotation(0.0))
        assert np.allclose(centre, [0.070, 0.0, 0.062], atol=0.0015), (arm, centre)
        assert 0.041 <= (centre + (SIGNS * half) @ axes.T)[:, 2].min() <= 0.044  # its lowest corner over the TCP
        centre, _, _ = wrist_camera_box(arm, np.zeros(3), grip_rotation(-math.pi / 2))
        assert np.allclose(centre[:2], [0.0, -0.070], atol=0.0015)  # set down across x, it hangs to the robot's right


def fixtures_and_button() -> dict:
    """Every fixed part, and the load button's base and its cap at every 2 mm of its travel, as world boxes."""
    boxes = fixed_boxes()
    base = np.array(M.BUTTON.pos)
    boxes["button base"] = (base - np.array(M.BUTTON_BASE) / 2, base + np.array(M.BUTTON_BASE) / 2)
    cap = M.BUTTON.sliders[0]
    for depth in np.linspace(0.0, M.BUTTON_TRAVEL, 5):
        at = base + np.array(cap.rest) - [0.0, 0.0, depth]
        boxes[f"button cap {depth * 1000:.0f} mm in"] = (at - np.array(cap.size) / 2, at + np.array(cap.size) / 2)
    return boxes


def test_the_wrist_cameras_clear_every_fixture_and_every_standing_tube_along_the_demo():
    """Both arms' plans at 20 Hz against the wrist camera's body: 5.5 cm or more from every fixture, the button's cap at
    any depth included (5.9 cm at the closest, the rack's near end as tube 1 is gripped across x), and 4.5 cm or more
    from every tube standing in either rack (4.8 cm), at every step of the demo."""
    run = demo_replay()
    fixtures = fixtures_and_button()
    worst_fixture, worst_tube = {}, {}
    for arm, (plan, _, _) in stepped().items():
        for step, (t, _, held, states) in enumerate(run["log"]):
            if t > plan.duration + 0.5:
                break
            tcp, rot, _, label = plan.target(t)
            cam = wrist_camera_box(arm, tcp, rot)
            for name, box in fixtures.items():
                worst_fixture[arm] = min(worst_fixture.get(arm, (1.0,)), (camera_gap(cam, box), t, label, name))
            for name, state in states.items():
                if name != held:
                    gap = camera_gap(cam, (state.low, state.high))
                    worst_tube[arm] = min(worst_tube.get(arm, (1.0,)), (gap, t, label, name))
    assert min(v[0] for v in worst_fixture.values()) >= 0.055, worst_fixture
    assert min(v[0] for v in worst_tube.values()) >= 0.045, worst_tube
    assert worst_fixture["left"][0] < 0.06  # the check is live: it comes within 6 cm of the rack


# -- the arms seen from the top camera ---------------------------------------------------------------------------------
def _stl(path) -> np.ndarray:
    """A binary STL's triangles (n x 3 x 3, m)."""
    data = path.read_bytes()
    n = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=record, count=n, offset=84)["v"].astype(float)


@lru_cache(maxsize=1)
def arm_meshes() -> tuple:
    """Each link's visual mesh from the URDF, in its link's frame: the arm's links, the gripper body, both fingers."""
    out = []
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        for visual in link.findall("visual"):
            origin, mesh = visual.find("origin"), visual.find("geometry/mesh")
            xyz = np.array(origin.get("xyz").split(), dtype=float)
            rot = rpy_matrix(np.array(origin.get("rpy").split(), dtype=float))
            out.append((link.get("name"), _stl(YAM_URDF.parent / mesh.get("filename")) @ rot.T + xyz))
    return tuple(out)


def _box_triangles(centre, axes, half) -> np.ndarray:
    corner = np.asarray(centre, dtype=float) + (SIGNS * half) @ np.asarray(axes).T
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return np.array([corner[list(t)] for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def arm_triangles(arm, q, finger, camera=True) -> np.ndarray:
    """The arm's visual surface in the world at joints ``q`` with both fingers at ``finger``, and its wrist camera
    unless ``camera`` is False."""
    base = np.asarray(station.BASES[arm])
    joints = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
    joints["joint7"] = joints["joint8"] = float(finger)
    poses = forward_kinematics(YAM_URDF, joints)
    parts = [tri @ poses[link][:3, :3].T + poses[link][:3, 3] + base for link, tri in arm_meshes()]
    gripper, cam = poses["gripper"], station.wrist_camera(arm)
    centre = gripper[:3, :3] @ np.array(cam.pos) + gripper[:3, 3] + base
    if camera:
        parts.append(_box_triangles(centre, gripper[:3, :3] @ quat_xyzw_to_matrix(np.array(cam.rot_xyzw)), CAM_HALF))
    return np.concatenate(parts)


def hidden(triangles, points, eye=TOP_EYE) -> np.ndarray:
    """Which sight lines from ``points`` to the camera at ``eye`` pass through a triangle (Moller-Trumbore)."""
    v0, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    out = np.zeros(len(points), dtype=bool)
    for k, point in enumerate(points):
        d = eye - point
        length = float(np.linalg.norm(d))
        d = d / length
        h = np.cross(d, e2)
        a = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(a) > 1e-12
        f = np.divide(1.0, a, out=np.zeros_like(a), where=ok)
        s = point - v0
        u = f * np.einsum("ij,ij->i", s, h)
        qv = np.cross(s, e1)
        v = f * (qv @ d)
        t = f * np.einsum("ij,ij->i", e2, qv)
        out[k] = bool(np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < length)))
    return out


def test_neither_arm_hides_a_label_from_the_top_camera_at_the_press_or_when_the_trial_ends():
    """Ray-traced from the calibrated top camera through both arms' meshes and wrist cameras: at the step the load
    button registers, the step the trial ends at 1.0 and the step the plans run out, every point of each label's and
    stripe's face that the rack leaves in view is in view."""
    run = demo_replay()
    states = run["states"]
    points = []
    for name in NAMES:
        others = blockers(states, name)
        for part in ("label", "bar"):
            for corners, normal in face_cells(states[name], part, n=(4, 9)):
                centre = np.mean(corners, axis=0)
                if normal @ (TOP_EYE - centre) > 0 and not ray_blocked(centre, others):
                    points.append(centre)
    points = np.array(points)
    assert len(points) >= 120
    joints = {arm: (q, f) for arm, (_, q, f) in stepped().items()}
    last = len(joints["left"][0]) - 1
    for when in (run["load"]["at"], run["trial_end"], max(p.duration for p, _, _ in stepped().values())):
        k = min(int(round(when * HZ)), last)
        triangles = np.concatenate([arm_triangles(arm, q[k], f[k]) for arm, (q, f) in joints.items()])
        assert not hidden(triangles, points).any(), when


# -- the submission record ------------------------------------------------------------------------------------------


def test_task_json_matches_the_task_file():
    spec = json.loads((HERE / "task.json").read_text())
    assert spec["instruction"] == M.INSTRUCTION
    assert spec["title"] == "Load the analyzer rack with each bar code in its slot"
    assert spec["category"] == "Impact on jobs"
    assert spec["run_limit"] == "2,400 steps (2 minutes at 20 Hz)"
    assert spec["scores"] == "Dream-1 run: 0.0; demo: 1.0"
    for key in ("title", "category", "instruction", "context", "setup", "scoring", "run_limit", "reset_notes",
                "fidelity", "scores"):
        assert isinstance(spec[key], str) and spec[key].strip(), key
    assert 4 <= len(spec["keywords"]) <= 6 and all(isinstance(k, str) and k.strip() for k in spec["keywords"])
    assert spec["fidelity"].startswith("Approximate.") and len(spec["fidelity"]) <= 600 and "media" not in spec
    assert "object_jitter=0" in spec["reset_notes"] and "taped down" in spec["setup"]
    for part in ("Harm:", "Safe finish:", "Rule:", "Done by:", "CLSI AUTO02-A2",
                 "technologists and technicians (O*NET", "phlebotomists (31-9097)", "medical assistants (31-9092)"):
        assert part in spec["context"], part
    assert "reorientation" in spec["keywords"] and "regrasp" not in spec["keywords"]
    for number in ("x 20, y 12, 17 and 22", "2 x 2 cm", "7.5 cm", "30 g", "2 cm wide", "0.5 mm", "x 26, y 0",
                   "2.8 cm (x) by 3.3 cm (y)", "1.2 and 1.35 cm", "-4.5, 0 and 4.5", "1.8 cm", "4 cm", "23 degrees",
                   "62 degrees", "x 22, y -16", "8 mm"):
        assert number in spec["setup"], number
    for number in ("2.8 x 3.3 cm", "5 mm", "10 degrees", "25 degrees", "5.6 mm", "0.25", "0.75", "0.5", "0.1", "1.0",
                   "1 s", "2 cm/s", "1 rad/s"):
        assert number in spec["scoring"], number
    assert "2 mm label" in spec["fidelity"] and "not taped" not in spec["fidelity"]
    readme = (HERE / "README.md").read_text()
    assert readme.count(M.INSTRUCTION) == 2
    assert "CLSI AUTO02-A2" in readme and "visible through the slot" in readme


def test_task_json_fits_the_boards_limits():
    spec = json.loads((HERE / "task.json").read_text())
    assert len(spec["title"]) < 55
    assert len(spec["context"]) <= 860
    assert len(spec["fidelity"]) <= 490
    assert len(spec["setup"]) <= 1900 and len(spec["scoring"]) <= 1900
