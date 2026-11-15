# Load the analyzer rack with each bar code in its slot

Three sample tubes stand in a bench rack, each with a white bar-code label down one side. Beside them an analyzer rack
has three sockets behind a front wall with a slot in front of each, and a load button sends the rack in. The robot is
told to load the tubes one per position, each turned so its label faces out through its slot, then press load. The
labels start facing three ways: one to the robot's left, one towards the arms, one to its right. The first faces away
from the overhead camera, so the robot has to look closer to find it. So the job is to find which way each label
faces, turn its tube to suit (a quarter turn anticlockwise, none, a quarter turn clockwise), and press load only once
the rack is right. On a real analyzer a tube with its bar code turned away from the reader is a no-read.

The task needs both arms: the left loads the tubes, turning two of them in the hand, and the right presses the load
button.

## Files

| File | What |
|---|---|
| `load_the_analyzer_rack_with_each_bar_code_in_its_slot.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` and the scripted `demo()` |
| `benchkit.py` | Tools built from several boxes (each tube with its label and stripe, one rigid body) with spring-loaded sliders (the button's cap), and per-step hooks |
| `plantkit.py` | A step clock, tilt and press checks, and the shared colours |
| `test_load_the_analyzer_rack_with_each_bar_code_in_its_slot.py` | Offline tests: no GPU needed |

`benchkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the simulator's environment (the Robocurve Brev launchable, or yam-isaac-lab-arena's `setup.sh`), with a Dream API
key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=load_the_analyzer_rack_with_each_bar_code_in_its_slot.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the three sample tubes into the analyzer rack, one per position, each turned so its bar code label faces out through the rack's front slot, then press the rack's load button." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rack-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=load_the_analyzer_rack_with_each_bar_code_in_its_slot.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
  --instruction "Load the three sample tubes into the analyzer rack, one per position, each turned so its bar code label faces out through the rack's front slot, then press the rack's load button." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rack-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/rack-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. A trial ends at 1.0 once the load button has been pressed with all three tubes
loaded right and they have stood still for 1 s; otherwise it scores the last step.

Offline tests (no GPU), with `yam_arena` importable:

```bash
python -m pytest -q test_load_the_analyzer_rack_with_each_bar_code_in_its_slot.py
```

## Scoring

Scored from the end state, and from the first press of the load button. A position is loaded right when a tube stands in
its socket with its label facing out through the slot: the tube's centre inside the 2.8 x 3.3 cm opening seen from
above, its bottom within 5 mm of the socket floor (down on the floor, not hanging above it in the fingers), leaning no
more than 10 degrees, and its label's direction, read from the tube's own orientation, within 25 degrees of -x seen from
above. Each socket counts once. The button is pressed when its cap goes in 5.6 mm (70% of its 8 mm travel). What the
rack holds at that first press is recorded, because a real analyzer takes the rack in then.

| Outcome | Score |
|---|---|
| All three tubes in with their labels facing the slots, then the load button pressed | 1.0 |
| All three right, load not pressed | 0.75 |
| Two right (the third missing, or in its socket unturned or turned the wrong way), load not pressed | 0.5 |
| One right, load not pressed | 0.25 |
| Load pressed while fewer than three were right (a socket empty or a label turned away), at least one right at the end | 0.1 |
| Load pressed with all three right, then a tube taken out again | 0.25 per position still right |
| Tubes moved but none right, load not pressed | 0 |
| A tube dropped off the bench | 0 |
| Load pressed and no position ends right | 0 |
| Nothing done | 0 |

## Design notes

- **The label is the job.** Tube 1's label starts to the robot's left and needs a quarter turn anticlockwise, tube 2's
  faces the arms already, tube 3's faces the robot's right and needs a quarter turn clockwise. A tube set in as it
  comes, or turned the wrong way, has its label against a side wall or the back wall and earns nothing, so two right and
  one not is 0.5. The label's direction is read from each tube's own rotation, never from which socket it is in.
- **Pressing load is the commitment.** The analyzer takes the rack at the first press, so pressing before the rack is
  right sends a tube to a no-read or an empty position: at most 0.1, below every honest partial result (0.25 for one
  tube right and stopping). Pressing with nothing loaded, or doing nothing, scores 0. A later press doesn't undo the
  first. Full marks need all three right at the first press and at the end. The trial then ends 1 s later.
- **A seated tube reads plainly.** Each socket is 2.8 cm from front to back and 3.3 cm along the row. A 2 cm square tube
  could turn nearly 37 degrees in it, but its 2 mm label covers the face's full width and meets the front and back walls
  once a tube facing the slot is past 22.6 degrees off square, inside the 25-degree band. A tube with its label to a
  side wall can turn up to 28 degrees along the row and still sits 62 degrees or more off the slot. So a seated tube's
  label faces the slot, a side wall or the back, and nothing in between. (A 1.6 cm label would let a tube sit 29 degrees
  off and read neither way.) It also means a tube can't be turned in its socket by pushing: it has to come out. The fit
  leaves 2.75 mm a side across the label and 6.5 mm along the row. A place 2 mm across, 5 mm along and 3 degrees off
  still goes in. The wider row also lets a tube that is let go off centre rest leaning 12.8 to 19.5 degrees against a
  divider; that fails the 10-degree upright rule, so it doesn't count until it is stood up straight.
- **Loaded means down on the floor.** The floor check is 5 mm and the demo lets go with the tube's bottom 7 mm up, so a
  tube hanging in the fingers above the floor isn't loaded; it counts once it drops.
- **The labels show in the top camera.** Projected through the calibrated top camera at Dream-1's 320 x 176, a label
  facing its slot is 4.0 x 3.7 px and its stripe 2.0 x 2.85 px (5.6 px² in view). All of the stripe shows, through the
  slot and over the wall, and 95 to 100% of the label: the 1.8 cm slot hides the label's outer millimetre each side
  below the wall's top. The tubes are dark red, like tubes of blood, so the white label stands out as well as its
  stripe. A tube standing 6.5 mm off centre along the row, as far as it can, still shows 5.0 px² of its stripe. A stripe
  turned to a side wall shows a sliver of at most 0.55 px², and one turned to the back shows nothing. Ray-traced through
  both arms' meshes and wrist cameras, every point of every label is in view at the press, when the trial ends and when
  the plans run out.
- **Tube 1's label has to be looked for.** From the bench, on the left arm's side, the top camera sees the tubes' faces
  towards the arms and towards the robot's right. At the start it shows tube 2's label at 11.3 px² (stripe 4.7 px²) and
  tube 3's at 7.7 px² (stripe 3.1 px²). Tube 1's label faces the robot's left, away from the top camera, and doesn't
  show at all, so from there it can't be told from a label facing the back. A policy has to look before it turns tube 1.
  The demo's own hover over it sees nothing either, but the left wrist camera turned towards it from just short of it
  (the TCP over x 20, y 9, 16 cm up, the grip turned 45 degrees, joint6 at -92 degrees) sees the label at 38.9 px² and
  its stripe at 13.9 px², past the fingers. Turning the tube and watching works too, as a technician turns a tube to
  find its label.
- **The turns and the wrist.** joint6 runs at about the arm's heading minus the grip's yaw. At the racks the left arm
  heads 23 to 52 degrees to its right, so a grip turned +90 degrees would put joint6 at its 120-degree stop. Tube 1 is
  gripped across x and set down across y (a quarter turn anticlockwise). Tube 3 is gripped across y, on its label and
  the opposite face, and set down across x (a quarter turn clockwise). The wrist stays between -49 and +47 degrees.
- **Tube 3 goes last, into the right-hand socket, round the front.** Set down across x, the wrist camera hangs 7 cm to
  the robot's right of the fingers. At the right-hand socket that is over open table beyond the rack's end; anywhere
  else it would be over a loaded socket. Carried straight across from the bench, tube 3 would pass 1.1 cm into the tops
  of the tubes already in the rack. So it goes along x 20 in front of the rack and then across. No carry passes over a
  standing tube.
- **Reach.** Carried 14 cm or more over the table, the left arm couldn't reach the right-hand socket with the rack at x
  30, nor at 17 cm even the middle one (IK misses by 9 to 27 mm). So the rack sits at x 26, and tubes are carried 14 cm
  up, their bottoms 3.6 cm over the rack and 5 cm over the bench collars.
- **The fingers open only part way.** Fully open, they reach 5 cm or more either side of a tube, onto the tube beside it
  in the bench rack (5 cm away) or in the next socket (4.5 cm). So they open to 3.8 cm to come down round a tube (7.75
  mm a side, 7.5 mm clear of the tube beside) and to 2.6 cm to let go (1.75 mm a side, 1.1 cm clear of the next socket's
  tube).
- **The set-down.** The joints trail their targets, the wrist by 0.15 s, and on a carry the TCP runs up to 6 mm off its
  path, more than the fit. So the set-down stops 1.2 cm over the rack, settles for 0.6 s and goes in over 4.4 s, at
  under 2 cm/s. With every joint's lag modelled, each tube's bottom keeps 1.8 mm from the walls across the label and 5.2
  mm along the row as it enters (2.7 mm with the row 2.8 cm wide); a 2.2 s seat would leave 0.4 mm across. The places
  aim 1.25 mm back from each socket's centre, so the label and stripe, 2.5 mm proud, sit centred. Demo-only `TRIM`
  offsets are there for calibration on the GPU and play no part in scoring.
- **Clearances.** The pad tips pass 1.7 cm over the bench collars and 1.4 cm over the rack's walls. Placed from
  `station.wrist_camera`, the wrist camera keeps 5.9 cm from every fixture (closest at the rack's near end, as tube 1 is
  gripped across x) and 4.8 cm from every standing tube. The two arms keep at least 15 cm apart. The left arm ends at y
  26, and the right waits 15 cm to its side of the button.
- **Unintended strategies.** Pressing first: 0.1 at most, and 0 with nothing right at the end. Tubes lying down, upside
  down, leaning more than 10 degrees, held above a socket, standing on the rack's top or left on the bench: nothing. Two
  tubes crowded into one socket count once. A tube knocked over but left on the table just doesn't count, as one left on
  the bench doesn't; only a tube off the table scores 0 for the whole trial.
- **The rule behind it.** CLSI AUTO02-A2, *Laboratory Automation: Bar Codes for Specimen Container Identification*,
  covers bar code label placement and orientation on specimen containers (archived by CLSI but still technically valid).
  Analyzer loading procedures, for example a blood-bank analyzer's sample-loading procedure, have tubes placed in the
  rack "so the barcode label is visible through the slot".
- **What isn't modelled.** No bar code is read: the label is a white box with a black stripe, and facing the slot means
  its direction is within 25 degrees of -x. The tubes are square in section, standing in for round ones, and the label
  is a 2 mm strip, thicker than paper, which is what keys a seated tube's turn. The rack is fixed, with no reader or
  conveyor, and the load button only records its first press; nothing lights.

## Runtime and seeds

- The scripted demo takes about 61 s of simulated time (1,226 steps at 20 Hz): the left arm's plan runs 56 s, and the
  right arm presses load at about 56.8 s. In the offline replay the trial ends at 1.0 about 57.75 s in (1,155 steps).
- Dream-1 runs the full 2,400 steps unless it finishes first (2 minutes of simulated time).
- One fixed layout: `-E object_jitter=0`, so the seed does not change the start.
