# ChArUco calibration board, 5 x 5, 28 mm, AprilTag 36h11

Used for the hand-eye calibration of the head camera. The board is fixed on a rigid plate
that is mounted at the end of the arm.

| File | Purpose |
|---|---|
| `charuco_5x5_28mm_A4.pdf` | Print this one (A4, includes a 100 mm check ruler) |
| `charuco_5x5_28mm_board.png` | Board image, 3200 x 3200 px, 20 px/mm |
| `charuco_5x5_28mm.json` | Board parameters; set `square_length_m` to the value measured after printing |
| `make_charuco.py` | Generator. The detector imports `make_board()` from it, so both sides use the same parameters |

Specification: ChArUco 5 x 5, 28 mm squares, 20 mm markers, `DICT_APRILTAG_36h11`, marker IDs 0-11,
10 mm white margin, 160 x 160 mm overall, 16 ChArUco corners.

## Printing and use

1. Print at "actual size / 100 %", not "fit to page".
2. Measure the 100 mm check ruler with a caliper to confirm the print scale.
3. Measure the real square size and write it to `square_length_m` in the json.
   `handeye_joint_solve.py --scale` can also estimate the square size from the data.
4. Use matte paper on a rigid flat plate.
5. When collecting samples, tilt the board clearly relative to the camera (more than 20-30 degrees);
   a board facing the camera head-on gives an ambiguous planar pose.
