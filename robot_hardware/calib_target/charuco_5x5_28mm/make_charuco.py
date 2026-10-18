#!/usr/bin/env python3
"""Generate the ChArUco calibration board (5x5 squares, AprilTag 36h11) as a PDF.

Print the PDF at 100 % and measure the square size afterwards. The detection side
(`tools/cam_server.py`) builds the board with `make_board()`, so both use the same
parameters.
"""

import json
import os

import cv2
from PIL import Image, ImageDraw

SQUARES_X, SQUARES_Y = 5, 5
SQUARE_MM, MARKER_MM = 28.0, 20.0
MARGIN_MM = 10.0  # white margin; the board is 160 x 160 mm overall
DICT = cv2.aruco.DICT_APRILTAG_36h11
PX_PER_MM = 20  # 508 dpi; a square is an integer number of pixels
OUT = os.path.dirname(os.path.abspath(__file__))


def make_board():
    """The board object used for generation and detection."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM / 1000, MARKER_MM / 1000, d)


if __name__ == "__main__":
    b = make_board()
    W = int((SQUARES_X * SQUARE_MM + 2 * MARGIN_MM) * PX_PER_MM)
    H = int((SQUARES_Y * SQUARE_MM + 2 * MARGIN_MM) * PX_PER_MM)
    img = b.generateImage((W, H), marginSize=int(MARGIN_MM * PX_PER_MM), borderBits=1)
    cv2.imwrite(f"{OUT}/charuco_5x5_28mm_board.png", img)

    # A4 page: the board centred, with cut marks and a 100 mm ruler to check the print scale.
    A4W, A4H = 210 * PX_PER_MM, 297 * PX_PER_MM
    page = Image.new("L", (A4W, A4H), 255)
    ox, oy = (A4W - W) // 2, 30 * PX_PER_MM
    page.paste(Image.fromarray(img), (ox, oy))
    dr = ImageDraw.Draw(page)

    # Dashed cut line along the outer edge of the margin.
    def dashed(x0, y0, x1, y1, dash=3 * PX_PER_MM, gap=2 * PX_PER_MM, width=6, fill=110):
        L = max(abs(x1 - x0), abs(y1 - y0))
        n = 0
        while n < L:
            a, b = n, min(n + dash, L)
            if x0 == x1:
                dr.line([x0, y0 + a, x1, y0 + b], fill=fill, width=width)
            else:
                dr.line([x0 + a, y0, x0 + b, y1], fill=fill, width=width)
            n += dash + gap

    x1, y1 = ox + W, oy + H
    dashed(ox, oy, x1, oy)
    dashed(ox, y1, x1, y1)
    dashed(ox, oy, ox, y1)
    dashed(x1, oy, x1, y1)
    # Cut marks outside the corners.
    g, ln = 2 * PX_PER_MM, 8 * PX_PER_MM
    for cx, sx in ((ox, -1), (x1, 1)):
        for cy, sy in ((oy, -1), (y1, 1)):
            dr.line([cx + sx * g, cy, cx + sx * (g + ln), cy], fill=0, width=4)
            dr.line([cx, cy + sy * g, cx, cy + sy * (g + ln)], fill=0, width=4)
    y = oy + H + 25 * PX_PER_MM
    x0 = (A4W - 100 * PX_PER_MM) // 2
    dr.line([x0, y, x0 + 100 * PX_PER_MM, y], fill=0, width=6)
    for k in range(11):
        xx = x0 + k * 10 * PX_PER_MM
        h = 5 * PX_PER_MM if k % 5 == 0 else 3 * PX_PER_MM
        dr.line([xx, y - h, xx, y], fill=0, width=4)
    page.save(f"{OUT}/charuco_5x5_28mm_A4.pdf", resolution=25.4 * PX_PER_MM)

    json.dump(
        {
            "type": "charuco",
            "dictionary": "DICT_APRILTAG_36h11",
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": SQUARE_MM / 1000,
            "marker_length_m": MARKER_MM / 1000,
            "board_outer_mm": [W / PX_PER_MM, H / PX_PER_MM],
            "legacy_pattern": False,
            "note": "Replace square_length_m with the value measured after printing.",
        },
        open(f"{OUT}/charuco_5x5_28mm.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    print("board px", W, H)
