# combined_line_follow.py
# Combines your Sparky motor control (from code 1) with the simple OpenCV line follower (from code 2).
# Behavior:
#  - Detects the largest contour in the camera mask (same mask logic as your second script)
#  - Uses move_func_t(...) to command motors (same mapping as your first script)
#  - Stops motors and cleans up on exit

import time
import cv2
import numpy as np
import sys
import threading

# -------------------------
# Sparky setup (from code 1)
# -------------------------
from sparkybotmini import SparkyBotMini
mr_sparky = SparkyBotMini(port="/dev/ttyUSB0")

def move_func_t(dir, power):
    """Same motor mapping you provided in code 1."""
    try:
        if dir == "forward":
            mr_sparky.set_motor(power, -power, -power, power)
        elif dir == "backward":
            mr_sparky.set_motor(-power, power, power, -power)
        elif dir == "right":
            mr_sparky.set_motor(power, power, power, power)
        elif dir == "left":
            mr_sparky.set_motor(-power, -power, -power, -power)
        elif dir == "t_left":
            mr_sparky.set_motor(-power, -power, power, power)
        elif dir == "t_right":
            mr_sparky.set_motor(power, power, -power, -power)
        elif dir == "for_left":
            mr_sparky.set_motor(0, -power, -power, 0)
        elif dir == "for_right":
            mr_sparky.set_motor(power, 0, 0, power)
        elif dir == "bac_left":
            mr_sparky.set_motor(-power, 0, 0, -power)
        elif dir == "bac_right":
            mr_sparky.set_motor(0, power, power, 0)
        else:
            # unknown command -> stop
            mr_sparky.set_motor(0, 0, 0, 0)
    except Exception as e:
        # If motor command fails, print and attempt safe stop
        print(f"[move_func_t] motor error: {e}")
        try:
            mr_sparky.set_motor(0, 0, 0, 0)
        except Exception:
            pass

# -------------------------
# Camera and detection params (from code 2)
# -------------------------
CAM_INDEX = 0
CAM_W = 160
CAM_H = 120
POWER = 50            # default motor power used for movement commands
CX_RIGHT_THRESH = 120 # same thresholds used in your second script
CX_LEFT_THRESH = 40

# Open camera
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

# Ensure Sparky connected before starting movement
if not mr_sparky.connect():
    print("Problem connecting to MrSparky. Exiting.")
    cap.release()
    sys.exit(1)

print("MrSparky connected. Starting line follower...")

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        # Use same mask logic as your second script
        # Note: original used low_b = [5,5,5] and high_b = [0,0,0] which is inverted;
        # we keep the same values to preserve behavior, but swap order for cv2.inRange
        low_b = np.uint8([5, 5, 5])
        high_b = np.uint8([0, 0, 0])
        # cv2.inRange expects lower <= upper; original code used inverted values.
        # To preserve the original intent (detect dark pixels), compute mask for dark pixels:
        # dark_mask = (frame <= low_b).all across channels -> use inRange with (0,0,0) .. (5,5,5)
        mask = cv2.inRange(frame, np.array([0, 0, 0], dtype=np.uint8), low_b)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Debug print
                print(f"CX: {cx}  CY: {cy}")

                # Map the original three-branch logic to your move_func_t commands
                if cx >= CX_RIGHT_THRESH:
                    # original: "Turn Left"
                    print("Turn Left -> t_left")
                    move_func_t("t_left", POWER)
                elif cx < CX_RIGHT_THRESH and cx > CX_LEFT_THRESH:
                    # original: "On Track!"
                    print("On Track -> forward")
                    move_func_t("forward", POWER)
                elif cx <= CX_LEFT_THRESH:
                    # original: "Turn Right"
                    print("Turn Right -> t_right")
                    move_func_t("t_right", POWER)

                # draw centroid for visualization
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            else:
                # moment zero -> stop
                print("Moment zero - stopping motors")
                move_func_t("forward", 0)
        else:
            # no contours -> stop motors (same as original GPIO off)
            print("I don't see the line -> stopping motors")
            move_func_t("forward", 0)

        # draw contours safely (only if c exists)
        try:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
        except Exception:
            pass

        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit pressed - stopping")
            move_func_t("forward", 0)
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt - stopping motors")
    move_func_t("forward", 0)

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    try:
        mr_sparky.disconnect()
    except Exception:
        pass
    print("Clean exit.")
