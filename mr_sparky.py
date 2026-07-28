import time
import sys
import tty
import termios
from sparkybotmini import SparkyBotMini
import threading

mr_sparky = SparkyBotMini(port = "/dev/ttyUSB0")

from ultralytics import YOLO
model = YOLO("/home/pi/Downloads/best_V14.onnx")

def main(args):
    return 0
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 90)

def move_func_t(dir, power):
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
    
def _getch():
    """Read a single character from stdin without blocking for Enter (Unix only)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def OD_cam_loop():
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.flip(frame, 0)   # vertical flip
    frame = cv2.flip(frame, 1)   # vertical flip
              
    results = model.predict(
      source = frame,
      imgsz = 320,
      conf = 0.40
    )
              
    #cv2.imshow('Original', frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Results', annotated_frame)
    if cv2.waitKey(1) == ord('q'):
      break
  cap.release()
  v2.destroyAllWindows()

def LD_cam_loop():
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.flip(frame, 0)   # vertical flip
    frame = cv2.flip(frame, 1)   # vertical flip
              
    cv2.imshow('Original', frame)
    if cv2.waitKey(1) == ord('q'):
      break
  cap.release()
  v2.destroyAllWindows()

def control_with_keyboard(power: int = 75):
    """Control mr_sparky using keyboard WASD keys.

    Controls:
      - W: forward
      - S: backward
      - A: strafe/turn left
      - D: strafe/turn right
      - Space: stop motors
      - Q: quit control loop

    This function reads single-key input from the terminal (Unix-only).
    """
    print("Keyboard control active. Use W/A/S/D to move, Space to stop, Q to quit.")
    try:
        key_pressed = ''
        while True:
            ch = _getch().lower()
            if ch == 'w':
                if key_pressed == 'e':
                    move_func_t("for_right", 50)
                elif key_pressed == 'q':
                    move_func_t("for_left", 50)
                else:
                    move_func_t("forward", 50)
                print("w forward")
            elif ch == 's':
                if key_pressed == 'e':
                    move_func_t("bac_right", 50)
                elif key_pressed == 'q':
                    move_func_t("bac_left", 50)
                else:
                    move_func_t("backward", 50)
                print("s backward")
            elif ch == 'a':
                move_func_t("t_left", 50)
                print("a left")
            elif ch == 'd':
                move_func_t("t_right", 50)
                print("d right")
            elif ch == 'q':
                if key_pressed == 'w':
                    move_func_t("for_left", 50)
                elif key_pressed == 's':
                    move_func_t("bac_left", 50)
                else:
                    move_func_t("left", 50)
                print("q left")
            elif ch == 'e':
                if key_pressed == 'w':
                    move_func_t("for_right", 50)
                elif key_pressed == 's':
                    move_func_t("bac_rightq", 50)
                else:
                    move_func_t("right", 50)
                print("e right")
            elif ch == 'f':
                mr_sparky.set_motor(0, 0, 0, 0)
            elif ch == 'r':
                mr_sparky.beep(50)
                print("beep")
            else:
                # ignore other keys
                continue
            key_pressed = ch
    except KeyboardInterrupt:
        print("KeyboardInterrupt - stopping motors")
    finally:
        mr_sparky.set_motor(0, 0, 0, 0)


if mr_sparky.connect():
  print("MrSparky has been connected")
  # Start interactive keyboard control (replace main_function)
  # t1 = threading.Thread(target=OD_cam_loop)
  t2 = threading.Thread(target=LD_cam_loop)
  # t1.start()
  t2.start()
  control_with_keyboard()
  

  if __name__ == '__main__':
      import sys
      sys.exit(main(sys.argv))
else:
  print("problem connecting to MrSparky")
  exit()

mr_sparky.disconnect()
