import time
from sparkybotmini import SparkyBotMini

mr_sparky = SparkyBotMini(port = "/dev/ttyUSB0")

def move_func_t(dir, power, time):
  if dir == "forward":
    mr_sparky.set_motor(power, power, power, power)
  elif dir == "backward":
    mr_sparky.set_motor(-power, -power, -power, -power)
  elif dir == "right":
    mr_sparky.set_motor(power, -power, power, -power)
  elif dir == "left":
    mr_sparky.set_motor(-power, power, -power, power)
  elif dir == "t_left":
    mr_sparky.set_motor(power, power -power, -power)
  elif dir == "t_right":
    mr_sparky.set_motor(-power, -power, power, power)
    
  time.sleep(time)
  
  mr_sparky.set_motor(0, 0, 0, 0)

# AI function
def drive_forward_5_seconds(power: int = 75):
    """
    Drive the robot forward for 5 seconds using Mecanum wheels.
    
    For Mecanum wheels in forward motion, all motors rotate in the same direction
    with the same power level. This leverages your drivetrain specs:
    - 4x 12V 550 RPM motors
    - 2" diameter Mecanum wheels
    - 5" x 8" wheelbase
    
    Args:
        power: Motor power level (0-100). Default 75 for balanced speed/control.
               Recommended range: 60-85 for stable forward motion.
    """
    print(f"? Driving forward at {power}% power for 5 seconds...")
    mr_sparky.set_motor(power, power, power, power)
    time.sleep(5)
    mr_sparky.set_motor(0, 0, 0, 0)
    print("? Forward motion complete")

def main_function():
  drive_forward_5_seconds()

if mr_sparky.connect():
  print("MrSparky has been connected")
  main_function()
else:
  print("problem connecting to MrSparky")
  exit()

mr_sparky.disconnect()
