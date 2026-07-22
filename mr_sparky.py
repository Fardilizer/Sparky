import time
from sparkybotmini import SparkyBotMini

mr_sparky = SparkyBotMini(port = "/dev/ttyUSB0")

def move_function(dir, power, time):
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

def main_function():
  mr_sparky.set_motor(-40, 40, 40, -40)

if mr_sparky.connect():
  print("MrSparky has been connected")
  main_function()
else:
  print("problem connecting to MrSparky")
  exit()

mr_sparky.disconnect()
