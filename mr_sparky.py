import time
from sparkybotmini import SparkyBotMini

mr_sparky = SparkyBotMini(port = "/dev/ttyUSB0")

def main_function():
  mr_sparky.set_motor(40, -40, -40, 40)

if mr_sparky.connect():
  print("MrSparky has been connected")
  main_function()
else:
  print("problem connecting to MrSparky")
  exit()

mr_sparky.disconnect()
