import time
from sparkybotmini import SparkyBotMini

mr_sparky = SparkyBotMini(port = "/dev/ttyUSB0")

if mr_sparky.connect():
  print("MrSparky has been connected")
else:
  print("problem connecting to MrSparky")
  exit()

mr_sparky.disconnect()
