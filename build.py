import sys
from cx_Freeze import setup, Executable
base = None    

setup(name = "Facedetection",
      description = "Facedetection system",
      executables = [Executable("FaceDetectionsRun.py",base=base)])
