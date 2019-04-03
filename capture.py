from PIL import ImageGrab
import cv2
import numpy as np

while(True):
    capture = ImageGrab.grab(bbox=(0, 40, 1920, 1120))
    image = cv2.cvtColor(np.array(capture), cv2.COLOR_RGB2BGR)
