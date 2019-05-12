import cv2
import numpy as np
from mss import mss
from objects import Image


def capture():
    return np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720}))


while(True):
    image = Image.Image(capture())
    cv2.imshow('image', image.windshield)
    cv2.imwrite('images/image.jpg', image.image)
    cv2.imwrite('images/windshield.jpg', image.windshield)
    cv2.imwrite('images/left_mirror.jpg', image.left_mirror)
    cv2.imwrite('images/right_mirror.jpg', image.right_mirror)
    cv2.imwrite('images/navigation.jpg', image.navigation)
    cv2.imwrite('images/assistant.jpg', image.assistant)
    cv2.imwrite('images/odometer.jpg', image.odometer)
    cv2.imwrite('images/fuel_gauge.jpg', image.fuel_gauge)
    cv2.imwrite('images/left_turn.jpg', image.left_turn)
    cv2.imwrite('images/right_turn.jpg', image.right_turn)
    cv2.imwrite('images/parking_break.jpg', image.parking_break)
    cv2.imwrite('images/seat_belt.jpg', image.seat_belt)
    cv2.imwrite('images/battery_charge.jpg', image.battery_charge)
    cv2.imwrite('images/malfunction_indicator.jpg', image.malfunction_indicator)
    cv2.imwrite('images/glow_plug.jpg', image.glow_plug)
    cv2.imwrite('images/light0.jpg', image.light0)
    cv2.imwrite('images/light1.jpg', image.light1)
    cv2.imwrite('images/light2.jpg', image.light2)
    cv2.imwrite('images/light3.jpg', image.light3)
    cv2.imwrite('images/light4.jpg', image.light4)
    cv2.imwrite('images/light5.jpg', image.light5)
    cv2.imwrite('images/failure0.jpg', image.failure0)
    cv2.imwrite('images/failure1.jpg', image.failure1)
    cv2.imwrite('images/failure2.jpg', image.failure2)
    cv2.imwrite('images/failure3.jpg', image.failure3)
    cv2.waitKey(0)

