import cv2
import numpy as np
import time
import multiprocessing as mp
from mss import mss


def crops(img, top_left=(0, 0), bottom_right=(1920, 1080), file_name='image'):
    cv2.imwrite('images/'+file_name+'.jpg', img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])


if __name__ == "__main__":

    cv2.namedWindow('image')
    while(True):
        ctime = time.time()
        image = np.array(mss().grab({"top": 40, "left": 0, "width": 1280, "height": 720}))
        cv2.imshow('image', image)
        cv2.imwrite('image.jpg', image)
        crops(image, (0, 0), (1200, 490), 'windshield')
        crops(image, (16, 58), (193, 296), 'left_mirror')
        crops(image, (1082, 58), (1259, 296), 'right_mirror')
        crops(image, (246, 578), (424, 683), 'navigation')
        crops(image, (27, 472), (287, 268), 'assistant')
        crops(image, (626, 653), (680, 662), 'odometer')
        crops(image, (1360, 920), (1490, 950), 'fuel_gauge')
        crops(image, (735, 565), (756, 584), 'left_turn')
        crops(image, (863, 565), (883, 584), 'right_turn')
        crops(image, (901, 565), (924, 584), 'parking_break')
        crops(image, (927, 565), (943, 585), 'seat_belt')
        crops(image, (1108, 567), (1120, 575), 'light0')
        crops(image, (1123, 569), (1135, 577), 'light1')
        crops(image, (1138, 570), (1150, 578), 'light2')
        crops(image, (1154, 540), (1758, 878), 'light3')
        crops(image, (1174, 572), (1179, 578), 'light4')
        crops(image, (704, 579), (718, 583), 'light5')
        crops(image, (969, 578), (982, 581), 'battery_charge')
        crops(image, (907, 679), (929, 694), 'malfunction_indicator')
        crops(image, (933, 680), (951, 694), 'glow_plug')
        crops(image, (624, 684), (641, 693), 'failure0')
        crops(image, (647, 682), (656, 694), 'failure1')
        crops(image, (670, 682), (688, 693), 'failure2')
        crops(image, (693, 682), (712, 693), 'failure3')
        print(time.time()-ctime)
        cv2.waitKey()
