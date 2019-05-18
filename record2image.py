import cv2 as cv
import os
import time
import progressbar


def crops(img, top_left=(0, 0), bottom_right=(1920, 1080), file_name='image'):
    cv.imwrite(file_name+'.jpg', img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])


if __name__ == "__main__":
    vpath = 'D:\\Videos'
    ipath = 'D:\\OneDrive\\OneDrive - Australian National University\\COMP\\4528\\etsts'
    folder = os.listdir(vpath)
    counter = 0
    for file in folder:
        if not os.path.isdir(os.path.join(vpath, file)):
            ctime = time.time()
            print('Processing file: ' + file)
            capture = cv.VideoCapture(os.path.join(vpath, file))
            frame = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            pbar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
            for index in range(frame):
                if index % 360 == 0:
                    capture.set(cv.CAP_PROP_POS_FRAMES, index)
                    ret, image = capture.read()
                    file_name = str(counter).zfill(6)
                    cv.imwrite(ipath + "/shot/" + file_name+'.jpg', image)
                    crops(image, (0, 0), (1800, 800), ipath+"/windshield/w"+file_name)
                    crops(image, (20, 90), (295, 455), ipath+"/left_mirror/l"+file_name)
                    crops(image, (1895, 90), (1620, 455), ipath+"/right_mirror/r"+file_name)
                    crops(image, (380, 880), (640, 1040), ipath+"/navigation/n"+file_name)
                    crops(image, (40, 715), (430, 950), ipath+"/assistant/a"+file_name)
                    counter = counter+1
                    pbar.update(int((index / (frame - 1)) * 100))
            pbar.finish()

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
        crops(image, (27, 268), (287, 472), 'assistant')
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
        crops(image, (647, 682), (665, 694), 'failure1')
        crops(image, (670, 682), (688, 693), 'failure2')
        crops(image, (693, 682), (712, 693), 'failure3')
        print(time.time()-ctime)
        cv2.waitKey()

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
