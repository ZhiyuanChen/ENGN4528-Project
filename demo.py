import cv2 as cv
import numpy as np

demo = cv.imread('images/AR000002.png')

demo_enhanced = demo.copy()
#  道路
cv.rectangle(demo_enhanced, (600, 400), (2600, 1300), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Road', (600, 400), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  左后视镜
cv.rectangle(demo_enhanced, (40, 180), (595, 910), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Left Mirror', (40, 180), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  右后视镜
cv.rectangle(demo_enhanced, (3240, 180), (3795, 910), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Right Mirror', (3240, 180), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  导航
cv.rectangle(demo_enhanced, (925, 1435), (1240, 1600), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Navigation', (925, 1435), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  限速
cv.rectangle(demo_enhanced, (935, 1510), (995, 1570), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Speed Limitation', (935, 1510), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  里程表
cv.rectangle(demo_enhanced, (1600, 1555), (1760, 1585), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Odometer', (1600, 1555), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  燃油表
cv.rectangle(demo_enhanced, (2100, 1480), (2255, 1510), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Fuel gauge', (2100, 1480), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  左转向灯
cv.rectangle(demo_enhanced, (1780, 1400), (1820, 1440), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Left turn signal', (1780, 1400), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  右转向灯
cv.rectangle(demo_enhanced, (2040, 1400), (2080, 1440), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Right turn signal', (2040, 1400), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
#  指示灯
cv.rectangle(demo_enhanced, (1600, 1590), (2260, 1630), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Indication signal', (1600, 1670), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

#  载具
cv.rectangle(demo_enhanced, (1620, 820), (1760, 980), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Vehicle', (1620, 820), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
cv.rectangle(demo_enhanced, (1760, 870), (1840, 930), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Vehicle', (1760, 870), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

#  标识
cv.rectangle(demo_enhanced, (2020, 850), (2060, 980), (18, 0, 139), 2)
cv.putText(demo_enhanced, 'Traffic sign', (2020, 850), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

#  道路
cv.line(demo_enhanced, (1650, 1270), (1785, 950), (18, 0, 139), 2)
cv.line(demo_enhanced, (2390, 1300), (1950, 950), (18, 0, 139), 2)

demo_road = demo.copy()[400:1300, 600:2600]

cv.imwrite('images/AR000002_enhanced.png', demo_enhanced)

