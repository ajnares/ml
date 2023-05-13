# computer vision
import cv2
import os

# read image
# img = cv2.imread('Me.jpg')

# camera

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'coco.names'

# open file
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# set configuration
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

# detection model
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # detect img from camera
    check, img = cap.read()

    # detect object
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds)


    for classID, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
        cv2.putText(img, classNames[classID - 1], (box[0] + 10, box[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # show image
        cv2.imshow("Output", img)
        # delay
        cv2.waitKey(1)
        # play audio from other file
        os.system('python sound.py')
        # cv2.putText(img, confidence, (box[0] + 10, box[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 255, 0), 2)



cap.release()
cv2.destroyAllWindows()
