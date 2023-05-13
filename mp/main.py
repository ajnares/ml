# Computer Vision
import cv2
import numpy as np

# Automation Image
img2 = cv2.imread('automation2.jpg')
height, width, c = img2.shape

i = 0

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 720)
cap.set(4, 720)

classNames = []
classFile = 'coco.names'

# Open File
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

# Set Configuration
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

# Detection Model
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:

    # Size of the automation
    i += 5
    l = img2[:, :(i % width)]
    r = img2[:, (i % width):]
    img2 = np.hstack((r, l))

    # Detect img from camera
    check, img = cap.read()

    # Detect Object
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds)

    for classID, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        # Confidence Level Computation
        confidence = "  {0}%".format(round(confidence * 100))
        if (classID == 1):
            # Detection Box
            cv2.rectangle(img, box, color=(0, 255, 255), thickness=2)

            # Detected Person
            cv2.putText(img, classNames[classID - 1].upper(),
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,
                        0.70, (0, 255, 255), 2)
            # Confidence Level
            cv2.putText(img, str(confidence),
                        (box[0] + 90, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, (0, 255, 255), 2)

            # Count Number of Person
            countText = "# of Person/s on Screen: ", len(bbox)
            cv2.putText(img, str(countText),
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, (0, 255, 255), 2)

            # Automation if number of person detected is 5
            if len(classIds) >= 5:
                # Image Automation
                cv2.imshow('More than 5 Persons Detected', img2)
                print(len(classIds))

        else:
            break

    cv2.imshow("Output", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()