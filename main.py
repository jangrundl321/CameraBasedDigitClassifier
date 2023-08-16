import cv2
import helpers
from Detection import handTracker

STOP = False

LAST_POS = (0, 0)
NEW_POS = (0, 0)
cap = cv2.VideoCapture(0)
tracker = handTracker()
DRAWING = False
list_of_circles = []
First = True
success, image = cap.read()
list = []

while True:
    success, image = cap.read()
    imageRGB = tracker.handsFinder(image)

    c = cv2.rectangle(image, (50, 50), (600, 500), (0, 0, 0), -1)

    if not STOP:
        lmlist = tracker.positionFinder(image)

        for x in lmlist:
            if x[0] == 8:
                if 50 < x[1] < 600 and 50 < x[2] < 500:
                    DRAWING = True
                    NEW_POS = (x[1], x[2])
                    if First:
                        LAST_POS = NEW_POS
                        First = False
                else:
                    DRAWING = False

        if DRAWING:
            list.append((LAST_POS, NEW_POS))
            cv2.line(c, LAST_POS, NEW_POS, (255, 255, 255), 2)
            LAST_POS = NEW_POS

    for x in list:
        cv2.line(c, x[0], x[1], (255, 255, 255), 2)

    if tracker.CheckIfTwoHands():
        STOP = True
    else:
        if STOP:
            STOP = False
            list = []

    if STOP:
        cropped_img = image[50:600, 50:500]
        cv2.imwrite("out.png", cropped_img)
        p = helpers.get_image_prediction("out.png")
        cv2.putText(image, 'CLASSIFIED AS: ' + str(p), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Output", image)
    cv2.waitKey(1)

