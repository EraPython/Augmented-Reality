import cv2
import cv2.aruco as aruco
import numpy as np
import os


def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of markers detected: ", noOfMarkers)
    augDics = {}

    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug

    return augDics


def findArucoMarkers(img, markerSize=4, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.Dictionary_get(getattr(aruco, f"DICT_{markerSize}X{markerSize}_{totalMarkers}"))
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


def augmentAruco(bbox, id, img, imgAug, drawID=True):
    top_left = bbox[0][0][0], bbox[0][0][1]
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_left = bbox[0][2][0], bbox[0][2][1]
    bottom_right = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0, 0], [w, 0], [h, w], [0, h]])
    mask = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)

    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), 255)
    imgOut += img

    # cv2.imshow("wutever", imgOut)
    # results = cv2.bitwise_and(imgOut, imgOut, img, mask=mask)

    # if drawID:
    #     cv2.putText(imgOut, str(id), top_left, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut


def main():
    cap = cv2.VideoCapture(0)
    augDics = loadAugImages("markers")

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox, id, img, augDics[int(id)])

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
