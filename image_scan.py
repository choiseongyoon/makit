import numpy as np
import cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def auto_scan_image():
    image = cv2.imread('canny.jpg')
    orig = image.copy()

    r = 800.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 75, 200)

    print("Step 1 : edge detection")

    cv2.imwrite('Image.png', image)
    cv2.imwrite('Edged.png', edged)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    print("STEP 2: Find Contours of Paper")

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imwrite('Outline.png', image)

    rect = order_points(screenCnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomLeft[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])

    dst = np.float32([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]])

    N = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(orig, N, (maxWidth, maxHeight))

    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    cv2.imwrite('warped.png', warped)

    # STEP4: Apply Adaptive Threshold
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # ovicolor ()E Saf sere emre nipa

    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    print("STEP 4: Apply Adaptive Threshold")
    cv2.imwrite('Original.png', orig)
    cv2.imwrite('Scanned.png', warped)
    cv2.imwrite('scannedimage.png', warped)


if __name__ == "__main__":
    auto_scan_image()