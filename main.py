import cv2
import numpy as np
import imutils
import easyocr


def detect_number_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction + edge detection
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)

    # Find contours
    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for c in contours:
        approx = cv2.approxPolyDP(c, 10, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        return None, img

    # Mask and crop plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)

    x, y = np.where(mask == 255)
    x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
    cropped = gray[x1:x2 + 1, y1:y2 + 1]

    # OCR
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cropped)

    if not result:
        return None, img

    text = result[0][-2]

    # Draw rectangle
    cv2.rectangle(
        img,
        tuple(plate_contour[0][0]),
        tuple(plate_contour[2][0]),
        (0, 0, 0),
        3
    )

    # Put detected text on right corner
    h, w, _ = img.shape
    cv2.putText(
        img,
        text,
        (w - 300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    return text, img


if __name__ == "__main__":
    image_path = r"C:\Users\r6190\Downloads\image2.jpg"
    plate_number, output = detect_number_plate(image_path)

    print("Detected Number Plate:", plate_number)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
