import cv2
import pytesseract
#set tesseract exe file
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def return_text(image = None, x1= None, y1 = None, x2=None, y2 = None):
    x1= int(x1)
    y1= int(y1)
    x2= int(x2)
    y2= int(y2)
    #extract region of interest
    roi = image[y1:y2, x1:x2]
    #convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #otsu's thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #get text
    text = pytesseract.image_to_string(gray, lang='eng')
    return text