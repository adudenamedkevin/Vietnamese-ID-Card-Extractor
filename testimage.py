# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO

# Specify path of input image and YOLO model file
img_path = 'test1.jpg'
model_path = 'best.pt'

# Define a dictionary to map class IDs to class names
class_name_dict = {0: 'CCCD',
                   1: 'no.',
                   2: 'fullname',
                   3: 'dateofbirth',
                   4: 'sex',
                   5: 'nationality',
                   6: 'placeoforigin',
                   7: 'placeofresidence'}

# Load the YOLO model
model = YOLO(model_path)

# Read the input image using OpenCV and display it using matplotlib
img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Specify the minimum confidence score required for a detection to be considered valid
threshold = 0.5

# Use YOLO to detect objects in the input image
results = model(img)[0]
print(results)

# Draw bounding boxes around the detected objects and perform OCR on each ROI
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
reader = easyocr.Reader(['vi'])
i = 0
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Draw a bounding box around the object
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        ocr_img = img[int(y1):int(y2), int(x1):int(x2), :].copy()
        cv2.imwrite('o'+str(i)+'.jpg', ocr_img)
        i = i + 1

        # Perform OCR on the ROI and print the detected text
        output = reader.readtext(ocr_img)
        print(output)
        for out in output:
            text_bbox, text, text_score = out
            print(text, text_score)

# Save the input image with the bounding boxes drawn around the detected objects and any text detected by OCR
cv2.imwrite('output.jpg',img)
