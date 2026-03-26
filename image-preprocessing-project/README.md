# Image Preprocessing for Image Classification

## 1. Python Code (OpenCV Pipeline)

```python
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('data/sample.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
resized = cv2.resize(img_rgb, (256, 256))

# Grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

# Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

# Edge Detection
edges = cv2.Canny(blur, 100, 200)

# Normalization
normalized = resized / 255.0

# Show steps
titles = ['Original', 'Resized', 'Gray', 'Blur', 'Threshold', 'Edges']
images = [img_rgb, resized, gray, blur, thresh, edges]

plt.figure(figsize=(10,6))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()
```

---

## 2. GitHub Repository Structure

```
image-preprocessing-project/
│── data/
│   └── sample.jpg
│── src/
│   └── preprocessing.py
│── outputs/
│   └── processed_images/
│── requirements.txt
│── README.md
│── report.md
```

---

## 3. README.md

````markdown
# Image Preprocessing for Image Classification

## Overview
This project demonstrates common image preprocessing techniques using OpenCV.

## Steps Performed
- Resize
- Grayscale conversion
- Noise removal (Gaussian Blur)
- Thresholding
- Edge Detection
- Normalization

## Installation
```bash
pip install -r requirements.txt
````

## Run the Project

```bash
python src/preprocessing.py
```

## Output

Displays intermediate preprocessing steps.

```

---

## 4. Project Report

### Introduction
Image preprocessing is a crucial step in image classification. Raw images often contain noise, irrelevant information, and inconsistent dimensions that can negatively impact model performance.

### Why Preprocessing Matters
- Improves model accuracy
- Reduces noise
- Standardizes input size
- Highlights important features

### Approach
We applied a sequence of transformations:
1. Resize images to uniform dimensions
2. Convert to grayscale to reduce complexity
3. Apply Gaussian blur to remove noise
4. Use thresholding to segment image
5. Detect edges for feature extraction
6. Normalize pixel values for model compatibility

### Key Decisions
- Used OpenCV for efficiency
- Selected Gaussian blur for noise reduction
- Used Canny edge detector for strong edge extraction

### Challenges
- Choosing optimal threshold values
- Balancing noise reduction vs detail preservation
- Handling different image sizes

### Learnings
- Preprocessing significantly affects classification performance
- Each step must be carefully tuned
- Visualization helps understand transformations

### Conclusion
Effective preprocessing leads to better feature extraction and improved classification results.
```
