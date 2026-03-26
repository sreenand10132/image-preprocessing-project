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
