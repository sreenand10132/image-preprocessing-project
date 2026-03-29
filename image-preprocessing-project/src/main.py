import cv2
import os
import argparse

def preprocess_classification(img):
    steps = {}
    resized = cv2.resize(img, (224, 224))
    steps['Resized'] = resized
    normalized = resized / 255.0
    steps['Normalized'] = normalized
    return steps


def preprocess_segmentation(img):
    steps = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps['Grayscale'] = gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    steps['Blur'] = blur
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    steps['Threshold'] = thresh
    return steps


def preprocess_edge_detection(img):
    steps = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps['Grayscale'] = gray
    edges = cv2.Canny(gray, 100, 200)
    steps['Edges'] = edges
    return steps


def show_steps(original, steps):
    cv2.imshow('Original', original)
    for name, img in steps.items():
        if len(img.shape) == 2:
            cv2.imshow(name, img)
        else:
            cv2.imshow(name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(img_path, mode):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return

    if mode == 'classification':
        steps = preprocess_classification(img)

    elif mode == 'segmentation':
        steps = preprocess_segmentation(img)

    elif mode == 'edges':
        steps = preprocess_edge_detection(img)

    else:
        print("Invalid mode selected")
        return

    show_steps(img, steps)


def process_directory(folder, mode):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing: {file}")
            process_image(path, mode)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['classification', 'segmentation', 'edges'],
                        help='Select application type')

    parser.add_argument('--input', required=True,
                        help='Path to image or directory')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_directory(args.input, args.mode)
    else:
        process_image(args.input, args.mode)
