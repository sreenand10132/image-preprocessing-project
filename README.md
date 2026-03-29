# Image Preprocessing Understanding
## The Problem
There is a misunderstanding of how ML models view images
based on the task at hand. The required steps also arent easily
visualized. 
## The Solution
We utilize python OpenCV to show the preprocessing
steps based on the task and dataset
## Usage
Step 1
Clone Repository on your machine
Step 2
Install requirements.txt
```python
pip install requirements.txt
```
Step 3
Run the code with the corresponding flags
```python
python src/main.py --mode [MODE] --input /path/to/dataset.txt
```
MODES : classification, segmentation, edges
