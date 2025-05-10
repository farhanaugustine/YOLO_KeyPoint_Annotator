[![DOI](https://zenodo.org/badge/963345981.svg)](https://doi.org/10.5281/zenodo.15379964)

# YOLOv8 Keypoint Annotation Tools

This repository contains two Python scripts designed to facilitate the creation of keypoint and bounding box annotations for training YOLOv8 pose estimation models for Animal Behavior Video Analysis.

**While these tools are created for annotation of animal body-parts, script can be used to annotate other objects as well.**

1.  **`Final_YOLO_Annotator_KeyPoint(SingleAnimal).py` :** Annotates exactly **one object/animal instance** per image.
2.  **`Final_YOLO_annotator_keypoint(multiAnimal).py` :** Annotates **multiple object/animals instances** per image, supports **multiple classes**, and allows **toggling keypoint visibility**.

Both scripts use OpenCV for the graphical interface and Tkinter for simple dialog boxes. They output annotations in the standard YOLOv8 (also compatable with v11) format (`.txt` files) and can generate a basic `data_config.yaml` file to help set up your dataset for training.

## Features

**Common Features:**

* Interactive GUI for placing keypoints and drawing bounding boxes.
* User-defined keypoint sequences.
* Outputs standard YOLO format annotation files (`.txt`), including normalized coordinates and visibility flags.
* Generates a template `data_config.yaml` file for YOLOv8 (also tested with YOLOv11) training.
* Basic controls for navigating images (next/previous), undoing actions, resetting, saving, and quitting.

**`Final_YOLO_Annotator_KeyPoint(SingleAnimal).py` Specifics:**

* Designed for datasets where each image contains only **one** object of interest.
* Annotates a single class per session.

**`Final_YOLO_annotator_keypoint(multiAnimal).py` Specifics:**

* Annotates **multiple object instances** within the same image.
* Supports defining and annotating **multiple distinct classes** (e.g., 'cat', 'dog') in the same session.
* Allows toggling the visibility flag (**Visible** / **Not Visible**) of the most recently placed keypoint using the **Middle Mouse Button**.
* Class selection via number keys (`1`-`9`, `0`) for each new instance within an image.

## Dependencies

* **Tested with Python 3.11**
* **OpenCV for Python:** `opencv-python`
* **NumPy:** `numpy`
* **PyYAML:** `PyYAML` (for generating the config file)
* **Tkinter:** Usually included with standard Python installations. If not, you may need to install it separately (e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu).
* **Use pip install:** ``pip install opencv-python numpy PyYAML``

## Usage

### Running the Scripts

Execute the desired script from your terminal:

```bash
# For the single-object annotator
python Final_YOLO_Annotator_KeyPoint(SingleAnimal).py

# For the multi-object annotator
python Final_YOLO_annotator_keypoint(multiAnimal).py
```
## Initial Setup (Dialog Boxes)
Both scripts will prompt you with dialog boxes upon starting:

* Keypoint Sequence: Enter the names of your object's keypoints, separated by commas, in the exact order you want to annotate them (e.g., nose,left_eye,right_eye,left_ear,right_ear,...). This order is critical and will be reflected in the output files and the data_config.yaml.
* Class Definition:
`Final_YOLO_Annotator_KeyPoint(SingleAnimal).py`: Asks for a single Class ID (integer, usually starting from 0) and the corresponding Class Name (e.g., person).
`Final_YOLO_annotator_keypoint(multiAnimal).py`: Enters a loop asking you to define classes one by one. Enter the Class Name for ID 0, then ID 1, and so on. Leave the name blank or press Cancel when you are finished defining all your classes.
* Image Directory: Select the folder containing the images you want to annotate.
* Output Directory: Select the folder where the generated .txt annotation files should be saved. It's recommended to choose an empty or dedicated labels directory.

## Annotation Workflow
### Final_YOLO_Annotator_KeyPoint(SingleAnimal).py:

1. The first image loads.
2. The status bar shows which keypoint to click (Click Keypoint 1/N: keypoint_name).
3. Left-click on the image to place each keypoint in sequence.
4. After placing all keypoints, the mode switches to bounding box annotation.
5. Left-click for the Top-Left corner of the bounding box.
6. Left-click for the Bottom-Right corner of the bounding box.
7. The annotation is marked as DONE. Use keybindings (n, p, s, q) to proceed.

### Final_YOLO_annotator_keypoint(multiAnimal).py:

1. The first image loads. The mode is SELECT_CLASS.
2. The status bar shows available classes and their corresponding selection keys ((1) cat | (2) dog ...).
3. Press the number key (1-9, 0) corresponding to the class of the first object you want to annotate.
4. The mode switches to keypoints. The status bar shows the selected class and prompts for the first keypoint.
5. Left-click to place each keypoint in sequence for the current object instance.
6. (Optional) Middle-click immediately after placing a keypoint to toggle its visibility between Visible (Green, flag 2) and Not Visible (Red, flag 1).
7. After placing all keypoints for the instance, the mode switches to bounding box annotation.
8. Left-click for the Top-Left corner.
9. Left-click for the Bottom-Right corner.
10. The annotation for this instance is completed and stored. The display updates to show it (e.g., magenta box).
11. The mode automatically resets to SELECT_CLASS. You can now:
12. Select another class (using 1-9, 0) to annotate another object in the same image.
13. Use navigation keys (n, p) to save all annotations for the current image and move to the next/previous one.
14. Use s to save annotations explicitly.
15. Use r to clear all annotations for the image and start over.
16. Generating the Config File
17. After you quit the annotation session (using 'q'), both scripts will ask if you want to generate the data_config.yaml file. 18. If you choose yes, it will prompt you to select a save location and filename. This YAML file contains the paths, number of classes, class names, and keypoint shape information needed by YOLOv8 for training. Important: Review the generated YAML file and manually create the images/train, images/val, labels/train, labels/val directory structure as described in its comments. 19. You will need to split your images and corresponding .txt label files into these train/validation sets yourself.

## Keybindings
### Final_YOLO_Annotator_KeyPoint(SingleAnimal).py Keybindings
| Key | Action                                                                  | Mode Restriction      |
| :-- | :---------------------------------------------------------------------- | :-------------------- |
| n | Move to Next image (prompts to save/discard if annotation started)  | Any                   |
| p | Move to Previous image (prompts to save/discard if annotation started)| Any                   |
| u | Undo the last placed keypoint                                       | keypoints mode only |
| r | Reset all annotations for the current image (asks confirmation)     | Any                   |
| s | Save the completed annotation for the current image                 | DONE mode only      |
| q | Quit the application (asks confirmation)                            | Any                   |
| LMB | Left Mouse Button: Place current keypoint / Set BBox corners        | Specific modes        |

### Final_YOLO_annotator_keypoint(multiAnimal).py Keybindings
|Key|	Action|	Mode Restriction|
|:---|:----|:----|
|1 through 9	|Select Class for the next object instance (maps to class IDs 0-8)|	SELECT_CLASS mode only|
0|	Select Class for the next object instance (maps to class ID 9, if defined)|	SELECT_CLASS mode only|
n|	Save all annotations for current image & move to Next image|	Any|
p|	Save all annotations for current image & move to Previous image|	Any|
u|	Undo the last placed keypoint (for the current instance being annotated)|	keypoints mode only|
r|	Reset ALL annotations for the current image (clears all instances, asks confirmation)|	Any|
s	|Save all annotations for the current image explicitly	|Any|
q	|Quit the application (asks confirmation)|	Any|
LMB	|Left Mouse Button: Place current keypoint / Set BBox corners|	Specific modes| 
MMB|	Middle Mouse Button: Toggle visibility (Visible/Not Visible) of last placed keypoint|	keypoints mode only|

## Remember to:

Verify the path is correct relative to where you will run your YOLOv8 training command.
Manually create the directory structure (images/train, images/val, labels/train, labels/val) inside your dataset directory.
Split your image files and the corresponding .txt annotation files into the train and val subdirectories.

