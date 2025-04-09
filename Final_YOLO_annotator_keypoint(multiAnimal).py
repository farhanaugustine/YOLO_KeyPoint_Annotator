# -*- coding: utf-8 -*-

import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import sys
import yaml # Import PyYAML
from collections import OrderedDict

# --- Configuration ---
# Colors (BGR)
KP_VISIBLE_COLOR = (0, 255, 0)    # Green for visible keypoints
KP_INVISIBLE_COLOR = (0, 0, 255)  # Red for invisible keypoints
BBOX_CURRENT_TEMP_COLOR = (255, 255, 0) # Cyan for drawing current bbox
BBOX_CURRENT_FINAL_COLOR = (0, 255, 255) # Yellow for final current bbox
BBOX_COMPLETED_COLOR = (255, 0, 255)    # Magenta for completed bboxes
KP_COMPLETED_COLOR = (255, 165, 0)   # Orange for completed keypoints (less emphasis)
TEXT_COLOR = (255, 0, 0)        # Blue for instructions/status
CLASS_TEXT_COLOR = (0, 0, 0)        # Black for class text on boxes
BACKGROUND_COLOR = (255, 255, 255)  # White background for text boxes
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
COMPLETED_LINE_THICKNESS = 1
CURRENT_LINE_THICKNESS = 2

# Annotation Modes
MODE_SELECT_CLASS = "select_class"
MODE_KP = "keypoints"
MODE_BBOX_START = "bbox_start"
MODE_BBOX_END = "bbox_end"
# MODE_DONE is implicit when no longer annotating a specific instance

# Visibility flags (YOLO standard)
VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_NOT_VISIBLE = 1
VISIBILITY_LABELED_VISIBLE = 2

class KeypointAnnotator:
    # Modified constructor to accept multiple classes
    def __init__(self, keypoint_names, available_classes, image_dir, output_dir):
        """
        Initializes the annotator.

        Args:
            keypoint_names (list): List of keypoint names in order.
            available_classes (dict): Dictionary mapping class_id (int) to class_name (str).
            image_dir (str): Path to the directory containing images.
            output_dir (str): Path to the directory where label files (.txt) will be saved.
        """
        if not keypoint_names:
            messagebox.showerror("Error", "Keypoint names list cannot be empty.")
            sys.exit(1)
        if not available_classes:
            messagebox.showerror("Error", "Available classes dictionary cannot be empty.")
            sys.exit(1)

        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.available_classes = OrderedDict(sorted(available_classes.items())) # Store sorted by ID
        self.image_dir = image_dir
        self.output_dir = output_dir # This is where labels (.txt) are saved

        self.image_files = self._get_image_files()
        if not self.image_files:
            messagebox.showerror("Error", f"No valid image files found in {self.image_dir}")
            sys.exit(1)

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Found {len(self.image_files)} images.")
        print(f"Output directory for annotations (labels): {self.output_dir}")
        print(f"Keypoint sequence: {self.keypoint_names}")
        print("Available Classes:")
        for cid, cname in self.available_classes.items():
            print(f"  ID {cid}: {cname}")

        self.current_image_index = 0
        self.current_image_path = None
        self.img_display = None
        self.img_height = 0
        self.img_width = 0
        self.img_original = None

        # --- State for the *current* annotation instance ---
        self.mode = MODE_SELECT_CLASS
        self.current_class_id = None
        self.current_class_name = None
        self.current_keypoint_index = 0
        self.current_keypoints = [] # List of {'name': str, 'x': int, 'y': int, 'v': int}
        self.current_bbox_start_point = None
        self.current_bbox_end_point = None
        # --- --- --- --- --- --- --- --- --- --- --- --- ---

        # --- State for *all completed* annotations in the current image ---
        self.annotations_in_image = [] # List of {'class_id': int, 'class_name': str, 'keypoints': list, 'bbox': tuple(x1,y1,x2,y2)}
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        self.current_mouse_pos = None

        self.window_name = 'YOLO Multi-Animal Keypoint & BBox Annotator'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_events)

    def _get_image_files(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        try:
            files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_extensions)]
            files = [f for f in files if not f.startswith('.')] # Filter out hidden files
            return sorted([os.path.join(self.image_dir, f) for f in files])
        except FileNotFoundError:
            messagebox.showerror("Error", f"Image directory not found: {self.image_dir}")
            return []
        except Exception as e:
            messagebox.showerror("Error", f"Error reading image directory: {e}")
            return []

    def _reset_current_instance_state(self):
        """Resets the state for annotating a single new instance."""
        self.mode = MODE_SELECT_CLASS # Go back to selecting class
        self.current_class_id = None
        self.current_class_name = None
        self.current_keypoint_index = 0
        self.current_keypoints = []
        self.current_bbox_start_point = None
        self.current_bbox_end_point = None
        self.current_mouse_pos = None
        print(" -> Ready to annotate next instance. Select Class.")

    def _clear_all_annotations_for_image(self):
        """Clears all completed annotations and resets the current instance state."""
        self.annotations_in_image = []
        self._reset_current_instance_state()
        print("Cleared all annotations for the current image.")


    def _load_image(self, index):
        if not (0 <= index < len(self.image_files)):
            print("Invalid image index.")
            return False

        self.current_image_index = index
        self.current_image_path = self.image_files[self.current_image_index]
        self._clear_all_annotations_for_image() # Clear everything when loading new image

        img = cv2.imread(self.current_image_path)
        if img is None:
            if os.path.exists(self.current_image_path):
                 print(f"Warning: cv2.imread failed for existing file: {self.current_image_path}. Check file integrity/permissions.")
            messagebox.showerror("Error", f"Failed to load image: {self.current_image_path}")
            # Try to gracefully move to the next image or handle the error
            # For now, return False and let the main loop handle it
            return False

        self.img_original = img
        self.img_height, self.img_width = img.shape[:2]

        print(f"\nLoaded image: {os.path.basename(self.current_image_path)} ({self.current_image_index + 1}/{len(self.image_files)})")
        self._update_display()
        return True

    def _handle_mouse_events(self, event, x, y, flags, param):
        # Clamp coordinates to image boundaries
        x = max(0, min(x, self.img_width - 1))
        y = max(0, min(y, self.img_height - 1))

        # Update current mouse position for dynamic drawing
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_pos = (x, y)
            # Redraw only if needed (e.g., drawing bbox)
            if self.mode == MODE_BBOX_END:
                self._update_display()

        # --- Left Mouse Button Click ---
        elif event == cv2.EVENT_LBUTTONDOWN:
            # --- Annotating Keypoints ---
            if self.mode == MODE_KP:
                if self.current_keypoint_index < self.num_keypoints:
                    kp_name = self.keypoint_names[self.current_keypoint_index]
                    # Add keypoint, default to visible (can be toggled with MMB later)
                    self.current_keypoints.append({'name': kp_name, 'x': x, 'y': y, 'v': VISIBILITY_LABELED_VISIBLE})
                    print(f"   Placed KP {self.current_keypoint_index + 1}/{self.num_keypoints}: {kp_name} at ({x},{y}) [Visible]")
                    self.current_keypoint_index += 1

                    if self.current_keypoint_index == self.num_keypoints:
                        self.mode = MODE_BBOX_START
                        print("   All keypoints placed for this instance. Now draw bounding box.")
                else:
                     # Should not happen if logic is correct, but as safeguard:
                     self.mode = MODE_BBOX_START
                     print("   Already placed all keypoints. Switching to BBox mode.")

            # --- Starting Bounding Box ---
            elif self.mode == MODE_BBOX_START:
                self.current_bbox_start_point = (x, y)
                self.mode = MODE_BBOX_END
                print(f"   BBox Start Point: ({x},{y})")

            # --- Ending Bounding Box ---
            elif self.mode == MODE_BBOX_END:
                if self.current_bbox_start_point is None:
                    print("Error: BBox start point not set. Click top-left first.")
                    self.mode = MODE_BBOX_START # Revert state
                    return

                x1, y1 = self.current_bbox_start_point
                x2, y2 = x, y

                # Ensure x2 > x1 and y2 > y1
                if x2 <= x1 or y2 <= y1:
                    messagebox.showwarning("BBox Error", "Bottom-right corner must be below and to the right of the top-left corner. Please click again.")
                    # Don't change mode, let user click bottom-right again
                    return

                self.current_bbox_end_point = (x, y)
                print(f"   BBox End Point: ({x},{y})")

                # --- Instance Annotation Complete ---
                # Store the completed annotation
                completed_annotation = {
                    'class_id': self.current_class_id,
                    'class_name': self.current_class_name,
                    'keypoints': self.current_keypoints.copy(), # Store a copy
                    'bbox': (x1, y1, x2, y2)
                }
                self.annotations_in_image.append(completed_annotation)
                print(f"   Completed annotation for instance {len(self.annotations_in_image)} ({self.current_class_name}).")

                # Reset state for the *next* instance, back to class selection
                self._reset_current_instance_state()

            elif self.mode == MODE_SELECT_CLASS:
                messagebox.showinfo("Info", "Select a class using number keys (0-9) first.")

            self._update_display() # Update visuals after any significant click

        # --- Middle Mouse Button Click ---
        elif event == cv2.EVENT_MBUTTONDOWN:
            if self.mode == MODE_KP and self.current_keypoints:
                # Toggle visibility of the *last placed* keypoint
                last_kp_index = self.current_keypoint_index - 1
                if 0 <= last_kp_index < len(self.current_keypoints):
                    kp = self.current_keypoints[last_kp_index]
                    if kp['v'] == VISIBILITY_LABELED_VISIBLE:
                        kp['v'] = VISIBILITY_LABELED_NOT_VISIBLE
                        vis_str = "Not Visible"
                    elif kp['v'] == VISIBILITY_LABELED_NOT_VISIBLE:
                        kp['v'] = VISIBILITY_LABELED_VISIBLE
                        vis_str = "Visible"
                    else: # Should not happen if placed via LMB first
                        kp['v'] = VISIBILITY_LABELED_VISIBLE
                        vis_str = "Visible (Reset)"

                    print(f"   Toggled visibility for KP {last_kp_index + 1} ({kp['name']}) to: {vis_str}")
                    self._update_display() # Update display to show color change
                else:
                     print("   Cannot toggle visibility: Invalid keypoint index.")
            elif self.mode == MODE_KP and not self.current_keypoints:
                print("   Place at least one keypoint before toggling visibility.")
            else:
                print(f"   Cannot toggle visibility in mode: {self.mode}")


    def _update_display(self):
        if self.img_original is None:
            return
        self.img_display = self.img_original.copy()
        h, w = self.img_height, self.img_width

        # --- Draw Completed Annotations ---
        for i, ann in enumerate(self.annotations_in_image):
            x1, y1, x2, y2 = ann['bbox']
            cv2.rectangle(self.img_display, (x1, y1), (x2, y2), BBOX_COMPLETED_COLOR, COMPLETED_LINE_THICKNESS)

            # Draw class name label for completed boxes
            label = f"{ann['class_id']}:{ann['class_name']}"
            (label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE*0.8, FONT_THICKNESS)
            cv2.rectangle(self.img_display, (x1, y1 - label_height - baseline), (x1 + label_width, y1), BBOX_COMPLETED_COLOR, -1)
            cv2.putText(self.img_display, label, (x1, y1 - baseline // 2), FONT, FONT_SCALE*0.8, CLASS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


            # Draw completed keypoints (less emphasis)
            for kp in ann['keypoints']:
                 color = KP_VISIBLE_COLOR if kp['v'] == VISIBILITY_LABELED_VISIBLE else KP_INVISIBLE_COLOR
                 cv2.circle(self.img_display, (kp['x'], kp['y']), 3, color, -1) # Smaller circle

        # --- Draw Current Annotation ---
        # Keypoints being placed currently
        for i, kp in enumerate(self.current_keypoints):
            color = KP_VISIBLE_COLOR if kp['v'] == VISIBILITY_LABELED_VISIBLE else KP_INVISIBLE_COLOR
            cv2.circle(self.img_display, (kp['x'], kp['y']), 5, color, -1) # Larger circle
            cv2.putText(self.img_display, str(i+1), (kp['x']+5, kp['y']+5), FONT, 0.5, color, 1)

        # Bbox start point if placed
        if self.current_bbox_start_point and self.mode == MODE_BBOX_END:
             cv2.circle(self.img_display, self.current_bbox_start_point, 5, BBOX_CURRENT_TEMP_COLOR, -1)

        # Bbox being drawn currently (temporary rectangle)
        if self.mode == MODE_BBOX_END and self.current_bbox_start_point and self.current_mouse_pos:
            x_end = max(0, min(self.current_mouse_pos[0], w - 1))
            y_end = max(0, min(self.current_mouse_pos[1], h - 1))
            cv2.rectangle(self.img_display, self.current_bbox_start_point, (x_end, y_end), BBOX_CURRENT_TEMP_COLOR, CURRENT_LINE_THICKNESS)

        # Final bbox for the current instance (before adding to completed list)
        if self.current_bbox_start_point and self.current_bbox_end_point: # Will only show briefly before reset
            cv2.rectangle(self.img_display, self.current_bbox_start_point, self.current_bbox_end_point, BBOX_CURRENT_FINAL_COLOR, CURRENT_LINE_THICKNESS)


        # --- Draw Text Instructions ---
        y_offset = 30
        status_text = f"Image: {os.path.basename(self.current_image_path)} ({self.current_image_index + 1}/{len(self.image_files)}) | Annotations: {len(self.annotations_in_image)}"
        cv2.putText(self.img_display, status_text, (10, y_offset), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        y_offset += 30

        instruction_text = ""
        if self.mode == MODE_SELECT_CLASS:
            instruction_text = "Select Class (Press #): "
            class_options = []
            for i, (cid, cname) in enumerate(self.available_classes.items()):
                 # Allow selection using 1-9, then potentially 0 or other keys if > 9
                 key_char = str(i + 1) if (i+1) < 10 else '?' # Basic mapping for now
                 class_options.append(f"({key_char}) {cname}")
            instruction_text += " | ".join(class_options)
            if not self.available_classes: instruction_text = "No classes defined!"
        elif self.mode == MODE_KP:
            if self.current_keypoint_index < self.num_keypoints:
                kp_name = self.keypoint_names[self.current_keypoint_index]
                instruction_text = f"Class '{self.current_class_name}'. Click KP {self.current_keypoint_index + 1}/{self.num_keypoints}: {kp_name}. (MMB: Toggle Vis)"
            else: instruction_text = "Error: Keypoint index out of bounds." # Should not happen
        elif self.mode == MODE_BBOX_START:
            instruction_text = f"Class '{self.current_class_name}'. Click TOP-LEFT corner of the bounding box."
        elif self.mode == MODE_BBOX_END:
             instruction_text = f"Class '{self.current_class_name}'. Click BOTTOM-RIGHT corner of the bounding box."

        cv2.putText(self.img_display, instruction_text, (10, y_offset), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        y_offset += 30
        cv2.putText(self.img_display, "Keys: (n)ext (p)rev (u)ndo_kp (r)eset_img (s)ave_now (q)uit", (10, y_offset), FONT, 0.5, (255, 255, 0), 1)
        y_offset += 20
        cv2.putText(self.img_display, "(MMB: MiddleMouseBtn)", (10, y_offset), FONT, 0.5, (255, 255, 0), 1)


        cv2.imshow(self.window_name, self.img_display)

    def _undo_last_keypoint(self):
        """Undoes the last placed keypoint for the CURRENT instance."""
        if self.mode == MODE_KP and self.current_keypoints:
            removed_kp = self.current_keypoints.pop()
            self.current_keypoint_index -= 1
            print(f"   Undo: Removed keypoint {removed_kp['name']} for current instance.")
            self._update_display()
        elif self.mode != MODE_KP:
            messagebox.showwarning("Undo", "Can only undo keypoints while in Keypoint placement mode.")
        else:
            print("No keypoints placed for the current instance to undo.")

    def _save_annotations_for_image(self):
        """Saves all annotations currently stored for this image."""
        if not self.annotations_in_image:
            # Decide whether to save an empty file or just inform the user
            print(f"No annotations to save for {os.path.basename(self.current_image_path)}.")
            # Optionally, save an empty file:
            # output_filename = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
            # output_path = os.path.join(self.output_dir, output_filename)
            # open(output_path, 'w').close()
            # print(f"Saved empty annotation file: {output_path}")
            return True # Indicate success (nothing to save is not an error)

        output_filename = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
        output_path = os.path.join(self.output_dir, output_filename)

        lines_to_write = []
        img_h, img_w = self.img_height, self.img_width

        for ann in self.annotations_in_image:
            class_id = ann['class_id']
            x1, y1, x2, y2 = ann['bbox']
            keypoints = ann['keypoints'] # This is the list of dicts

            # Calculate normalized bounding box center, width, height
            box_w = x2 - x1
            box_h = y2 - y1
            cx = x1 + box_w / 2
            cy = y1 + box_h / 2

            norm_cx = max(0.0, min(1.0, cx / img_w))
            norm_cy = max(0.0, min(1.0, cy / img_h))
            norm_w = max(0.0, min(1.0, box_w / img_w))
            norm_h = max(0.0, min(1.0, box_h / img_h))

            # Prepare keypoints in YOLO format (x_norm, y_norm, visibility) * num_keypoints
            # Ensure they are in the order defined by self.keypoint_names
            yolo_kpts_flat = []
            placed_kpts = {kp['name']: kp for kp in keypoints} # Map name to the placed kp data

            for kp_name in self.keypoint_names: # Iterate through the canonical order
                if kp_name in placed_kpts:
                    kp = placed_kpts[kp_name]
                    norm_x = max(0.0, min(1.0, kp['x'] / img_w))
                    norm_y = max(0.0, min(1.0, kp['y'] / img_h))
                    visibility = kp['v'] # Already 0, 1, or 2
                    yolo_kpts_flat.extend([norm_x, norm_y, float(visibility)])
                else:
                    # This case shouldn't happen if annotation completes fully
                    # but if it does, mark as not labeled
                    print(f"Warning: Keypoint '{kp_name}' not found in completed annotation for {output_filename}. Saving as non-labeled.")
                    yolo_kpts_flat.extend([0.0, 0.0, float(VISIBILITY_NOT_LABELED)])

            # Format the line: class_id cx cy w h kpt1_x kpt1_y kpt1_v kpt2_x ...
            bbox_str = f"{norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"
            # Format kpts: x, y as float, v as int
            kpts_str = " ".join([f"{val:.6f}" if i % 3 < 2 else str(int(val)) for i, val in enumerate(yolo_kpts_flat)])

            yolo_line = f"{class_id} {bbox_str} {kpts_str}"
            lines_to_write.append(yolo_line)

        try:
            with open(output_path, 'w') as f:
                for line in lines_to_write:
                    f.write(line + "\n")
            print(f"Annotations ({len(lines_to_write)} instances) saved to: {output_path}")
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write annotation file: {e}")
            return False

    def run(self):
        if not self._load_image(self.current_image_index):
             print("Could not load initial image. Exiting.")
             return

        # --- Define mapping from keyboard keys ('1'-'9', '0') to class IDs ---
        class_keys = {}
        key_to_class_string_map = {} # For display purposes
        class_ids = list(self.available_classes.keys()) # Get the list of defined class IDs

        for i, cid in enumerate(class_ids):
            if i < 9:  # Map keys '1' through '9' to the first 9 classes (indices 0-8)
                key_char = str(i + 1)
                class_keys[key_char] = cid
                key_to_class_string_map[key_char] = f"{self.available_classes[cid]}" # Store display string: "1: mouse"
            elif i == 9: # Map key '0' to the 10th class (index 9)
                key_char = '0'
                class_keys[key_char] = cid
                key_to_class_string_map[key_char] = f"{self.available_classes[cid]}" # Store display string: "0: person"
            # else: # Classes beyond the 10th cannot be selected with 0-9 keys
                # break # Or print a warning

        print(f"Key mapping for class selection: {key_to_class_string_map}") # Print mapping to console for debugging/info
        # --- End of key mapping definition ---

        while True:
            if self.img_display is None:
                print("Error: No image is currently displayed. Attempting to load next.")
                # Handle case where image loading failed previously
                if self.current_image_index + 1 < len(self.image_files):
                    if not self._load_image(self.current_image_index + 1):
                         print("Failed to load next image as well. Exiting.")
                         break # Exit if loading fails consecutively
                else:
                     print("Reached end of images after a loading error.")
                     break # Exit if last image failed to load

            # Display is updated within handlers now, but call once per loop just in case
            # self._update_display() # Can be removed if updates are reliable in handlers
            key = cv2.waitKey(1) & 0xFF

            # --- Quit ---
            if key == ord('q'):
                if messagebox.askyesno("Quit", "Are you sure you want to quit? Unsaved progress on the *current* image will be lost."):
                    break

            # --- Next Image ---
            elif key == ord('n'):
                # Save current image's annotations before moving
                print("Saving annotations for current image before moving to next...")
                if not self._save_annotations_for_image():
                     if not messagebox.askretrycancel("Save Error", "Failed to save annotations. Continue to next image anyway (annotations will be lost)?"):
                         continue # Stay on current image if user cancels

                # Move to next
                if self.current_image_index + 1 < len(self.image_files):
                    if not self._load_image(self.current_image_index + 1):
                         break # Exit if loading next fails
                else:
                    messagebox.showinfo("End", "Already at the last image.")

            # --- Previous Image ---
            elif key == ord('p'):
                 # Save current image's annotations before moving
                print("Saving annotations for current image before moving to previous...")
                if not self._save_annotations_for_image():
                     if not messagebox.askretrycancel("Save Error", "Failed to save annotations. Continue to previous image anyway (annotations will be lost)?"):
                         continue # Stay on current image

                # Move to previous
                if self.current_image_index - 1 >= 0:
                     if not self._load_image(self.current_image_index - 1):
                         break # Exit if loading previous fails
                else:
                     messagebox.showinfo("Start", "Already at the first image.")

            # --- Undo Last Keypoint ---
            elif key == ord('u'):
                self._undo_last_keypoint()

            # --- Reset Image ---
            elif key == ord('r'):
                if messagebox.askyesno("Reset Image", "Clear ALL annotations for this image?"):
                    self._clear_all_annotations_for_image()
                    self._update_display()

            # --- Save Now ---
            elif key == ord('s'):
                if self._save_annotations_for_image():
                    messagebox.showinfo("Saved", f"Annotations for {os.path.basename(self.current_image_path)} saved.")
                # Error message shown by save function if it fails

            # --- Class Selection ---
            elif self.mode == MODE_SELECT_CLASS and chr(key) in class_keys:
                selected_cid = class_keys[chr(key)]
                self.current_class_id = selected_cid
                self.current_class_name = self.available_classes[selected_cid]
                self.mode = MODE_KP # Move to keypoint mode
                self.current_keypoint_index = 0
                self.current_keypoints = []
                self.current_bbox_start_point = None
                self.current_bbox_end_point = None
                print(f"Selected Class: {self.current_class_name} (ID: {self.current_class_id}). Start placing keypoints.")
                self._update_display()


        cv2.destroyAllWindows()

    def generate_config_yaml(self):
        """Generates and saves a YOLOv8 dataset config YAML file based on all classes defined."""
        print("\nAttempting to generate dataset configuration YAML...")

        suggested_root = os.path.abspath(os.path.join(self.output_dir, '..', 'dataset')) # Suggest dataset dir one level up from labels
        train_img_path = "images/train"
        val_img_path = "images/val"
        # Labels assumed to be in corresponding labels/train, labels/val

        # Use the sorted available_classes dictionary
        class_names_map = {cid: name for cid, name in self.available_classes.items()}
        num_classes = len(self.available_classes)

        yaml_data = {
            'path': suggested_root, # Path relative to where YOLO command is run
            'train': train_img_path,
            'val': val_img_path,
            # 'test': '', # Optional

            'kpt_shape': [self.num_keypoints, 3], # [num_kpts, 3 (x,y,visible)]

            'nc': num_classes,
            'names': class_names_map
        }

        header_comments = f"""
# YOLOv8 Dataset Configuration File (Auto-generated by Annotator Tool)
#
# Please review and adjust paths if necessary. The 'path' should be the root
# directory containing 'images' and 'labels' folders.
#
# IMPORTANT: You MUST manually create the following directory structure
# relative to the location specified in 'path' ({suggested_root}):
#
# {os.path.basename(suggested_root)}/
#   ├── images/
#   │   ├── train/    <-- Put your training image files here (*.jpg, *.png, etc.)
#   │   └── val/      <-- Put your validation image files here
#   └── labels/
#       ├── train/    <-- Put the generated training label files (*.txt) here
#       └── val/      <-- Put the generated validation label files (*.txt) here
#
# - 'path': Root dataset directory. Training command should be run from its parent.
# - 'train', 'val': Paths to image directories relative to 'path'.
# - 'kpt_shape': [Num Keypoints ({self.num_keypoints}), Dimensions (3: x,y,vis)].
# - 'nc': Number of classes ({num_classes}).
# - 'names': Maps class IDs to names (check order matches your training).
#
# Keypoint order used during annotation (and expected by this config):
# {self.keypoint_names}
# ---

"""

        # Ask user where to save the YAML file
        yaml_save_path = filedialog.asksaveasfilename(
            title="Save Dataset Config YAML File",
            initialdir=os.path.dirname(self.output_dir), # Near labels dir
            initialfile="data_config.yaml",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )

        if not yaml_save_path:
            print("YAML file saving cancelled by user.")
            return

        try:
            with open(yaml_save_path, 'w', encoding='utf-8') as f:
                f.write(header_comments)
                # Dump using the sorted class names map
                yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)
            print(f"Dataset config YAML saved successfully to: {yaml_save_path}")
            messagebox.showinfo("YAML Saved", f"Dataset config YAML saved to:\n{yaml_save_path}\n\nPlease read the comments regarding directory structure and data splitting.")
        except Exception as e:
            print(f"Error saving YAML file: {e}")
            messagebox.showerror("YAML Save Error", f"Could not save YAML file:\n{e}")


# --- Main Execution (Modified for Multi-Class Setup) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    print("Starting Annotation Tool Setup...")

    # 1. Get Keypoint Sequence
    kp_seq_str = simpledialog.askstring("Keypoint Sequence", "Enter keypoint names in order, separated by commas (,)\ne.g., nose,left_eye,right_eye,...")
    if not kp_seq_str: print("No keypoint sequence entered. Exiting."); sys.exit(1)
    keypoint_names = [name.strip() for name in kp_seq_str.split(',') if name.strip()]
    if not keypoint_names: print("Invalid keypoint sequence entered. Exiting."); sys.exit(1)

    # 2. Get Classes (ID and Name) - Loop until cancel/empty
    available_classes = {}
    next_id = 0
    while True:
        prompt = f"Enter Class Name for ID {next_id} (or leave blank/cancel to finish defining classes):"
        class_name = simpledialog.askstring("Define Class", prompt)

        if class_name is None: # User pressed Cancel
            print("Class definition cancelled.")
            break
        class_name = class_name.strip()
        if not class_name: # User left blank and pressed OK
             print("Finished defining classes.")
             break

        if class_name in available_classes.values():
            messagebox.showwarning("Duplicate Name", f"Class name '{class_name}' already used. Please use a unique name.")
            continue # Ask again for the same ID

        available_classes[next_id] = class_name
        print(f"  Added Class ID {next_id}: {class_name}")
        next_id += 1

    if not available_classes:
        print("No classes were defined. Exiting.")
        sys.exit(1)

    # 3. Get Image Directory
    image_dir = filedialog.askdirectory(title="Select Directory Containing Images")
    if not image_dir: print("No image directory selected. Exiting."); sys.exit(1)

    # 4. Get Output Directory (for labels/.txt files)
    output_dir = filedialog.askdirectory(title="Select Directory to Save YOLO Annotations (.txt files)")
    if not output_dir: print("No output directory selected. Exiting."); sys.exit(1)

    # 5. Initialize and Run Annotator
    annotator = None
    try:
        annotator = KeypointAnnotator(keypoint_names, available_classes, image_dir, output_dir)
        annotator.run() # Start the main annotation loop
    except Exception as e:
        print(f"\nAn error occurred during annotation: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        messagebox.showerror("Runtime Error", f"An unexpected error occurred:\n{e}\n\nSee console for details.")
    finally:
        # --- Generate YAML after the loop finishes (user quits) ---
        if annotator:
            print("\nAnnotation session finished.")
            if messagebox.askyesno("Generate Config?", "Annotation session complete. Generate YOLO dataset config YAML file now?"):
                annotator.generate_config_yaml()
            else:
                print("Skipping YAML generation.")
        else:
            print("\nAnnotation tool did not initialize properly. Cannot generate YAML.")

    root.destroy() # Clean up Tkinter
    print("Exiting annotation tool.")