import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import numpy as np
import sys
import yaml # Import PyYAML

# --- Configuration ---
# Colors (BGR)
POINT_COLOR = (0, 255, 0)  # Green for keypoints
BBOX_TEMP_COLOR = (255, 255, 0) # Cyan for drawing bbox
BBOX_FINAL_COLOR = (0, 255, 255) # Yellow for final bbox
TEXT_COLOR = (255, 0, 0) # Blue for instructions
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1

# Annotation Modes
MODE_KP = "keypoints"
MODE_BBOX_START = "bbox_start"
MODE_BBOX_END = "bbox_end"
MODE_DONE = "done"

# Visibility flags (YOLO standard)
VISIBILITY_NOT_LABELED = 0
VISIBILITY_LABELED_NOT_VISIBLE = 1
VISIBILITY_LABELED_VISIBLE = 2

class KeypointAnnotator:
    # Modified constructor to accept class_name
    def __init__(self, keypoint_names, class_id, class_name, image_dir, output_dir):
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.class_id = class_id
        self.class_name = class_name # Store class name
        self.image_dir = image_dir
        self.output_dir = output_dir # This is where labels (.txt) are saved

        # --- rest of __init__ remains the same ---
        self.image_files = self._get_image_files()
        if not self.image_files:
            messagebox.showerror("Error", f"No valid image files found in {self.image_dir}")
            sys.exit(1) # Exit if no images

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Found {len(self.image_files)} images.")
        print(f"Output directory for annotations (labels): {self.output_dir}")
        print(f"Keypoint sequence: {self.keypoint_names}")
        print(f"Class ID: {self.class_id}, Class Name: {self.class_name}") # Print class name

        self.current_image_index = 0
        self.current_image_path = None
        self.img_display = None
        self.img_height = 0
        self.img_width = 0

        self.mode = MODE_KP
        self.current_keypoint_index = 0
        self.keypoints = []
        self.bbox_start_point = None
        self.bbox_end_point = None
        self.current_mouse_pos = None

        self.window_name = 'YOLO Keypoint & BBox Annotator'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._handle_mouse_events)


    # --- _get_image_files, _reset_state, _load_image methods remain the same ---
    def _get_image_files(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        try:
            files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_extensions)]
            # Filter out hidden files like .DS_Store
            files = [f for f in files if not f.startswith('.')]
            return sorted([os.path.join(self.image_dir, f) for f in files])
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading image directory: {e}")
            return []

    def _reset_state(self):
        """Resets annotation state for the current image."""
        self.mode = MODE_KP
        self.current_keypoint_index = 0
        self.keypoints = []
        self.bbox_start_point = None
        self.bbox_end_point = None
        self.current_mouse_pos = None
        # Don't print reset message here, too verbose

    def _load_image(self, index):
        if not (0 <= index < len(self.image_files)):
            print("Invalid image index.")
            return False

        self.current_image_index = index
        self.current_image_path = self.image_files[self.current_image_index]
        self._reset_state() # Reset state when loading a new image

        img = cv2.imread(self.current_image_path)
        if img is None:
            # Try alternative loading if path seems okay but loading failed
            if os.path.exists(self.current_image_path):
                 print(f"Warning: cv2.imread failed for existing file: {self.current_image_path}. Check file integrity/permissions.")
            messagebox.showerror("Error", f"Failed to load image: {self.current_image_path}")
            return False


        self.img_original = img # Store original for clean redraws
        self.img_height, self.img_width = img.shape[:2]

        print(f"Loaded image: {os.path.basename(self.current_image_path)} ({self.current_image_index + 1}/{len(self.image_files)})")
        self._update_display()
        return True


    # --- _handle_mouse_events, _update_display, _undo_last_keypoint methods remain the same ---
    def _handle_mouse_events(self, event, x, y, flags, param):
        # Update current mouse position for dynamic drawing
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse_pos = (x, y)
            # Redraw only if in bbox drawing mode for efficiency
            if self.mode == MODE_BBOX_END:
                 self._update_display()

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Clamp coordinates to image boundaries
            x = max(0, min(x, self.img_width - 1))
            y = max(0, min(y, self.img_height - 1))

            # --- Annotating Keypoints ---
            if self.mode == MODE_KP:
                if self.current_keypoint_index < self.num_keypoints:
                    kp_name = self.keypoint_names[self.current_keypoint_index]
                    self.keypoints.append({'name': kp_name, 'x': x, 'y': y, 'v': VISIBILITY_LABELED_VISIBLE})
                    print(f"  Placed keypoint {self.current_keypoint_index + 1}/{self.num_keypoints}: {kp_name} at ({x},{y})")
                    self.current_keypoint_index += 1

                    if self.current_keypoint_index == self.num_keypoints:
                        self.mode = MODE_BBOX_START
                        print("  All keypoints placed. Now draw bounding box.")
                else:
                     self.mode = MODE_BBOX_START

            # --- Starting Bounding Box ---
            elif self.mode == MODE_BBOX_START:
                self.bbox_start_point = (x, y)
                self.mode = MODE_BBOX_END
                print(f"  BBox Start Point: ({x},{y})")

            # --- Ending Bounding Box ---
            elif self.mode == MODE_BBOX_END:
                if self.bbox_start_point is None:
                    print("Error: BBox start point not set. Click top-left first.")
                    self.mode = MODE_BBOX_START # Revert state
                    return

                x1, y1 = self.bbox_start_point
                x2, y2 = x, y
                if x2 <= x1 or y2 <= y1:
                     messagebox.showwarning("BBox Error", "Bottom-right corner must be below and to the right of the top-left corner. Please click again.")
                     return

                self.bbox_end_point = (x, y)
                self.mode = MODE_DONE
                print(f"  BBox End Point: ({x},{y})")
                print("  Annotation complete for this image.")

            elif self.mode == MODE_DONE:
                messagebox.showinfo("Info", "Annotation already complete for this image. Press 'n' for next or 's' to save.")

            self._update_display() # Update visuals after any click


    def _update_display(self):
        if self.img_original is None:
            return
        self.img_display = self.img_original.copy()

        for i, kp in enumerate(self.keypoints):
            cv2.circle(self.img_display, (kp['x'], kp['y']), 5, POINT_COLOR, -1)
            cv2.putText(self.img_display, str(i+1), (kp['x']+5, kp['y']+5), FONT, 0.5, POINT_COLOR, 1)

        if self.bbox_start_point and self.bbox_end_point and self.mode == MODE_DONE:
             cv2.rectangle(self.img_display, self.bbox_start_point, self.bbox_end_point, BBOX_FINAL_COLOR, 2)
        elif self.bbox_start_point and self.mode == MODE_BBOX_END and self.current_mouse_pos:
            cv2.circle(self.img_display, self.bbox_start_point, 5, BBOX_TEMP_COLOR, -1)
            # Clamp end coordinates for drawing temporary rectangle
            x_end = max(0, min(self.current_mouse_pos[0], self.img_width - 1))
            y_end = max(0, min(self.current_mouse_pos[1], self.img_height - 1))
            cv2.rectangle(self.img_display, self.bbox_start_point, (x_end, y_end), BBOX_TEMP_COLOR, 1)

        status_text = f"Image: {os.path.basename(self.current_image_path)} ({self.current_image_index + 1}/{len(self.image_files)}) | Mode: {self.mode}"
        cv2.putText(self.img_display, status_text, (10, 30), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        instruction_text = ""
        if self.mode == MODE_KP:
            if self.current_keypoint_index < self.num_keypoints:
                kp_name = self.keypoint_names[self.current_keypoint_index]
                instruction_text = f"Click Keypoint {self.current_keypoint_index + 1}/{self.num_keypoints}: {kp_name}"
            else: instruction_text = "Error: Keypoint index out of bounds."
        elif self.mode == MODE_BBOX_START: instruction_text = "Click TOP-LEFT corner of the bounding box."
        elif self.mode == MODE_BBOX_END: instruction_text = "Click BOTTOM-RIGHT corner of the bounding box."
        elif self.mode == MODE_DONE: instruction_text = "Annotation DONE. Press 'n' for next, 's' to save."

        cv2.putText(self.img_display, instruction_text, (10, 60), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(self.img_display, "Keys: (n)ext (p)rev (u)ndo_kp (r)eset (s)ave (q)uit", (10, 90), FONT, 0.5, (255, 255, 0), 1)

        cv2.imshow(self.window_name, self.img_display)

    def _undo_last_keypoint(self):
        if self.mode == MODE_KP and self.keypoints:
            removed_kp = self.keypoints.pop()
            self.current_keypoint_index -= 1
            print(f"  Undo: Removed keypoint {removed_kp['name']}")
            self._update_display()
        elif self.mode != MODE_KP: messagebox.showwarning("Undo", "Can only undo keypoints before starting bounding box.")
        else: print("No keypoints to undo.")


    # --- _save_current_annotation method remains the same ---
    def _save_current_annotation(self):
        if self.mode != MODE_DONE:
            messagebox.showerror("Save Error", "Annotation incomplete. Finish placing keypoints and drawing the bounding box before saving.")
            return False

        if not self.bbox_start_point or not self.bbox_end_point:
             messagebox.showerror("Save Error", "Bounding box not defined.")
             return False

        output_filename = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
        output_path = os.path.join(self.output_dir, output_filename)

        x1, y1 = self.bbox_start_point
        x2, y2 = self.bbox_end_point
        box_w = x2 - x1
        box_h = y2 - y1
        cx = x1 + box_w / 2
        cy = y1 + box_h / 2

        norm_cx = max(0.0, min(1.0, cx / self.img_width))
        norm_cy = max(0.0, min(1.0, cy / self.img_height))
        norm_w = max(0.0, min(1.0, box_w / self.img_width))
        norm_h = max(0.0, min(1.0, box_h / self.img_height))

        yolo_kpts_flat = []
        placed_kpts = {kp['name']: kp for kp in self.keypoints}

        for kp_name in self.keypoint_names: # Use the stored sequence
            if kp_name in placed_kpts:
                kp = placed_kpts[kp_name]
                norm_x = max(0.0, min(1.0, kp['x'] / self.img_width))
                norm_y = max(0.0, min(1.0, kp['y'] / self.img_height))
                visibility = kp['v']
                yolo_kpts_flat.extend([norm_x, norm_y, float(visibility)])
            else:
                yolo_kpts_flat.extend([0.0, 0.0, float(VISIBILITY_NOT_LABELED)])

        bbox_str = f"{norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}"
        kpts_str = " ".join([f"{val:.6f}" if i % 3 < 2 else str(int(val)) for i, val in enumerate(yolo_kpts_flat)])

        yolo_line = f"{self.class_id} {bbox_str} {kpts_str}"

        try:
            with open(output_path, 'w') as f: f.write(yolo_line + "\n")
            print(f"Annotation saved to: {output_path}")
            return True
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write annotation file: {e}")
            return False

    # --- run method remains the same ---
    def run(self):
        if not self._load_image(self.current_image_index):
             print("Could not load initial image. Exiting.")
             return

        while True:
            if self.img_display is None:
                 print("Error: No image is currently displayed.")
                 if self.current_image_index + 1 < len(self.image_files):
                      if not self._load_image(self.current_image_index + 1): break
                 else: break

            cv2.imshow(self.window_name, self.img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if messagebox.askyesno("Quit", "Are you sure you want to quit? Unsaved progress will be lost."): break
            elif key == ord('n'):
                should_move = True
                if self.mode != MODE_DONE and (self.mode != MODE_KP or self.keypoints):
                    should_move = messagebox.askyesno("Confirm", "Annotation incomplete. Discard and move to next image?")
                if should_move:
                    if self.mode == MODE_DONE: self._save_current_annotation()
                    if self.current_image_index + 1 < len(self.image_files): self._load_image(self.current_image_index + 1)
                    else: messagebox.showinfo("End", "Already at the last image.")
            elif key == ord('p'):
                should_move = True
                if self.mode != MODE_KP or self.keypoints:
                     should_move = messagebox.askyesno("Confirm", "Annotation incomplete. Discard and move to previous image?")
                if should_move:
                    if self.current_image_index - 1 >= 0: self._load_image(self.current_image_index - 1)
                    else: messagebox.showinfo("Start", "Already at the first image.")
            elif key == ord('u'): self._undo_last_keypoint()
            elif key == ord('r'):
                if messagebox.askyesno("Reset", "Clear all annotations for this image?"):
                    self._reset_state()
                    self._update_display()
            elif key == ord('s'):
                if self.mode == MODE_DONE:
                     if self._save_current_annotation(): messagebox.showinfo("Saved", f"Annotation for {os.path.basename(self.current_image_path)} saved.")
                else: messagebox.showwarning("Save", "Annotation incomplete. Cannot save yet.")

        cv2.destroyAllWindows()


    # --- NEW METHOD to generate YAML ---
    def generate_config_yaml(self):
        """Generates and saves a YOLOv8 dataset config YAML file."""
        print("\nAttempting to generate dataset configuration YAML...")

        # Define the standard structure YOLO expects
        # Users NEED to manually create these directories and split data
        suggested_root = "../dataset" # Suggest putting YAML one level above images/labels
        train_img_path = "images/train"
        val_img_path = "images/val"
        # Labels assumed to be in corresponding labels/train, labels/val
        # The 'path' in YAML should point to the directory containing images/ and labels/

        yaml_data = {
            # Path relative to where the YOLO training command is run
            'path': suggested_root,
            'train': train_img_path, # Path to train images directory relative to 'path'
            'val': val_img_path,     # Path to validation images directory relative to 'path'
            # 'test': '', # Optional: Path to test images directory

            # Keypoint shape: [number of keypoints, number of dims (x, y, visibility)]
            'kpt_shape': [self.num_keypoints, 3],

            # Classes: number of classes and their names
            'nc': 1, # Assuming only one class was annotated based on input
            'names': {
                self.class_id: self.class_name
            }
        }

        # Create the header comments explaining the structure and user actions
        header_comments = f"""
# YOLOv8 Dataset Configuration File (Auto-generated by Annotator Tool)
#
# Please review and adjust paths if necessary.
#
# IMPORTANT: You MUST manually create the following directory structure
# relative to the location specified in 'path' ({suggested_root}):
#
# {suggested_root}/
#   ├── images/
#   │   ├── train/  <-- Put your training image files here (*.jpg, *.png, etc.)
#   │   └── val/    <-- Put your validation image files here
#   └── labels/
#       ├── train/  <-- Put the generated training label files (*.txt) here
#       └── val/    <-- Put the generated validation label files (*.txt) here
#
# - 'path': Specifies the root directory of your dataset. Adjust if needed.
# - 'train', 'val': Paths to image directories relative to 'path'.
# - 'kpt_shape': Defines the number of keypoints and dimensions [num_kpts, 3 (x,y,visible)].
# - 'nc': Number of classes. Currently set to 1 based on annotation session.
# - 'names': Maps class IDs to names.
#
# Keypoint order used during annotation (and expected by this config):
# {self.keypoint_names}
# ---

"""

        # Ask user where to save the YAML file
        yaml_save_path = filedialog.asksaveasfilename(
            title="Save Dataset Config YAML File",
            initialdir=os.path.dirname(self.output_dir), # Suggest saving near labels dir
            initialfile="data_config.yaml",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )

        if not yaml_save_path:
            print("YAML file saving cancelled by user.")
            return

        try:
            with open(yaml_save_path, 'w', encoding='utf-8') as f:
                f.write(header_comments) # Write the instructional comments first
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False) # Dump the actual config
            print(f"Dataset config YAML saved successfully to: {yaml_save_path}")
            messagebox.showinfo("YAML Saved", f"Dataset config YAML saved to:\n{yaml_save_path}\n\nPlease read the comments in the file regarding directory structure and data splitting.")
        except Exception as e:
            print(f"Error saving YAML file: {e}")
            messagebox.showerror("YAML Save Error", f"Could not save YAML file:\n{e}")


# --- Main Execution (Modified) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # Hide the main Tkinter window

    print("Starting Annotation Tool Setup...")

    # 1. Get Keypoint Sequence
    kp_seq_str = simpledialog.askstring("Keypoint Sequence", "Enter keypoint names in order, separated by commas (,)\ne.g., nose,left_eye,right_eye,...")
    if not kp_seq_str: print("No keypoint sequence entered. Exiting."); sys.exit(1)
    keypoint_names = [name.strip() for name in kp_seq_str.split(',') if name.strip()]
    if not keypoint_names: print("Invalid keypoint sequence entered. Exiting."); sys.exit(1)

    # 2. Get Class ID
    class_id = simpledialog.askinteger("Class ID", "Enter the Class ID for this object (e.g., 0 for person):", initialvalue=0, minvalue=0)
    if class_id is None: print("Class ID not provided. Exiting."); sys.exit(1)

    # 3. Get Class Name *** NEW ***
    class_name = simpledialog.askstring("Class Name", f"Enter the name for Class ID {class_id} (e.g., person):")
    if not class_name: print("Class name not provided. Exiting."); sys.exit(1)
    class_name = class_name.strip() # Clean up whitespace

    # 4. Get Image Directory
    image_dir = filedialog.askdirectory(title="Select Directory Containing Images")
    if not image_dir: print("No image directory selected. Exiting."); sys.exit(1)

    # 5. Get Output Directory (for labels/.txt files)
    output_dir = filedialog.askdirectory(title="Select Directory to Save YOLO Annotations (.txt files)")
    if not output_dir: print("No output directory selected. Exiting."); sys.exit(1)

    # 6. Initialize and Run Annotator
    annotator = None # Initialize to None
    try:
        annotator = KeypointAnnotator(keypoint_names, class_id, class_name, image_dir, output_dir)
        annotator.run() # Start the main annotation loop
    except Exception as e:
         print(f"\nAn error occurred during annotation: {e}")
         messagebox.showerror("Runtime Error", f"An unexpected error occurred:\n{e}")
    finally:
         # --- Generate YAML after the loop finishes (user quits) ---
         if annotator: # Check if annotator was successfully initialized
            print("\nAnnotation session finished.")
            # Ask user if they want to generate the config file
            if messagebox.askyesno("Generate Config?", "Annotation session complete. Generate YOLO dataset config YAML file now?"):
                 annotator.generate_config_yaml()
            else:
                 print("Skipping YAML generation.")
         else:
            print("\nAnnotation tool did not initialize properly. Cannot generate YAML.")

    root.destroy() # Clean up Tkinter
    print("Exiting annotation tool.")