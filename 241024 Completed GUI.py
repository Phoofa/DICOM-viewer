import os
import pydicom
import numpy as np
import flet as ft
import cv2
import base64
import subprocess  # To run the external Python file
import math  # For distance calculation
from Cryptodome.Cipher import AES as PyCryptoAES
from Cryptodome.Util.Padding import pad, unpad
from pydicom import dcmread, dcmwrite

class DICOMViewer:
    def __init__(self, page):
        self.page = page
        self.page.theme_mode = ft.ThemeMode.DARK
        self.file_paths = []  # List to store DICOM file paths
        self.current_index = 0  # Index of the currently displayed DICOM file
        self.is_flipped_horizontal = False  # Variable to track the horizontal flip state
        self.is_flipped_vertical = False
        self.rotation_angle = 0  # Variable to track the rotation angle
        self.window_width = None  # Default window width
        self.window_level = None  # Default window level
        self.scale_factor = 1.0  # Scale factor for zoom
        self.zoom_center = (0.5, 0.5)  # Center point for zoom (normalized coordinates)
        self.is_dragging_mode = False  # Variable to track dragging mode state
        self.is_adjusting_wl_ww = False  # Variable to track WL/WW adjustment mode
        self.drag_offset = (0, 0)  # Offset for dragging
        self.cached_image = None  # Cached image for efficient dragging
        self.start_drag_x = 0
        self.start_drag_y = 0
        self.show_annotations = True  # Variable to track if annotations should be shown
        self.page.window.maximized = True  # Updated to use the new way of setting the window maximized
        self.pixel_spacing = (1.0, 1.0)  # Default pixel spacing, assuming 1 mm if not provided by DICOM
        self.first_point = None  # To store the first clicked point
        self.second_point = None  # To store the second clicked point
        self.current_dicom = None  # Store the current DICOM metadata
        self.is_drawing_mode = False  # Variable to track if drawing mode is active
        self.drawing_points = []  # List to store drawing points
        self.drawing_image = None
        self.drawing_buffer = None

        # Initialize the list to store measurements
        self.measurements = []  # List to store measurement lines

        # Initialize the list to store drawn lines
        self.image_drawn_lines = {}  # Dictionary to store drawn lines per image index
        self.drawn_lines = self.image_drawn_lines.get(self.current_index, [])
        # Set up the keyboard event handler
        self.page.on_keyboard_event = self.on_keyboard_event

        # Create UI elements
        self.create_ui()

    def create_ui(self):
        self.error_label = ft.Text("")
        self.image_display = ft.Image(width=800, height=800)  # Adjust the size here
        self.gesture_detector = ft.GestureDetector(
            content=ft.Container(
                content=self.image_display,
                width=800,
                height=800,
            ),
            on_pan_start=self.on_pan_start,
            on_pan_update=self.on_pan_update,
            on_pan_end=self.on_pan_end,
        )

        # Adjust the width of the buttons
        self.next_button = ft.ElevatedButton("Next", on_click=self.show_next, width=100)
        self.prev_button = ft.ElevatedButton("Previous", on_click=self.show_previous, width=100)
        self.zoom_in_button = ft.ElevatedButton("Zoom In", on_click=self.zoom_in, width=100)
        self.zoom_out_button = ft.ElevatedButton("Zoom Out", on_click=self.zoom_out, width=100)
        self.toggle_drag_button = ft.ElevatedButton("Pan", on_click=self.toggle_drag_mode, width=100)
        self.toggle_wl_ww_button = ft.ElevatedButton("WL/WW Adjust", on_click=self.toggle_wl_ww_mode, width=100)
        self.toggle_annotation_button = ft.ElevatedButton("Annotation", on_click=self.toggle_annotation, width=100)
        self.toggle_measurement_button = ft.ElevatedButton("Measurement", on_click=self.toggle_measurement_mode, width=100)
        # Add the "Encryption Function" button to trigger the external file
        self.encryption_button = ft.ElevatedButton("Export", on_click=self.run_encryption_window, width=150)
        self.toggle_drawing_button = ft.ElevatedButton("Pencil", on_click=self.toggle_drawing_mode, width=100)


        self.wl_label = ft.Text(f"Window Level: {self.window_level}")
        self.ww_label = ft.Text(f"Window Width: {self.window_width}")

        # Create file pickers
        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.page.overlay.append(self.file_picker)
        
        self.directory_picker = ft.FilePicker(on_result=self.on_directory_selected)
        self.page.overlay.append(self.directory_picker)

        # Create a menu bar with combined rotate/flip options without the width attribute
        self.menubar = ft.MenuBar(
            expand=False,
            controls=[
                ft.SubmenuButton(
                    content=ft.Row([ft.Text("File"), ft.Icon(ft.icons.ARROW_DROP_DOWN)]),
                    controls=[
                        ft.MenuItemButton(
                            content=ft.Text("Browse Folder"),
                            on_click=self.browse_directory
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Browse File"),
                            on_click=self.browse_dcm_file
                        )
                    ]
                ),
                ft.SubmenuButton(
                    content=ft.Row([ft.Text("Rotate/Flip"), ft.Icon(ft.icons.ARROW_DROP_DOWN)]),
                    controls=[
                        ft.MenuItemButton(
                            content=ft.Text("Rotate Clockwise"),
                            on_click=lambda _: self.rotate_image(clockwise=False)
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Rotate Counterclockwise"),
                            on_click=lambda _: self.rotate_image(clockwise=True)
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Flip Horizontal"),
                            on_click=lambda _: self.flip_image(horizontal=True)
                        ),
                        ft.MenuItemButton(
                            content=ft.Text("Flip Vertical"),
                            on_click=lambda _: self.flip_image(vertical=True)
                        )
                    ]
                )
            ]
        )

        # Create the list for DICOM thumbnails
        self.thumbnail_list = ft.ListView(expand=True, spacing=10, padding=10, auto_scroll=True)

        # Annotation box on the right to display DICOM information
        self.annotation_box = ft.Column([], alignment=ft.MainAxisAlignment.START, expand=True)
        self.is_measurement_mode = False  # Track if measurement mode is active

        # Buttons for the right-side layout (Encryption mode, checkboxes, process button)
        self.mode_selector = ft.Dropdown(
            label="Mode",
            options=[ft.dropdown.Option("Encryption"), ft.dropdown.Option("Decryption")],
            value="Encryption",
            width=150
        )

        self.patient_info_checkbox = ft.Checkbox(label="Patient Information", value=True)
        self.image_checkbox = ft.Checkbox(label="Image", value=True)

        self.process_button = ft.ElevatedButton("Process", on_click=self.process_encryption, width=150)

        # Add elements to the page
        self.page.add(
            ft.Row([
                # Left Column: Thumbnail List
                ft.Container(
                    content=self.thumbnail_list,
                    width=200,
                    height=1200,
                ),
                # Divider between thumbnails and main content
                ft.VerticalDivider(width=1, thickness=2),
                # Main content column (image viewer + controls)
                ft.Column([
                    ft.ResponsiveRow([
                        # Wrap the menubar in a container with a specified width
                        ft.Container(content=self.menubar, col={"sm": 12, "md": 3, "lg": 2}, width=200),
                        ft.Container(self.prev_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.next_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.zoom_in_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.zoom_out_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.toggle_drag_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.toggle_wl_ww_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.toggle_annotation_button, col={"sm": 6, "md": 2, "lg": 2}),
                        # Add the Encryption Function button here
                        ft.Container(self.encryption_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.toggle_measurement_button, col={"sm": 6, "md": 2, "lg": 2}),
                        ft.Container(self.toggle_drawing_button, col={"sm": 6, "md": 2, "lg": 2}),

                    ], alignment=ft.MainAxisAlignment.START),  # Align buttons to the start
                    self.error_label,
                    ft.Container(
                        content=self.gesture_detector,
                        alignment=ft.alignment.center,
                        expand=True
                    ),
                    ft.Column([self.wl_label, self.ww_label]),
                ], expand=True),
                # Divider between main content and annotations
                ft.VerticalDivider(width=1, thickness=2),
                # Right Column: Annotation box + Encryption/Decryption options
                ft.Container(
                    content=ft.Column([
                        # Patient information annotation
                        self.annotation_box,
                        # Encryption/Decryption options (without the divider)
                        ft.Text("Encryption/Decryption Options", size=16, weight="bold"),
                        self.mode_selector,
                        self.patient_info_checkbox,
                        self.image_checkbox,
                        self.process_button
                    ], alignment=ft.MainAxisAlignment.START, spacing=10),  # Reduced spacing
                    width=300,
                    padding=ft.padding.only(top=10)  # Adjust padding to bring it closer
                )
            ], expand=True)
        )

    def toggle_drawing_mode(self, e):
        self.is_drawing_mode = not self.is_drawing_mode
        if self.is_drawing_mode:
            self.toggle_drawing_button.style = ft.ButtonStyle(bgcolor=ft.colors.GREEN)
            self.error_label.value = "Freehand Drawing Mode Active"
        else:
            self.toggle_drawing_button.style = ft.ButtonStyle(bgcolor=None)  # Reset to default
            self.error_label.value = "Freehand Drawing Mode Inactive"
        self.page.update()
    
    def on_keyboard_event(self, e: ft.KeyboardEvent):
        if e.key.lower() == "z" and e.ctrl:
            self.clear_drawings()

    def clear_drawings(self):
        """Clears all drawn lines for the current image and updates the display."""
        # Remove the drawn lines for the current image from the dictionary
        if self.current_index in self.image_drawn_lines:
            del self.image_drawn_lines[self.current_index]
        
        # Reset the local drawn_lines list
        self.drawn_lines = []
        
        # Reset the drawing buffer
        self.drawing_buffer = None
        
        # Update the display
        self.display_image(self.cached_image)
        
        # Provide feedback to the user
        self.error_label.value = "All drawings have been cleared for this image."
        self.page.update()

    def toggle_measurement_mode(self, e):
        self.is_measurement_mode = not self.is_measurement_mode
        if self.is_measurement_mode:
            self.is_dragging_mode = False  # Disable panning when measurement is on
            self.toggle_measurement_button.style = ft.ButtonStyle(bgcolor=ft.colors.GREEN)
            self.error_label.value = "Measurement Mode Active"
        else:
            self.toggle_measurement_button.style = ft.ButtonStyle(bgcolor=None)  # Reset to default
            self.error_label.value = "Measurement Mode Inactive"
            # Clear existing measurements
            self.measurements.clear()
            self.first_point = None
            self.second_point = None
            self.display_image(self.cached_image)
        self.page.update()


    def run_encryption_window(self, e):
        """Run the external encryption window file."""
        try:
            subprocess.run(["python", "encryption_window.py"])  # Run encryption_window.py
        except Exception as ex:
            self.error_label.value = f"Error running encryption window: {str(ex)}"
            self.page.update()

    def toggle_annotation(self, e):
        self.show_annotations = not self.show_annotations
        if self.show_annotations:
            # Update the annotation button style to reflect that annotations are enabled
            self.toggle_annotation_button.style = ft.ButtonStyle(bgcolor=ft.colors.GREEN)
            self.add_annotations_to_box()  # Show the annotations
        else:
            self.toggle_annotation_button.style = ft.ButtonStyle(bgcolor=None)  # Reset to default
            self.annotation_box.controls.clear()  # Clear the annotations
        self.page.update()
    
    def browse_directory(self, e):
        self.directory_picker.get_directory_path(initial_directory=os.getcwd())

    def browse_dcm_file(self, e):
        self.file_picker.pick_files(allowed_extensions=['dcm'], allow_multiple=False)

    def on_directory_selected(self, e: ft.FilePickerResultEvent):
        if e.path:
            selected_directory = e.path
            # Load DICOM files from the selected directory
            self.file_paths = [os.path.join(selected_directory, file) for file in os.listdir(selected_directory) if file.endswith('.dcm')]
            if self.file_paths:
                self.load_thumbnails()
                self.current_index = 0
                self.load_dicom_image()
            else:
                self.error_label.value = f"No DICOM files found in directory: {selected_directory}"
                self.page.update()

    def on_file_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            selected_file = e.files[0].path
            # Load the selected DICOM file
            self.file_paths = [selected_file]
            self.load_thumbnails()
            self.current_index = 0
            self.load_dicom_image()
        else:
            self.error_label.value = "No DICOM file selected."
            self.page.update()

    def load_thumbnails(self):
        self.thumbnail_list.controls.clear()
        self.series_file_mapping = []

        series_dict = {}

        for file_path in self.file_paths:
            dicom_data = pydicom.dcmread(file_path)

            # Check if the file belongs to a series
            if hasattr(dicom_data, 'SeriesInstanceUID'):
                series_uid = dicom_data.SeriesInstanceUID
                if series_uid not in series_dict:
                    series_dict[series_uid] = file_path

        for idx, file_path in enumerate(series_dict.values()):
            dicom_data = pydicom.dcmread(file_path)

            # Check if the DICOM file contains pixel data
            if hasattr(dicom_data, 'PixelData'):
                image = dicom_data.pixel_array
                thumbnail = self.generate_thumbnail(image)

                # Extract metadata for the thumbnail
                series_date = getattr(dicom_data, 'SeriesDate', 'Unknown Date')
                series_time = getattr(dicom_data, 'SeriesTime', 'Unknown Time')
                series_description = getattr(dicom_data, 'SeriesDescription', 'Unknown Series')
                modality = getattr(dicom_data, 'Modality', 'Unknown Modality')

                formatted_date = f"{series_date[6:8]}/{series_date[4:6]}/{series_date[0:4]}" if len(series_date) == 8 else "Unknown Date"
                formatted_time = f"{series_time[0:2]}:{series_time[2:4]}:{series_time[4:6]}" if len(series_time) >= 6 else "Unknown Time"

                thumbnail_image = ft.Image(
                    src_base64=thumbnail,
                    width=150,
                    height=150,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=ft.border_radius.all(10)
                )

                # Add the metadata as text below the thumbnail
                series_description_text = ft.Text(f"Series: {series_description}", size=12)
                modality_text = ft.Text(f"Type: {modality}", size=12)

                thumbnail_container = ft.Container(
                    content=ft.Column([thumbnail_image, ft.Text(f"Date: {formatted_date} {formatted_time}", size=12), series_description_text, modality_text]),
                    on_click=lambda e, file_path=file_path: self.show_image_from_file(file_path),
                    padding=ft.padding.all(10),
                    border_radius=ft.border_radius.all(10),
                    border=ft.border.all(1, ft.colors.WHITE),
                )

                self.thumbnail_list.controls.append(thumbnail_container)
                self.series_file_mapping.append(file_path)

        self.page.update()



    def generate_thumbnail(self, image):
        # Resize the original image for the thumbnail without applying any adjustments
        resized_image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)

        # Normalize the image to 8-bit if it's not already in that format
        if image.dtype != np.uint8:
            resized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX)
            resized_image = resized_image.astype(np.uint8)

        # Encode the resized image to JPEG and then to base64
        _, buf = cv2.imencode('.jpg', resized_image)
        thumbnail_base64 = base64.b64encode(buf).decode('utf-8')
        
        return thumbnail_base64

    def show_image_at_index(self, idx):
        self.current_index = idx
        self.load_dicom_image()

    def load_dicom_image(self):
        if self.file_paths:
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])

                # Check if the DICOM file contains pixel data
                if hasattr(dicom_data, 'PixelData'):
                    self.cached_image = dicom_data.pixel_array

                    # Extract pixel spacing if available
                    if hasattr(dicom_data, 'PixelSpacing'):
                        if isinstance(dicom_data.PixelSpacing, pydicom.multival.MultiValue):
                            self.pixel_spacing = float(dicom_data.PixelSpacing[0])
                        else:
                            self.pixel_spacing = float(dicom_data.PixelSpacing)
                    else:
                        self.pixel_spacing = None

                    # Detect modality-specific WW/WL defaults
                    self.window_width, self.window_level = self.detect_modality_defaults(dicom_data)

                    # If WW/WL are not provided by the modality, calculate them from the image
                    if self.window_width is None or self.window_level is None:
                        min_val = np.min(self.cached_image)
                        max_val = np.max(self.cached_image)
                        self.window_width = max_val - min_val
                        self.window_level = (max_val + min_val) / 2

                    # Clear measurements when a new image is loaded
                    self.measurements.clear()
                    self.first_point = None
                    self.second_point = None

                    # **Load drawn lines for the current image**
                    self.drawn_lines = self.image_drawn_lines.get(self.current_index, [])
                    # **Redraw the drawing buffer**
                    self.redraw_drawing_buffer()

                    # Apply window level and window width to display the image
                    self.display_image(self.cached_image)

                    if self.show_annotations:
                        self.add_annotations_to_box()

                    self.error_label.value = ""
                else:
                    # If there's no pixel data, show an error
                    self.error_label.value = "The selected DICOM file has no pixel data to display."
                    self.page.update()

            except Exception as e:
                self.error_label.value = f"Error: {str(e)}"
                self.page.update()

    def detect_modality_defaults(self, dicom_data):
        # Example defaults based on modality
        if dicom_data.Modality == "CT":
            return 2891, 1425  # Common WW/WL for soft tissues in CT
        elif dicom_data.Modality == "MR":
            return None, None  # For MRI, we may want to calculate based on the actual data
        else:
            return None, None  # Other modalities can be handled similarly

                
    def show_image_from_file(self, file_path):
        # Find the index of the file path in the original file_paths list
        if file_path in self.file_paths:
            self.current_index = self.file_paths.index(file_path)
            self.load_dicom_image()

    def display_image(self, image_array):
        """Display the image after applying window-level adjustments and draw measurement or drawing lines if needed."""
        # Apply window width and window level adjustments
        adjusted_image = self.apply_window_level(image_array, self.window_width, self.window_level)

        # Check if the image is grayscale (2D), convert it to 3-channel (BGR) if needed
        if len(adjusted_image.shape) == 2:  # Grayscale image
            adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)

        # Get the display area size
        display_width, display_height = 800, 800

        # Calculate the aspect ratio of the original image
        image_height, image_width = adjusted_image.shape[:2]
        image_aspect_ratio = image_width / image_height

        # Determine the new dimensions while maintaining the aspect ratio
        if display_width / display_height > image_aspect_ratio:
            new_height = int(display_height * self.scale_factor)
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_width = int(display_width * self.scale_factor)
            new_height = int(new_width / image_aspect_ratio)

        # Make sure the new dimensions do not go below 1
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        # Resize the image to the new dimensions
        resized_image = cv2.resize(adjusted_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create an empty canvas with the size of the display area (3 channels for color)
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Calculate the position to place the image on the canvas, considering drag offset
        start_x = int((display_width - new_width) // 2 + self.drag_offset[0])
        start_y = int((display_height - new_height) // 2 + self.drag_offset[1])
        end_x = start_x + new_width
        end_y = start_y + new_height

        # Calculate the canvas and image regions to copy
        canvas_start_x = max(start_x, 0)
        canvas_start_y = max(start_y, 0)
        canvas_end_x = min(end_x, display_width)
        canvas_end_y = min(end_y, display_height)

        image_start_x = max(-start_x, 0)
        image_start_y = max(-start_y, 0)
        image_end_x = image_start_x + (canvas_end_x - canvas_start_x)
        image_end_y = image_start_y + (canvas_end_y - canvas_start_y)

        # Place the resized and possibly clipped image onto the canvas
        canvas[canvas_start_y:canvas_end_y, canvas_start_x:canvas_end_x] = resized_image[image_start_y:image_end_y, image_start_x:image_end_x]

        # Overlay the drawing buffer if it exists
        if self.drawing_buffer is not None:
            # Resize the drawing buffer to match the new dimensions
            drawing_resized = cv2.resize(self.drawing_buffer, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Clip the drawing_resized to match the displayed image region
            drawing_clipped = drawing_resized[image_start_y:image_end_y, image_start_x:image_end_x]
            canvas_region = canvas[canvas_start_y:canvas_end_y, canvas_start_x:canvas_end_x]

            # Ensure the shapes match
            if canvas_region.shape == drawing_clipped.shape:
                # Overlay the drawing buffer on top of the image
                canvas[canvas_start_y:canvas_end_y, canvas_start_x:canvas_end_x] = cv2.addWeighted(
                    canvas_region,
                    1,
                    drawing_clipped,
                    1,
                    0
                )
            else:
                print("Shapes do not match, skipping addWeighted operation.")
                print("Canvas region shape:", canvas_region.shape)
                print("Drawing clipped shape:", drawing_clipped.shape)

        # Handle measurement lines correctly based on zoom/pan
        if self.measurements:
            for (pt1, pt2) in self.measurements:
                # Scale the points to the resized image
                x_scale = new_width / image_width
                y_scale = new_height / image_height

                scaled_first_point = (
                    int(pt1[0] * x_scale + start_x),
                    int(pt1[1] * y_scale + start_y)
                )
                scaled_second_point = (
                    int(pt2[0] * x_scale + start_x),
                    int(pt2[1] * y_scale + start_y)
                )

                # Draw the line between the points
                cv2.line(canvas, scaled_first_point, scaled_second_point, (0, 255, 0), 2)

                # Calculate the distance between the points in pixels
                distance_pixels = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

                # If pixel spacing is available, convert to millimeters
                if hasattr(self, 'pixel_spacing') and self.pixel_spacing:
                    distance_mm = distance_pixels * self.pixel_spacing
                    text = f"{distance_mm:.2f} mm"
                else:
                    text = f"{distance_pixels:.2f} px"

                # Draw the distance text near the second point
                cv2.putText(canvas, text, (scaled_second_point[0] + 10, scaled_second_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If a temporary measurement is in progress
        if self.is_measurement_mode and self.first_point and self.second_point:
            # Scale the points to the resized image
            x_scale = new_width / image_width
            y_scale = new_height / image_height

            scaled_first_point = (
                int(self.first_point[0] * x_scale + start_x),
                int(self.first_point[1] * y_scale + start_y)
            )
            scaled_second_point = (
                int(self.second_point[0] * x_scale + start_x),
                int(self.second_point[1] * y_scale + start_y)
            )

            # Draw the line between the points
            cv2.line(canvas, scaled_first_point, scaled_second_point, (0, 255, 0), 2)

            # Calculate the distance between the points in pixels
            distance_pixels = math.sqrt((self.second_point[0] - self.first_point[0])**2 + (self.second_point[1] - self.first_point[1])**2)

            # If pixel spacing is available, convert to millimeters
            if hasattr(self, 'pixel_spacing') and self.pixel_spacing:
                distance_mm = distance_pixels * self.pixel_spacing
                text = f"{distance_mm:.2f} mm"
            else:
                text = f"{distance_pixels:.2f} px"

            # Draw the distance text near the second point
            cv2.putText(canvas, text, (scaled_second_point[0] + 10, scaled_second_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert the image to base64
        _, buf = cv2.imencode('.jpg', canvas)
        image_base64 = base64.b64encode(buf).decode('utf-8')

        # Update the image display in Flet
        self.image_display.src_base64 = image_base64

        # Update WL and WW labels with only one decimal place
        self.wl_label.value = f"Window Level: {self.window_level:.1f}"
        self.ww_label.value = f"Window Width: {self.window_width:.1f}"

        self.page.update()

    def add_annotations_to_box(self):
        """Move DICOM patient information to the annotation box."""
        self.annotation_box.controls.clear()  # Clear previous annotations

        if self.file_paths and 0 <= self.current_index < len(self.file_paths):
            try:
                # Load the DICOM data from the current file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])

                # Extract patient information for annotation
                patient_name = getattr(dicom_data, 'PatientName', 'Unknown Patient')
                patient_id = getattr(dicom_data, 'PatientID', 'Unknown ID')
                patient_sex = getattr(dicom_data, 'PatientSex', 'Unknown Sex')
                study_description = getattr(dicom_data, 'StudyDescription', 'Unknown Study')
                series_description = getattr(dicom_data, 'SeriesDescription', 'Unknown Series')
                study_date = getattr(dicom_data, 'StudyDate', 'Unknown Date')
                study_time = getattr(dicom_data, 'StudyTime', 'Unknown Time')

                # Format the date and time
                if len(study_date) == 8:
                    formatted_date = f"{study_date[6:8]}/{study_date[4:6]}/{study_date[0:4]}"
                else:
                    formatted_date = "Unknown Date"
                if len(study_time) >= 6:
                    formatted_time = f"{study_time[0:2]}:{study_time[2:4]}:{study_time[4:6]}"
                else:
                    formatted_time = "Unknown Time"

                # Create a list of patient information lines
                annotation_texts = [
                    f"Patient Name: {patient_name}",
                    f"Patient ID: {patient_id}",
                    f"Patient Sex: {patient_sex}",
                    f"Study Description: {study_description}",
                    f"Series Description: {series_description}",
                    f"Date: {formatted_date} {formatted_time}"
                ]

                # Add each line of the patient information to the annotation box
                for line in annotation_texts:
                    self.annotation_box.controls.append(ft.Text(line, size=14))

            except Exception as e:
                self.annotation_box.controls.append(ft.Text(f"Error loading annotations: {str(e)}"))

        else:
            # If no DICOM files are loaded, display a message
            self.annotation_box.controls.append(ft.Text("No DICOM file loaded."))

        self.page.update()


    def apply_window_level(self, image, window_width, window_level):
        # Convert window width and level to float
        window_width = float(window_width)
        window_level = float(window_level)

        # Calculate the minimum and maximum pixel values based on window width and level
        min_pixel_value = window_level - (window_width / 2)
        max_pixel_value = window_level + (window_width / 2)

        # Apply the windowing logic: image intensity is clipped to the min and max pixel values
        adjusted_image = np.clip(image, min_pixel_value, max_pixel_value)

        # Normalize the image to the 0-255 range
        adjusted_image = (adjusted_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
        adjusted_image = (adjusted_image * 255.0).astype(np.uint8)

        return adjusted_image


    def show_next(self, e):
        if self.file_paths and self.current_index < len(self.file_paths) - 1:
            self.current_index += 1
            self.load_dicom_image()

    def show_previous(self, e):
        if self.file_paths and self.current_index > 0:
            self.current_index -= 1
            self.load_dicom_image()

    def zoom_in(self, e):
        self.scale_factor *= 1.2  # Increase the scale factor to zoom in
        self.load_dicom_image()  # Reload the image to apply the zoom

    def zoom_out(self, e):
        self.scale_factor /= 1.2  # Decrease the scale factor to zoom out
        self.load_dicom_image()  # Reload the image to apply the zoom

    def rotate_image(self, clockwise=True):
        if self.file_paths:
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])

                # Update rotation angle (rotate by 90 degrees in the specified direction)
                rotation_direction = 1 if clockwise else -1
                self.rotation_angle = (self.rotation_angle + rotation_direction * 90) % 360

                # Rotate the image
                rotated_image = np.rot90(dicom_data.pixel_array, k=self.rotation_angle // 90)

                # Display the rotated image using OpenCV
                self.display_image(rotated_image)
                self.cached_image = rotated_image
                self.error_label.value = ""
                self.page.update()
            except Exception as e:
                # Display an error message if there is an issue
                self.error_label.value = f"Error: {str(e)}"
                self.page.update()

    def flip_image(self, horizontal=False, vertical=False):
        if self.file_paths:
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])

                # Toggle flip states
                if horizontal:
                    self.is_flipped_horizontal = not self.is_flipped_horizontal
                elif vertical:
                    self.is_flipped_vertical = not self.is_flipped_vertical

                # Apply both horizontal and vertical flips if needed
                flipped_image = dicom_data.pixel_array
                if self.is_flipped_horizontal:
                    flipped_image = np.fliplr(flipped_image)
                if self.is_flipped_vertical:
                    flipped_image = np.flipud(flipped_image)

                # Display the flipped image using OpenCV
                self.display_image(flipped_image)
                self.cached_image = flipped_image
                self.error_label.value = ""
                self.page.update()
            except Exception as e:
                # Display an error message if there is an issue
                self.error_label.value = f"Error: {str(e)}"
                self.page.update()

    def on_pan_start(self, e):
        self.start_drag_x = e.local_x
        self.start_drag_y = e.local_y

        x, y = e.local_x, e.local_y

        # Get the display area size (this is the fixed size of the container)
        display_width, display_height = 800, 800

        # Calculate the aspect ratio of the original image
        image_height, image_width = self.cached_image.shape[:2]
        image_aspect_ratio = image_width / image_height

        # Determine the scaled dimensions while maintaining the aspect ratio
        if display_width / display_height > image_aspect_ratio:
            new_height = int(display_height * self.scale_factor)
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_width = int(display_width * self.scale_factor)
            new_height = int(new_width / image_aspect_ratio)

        # Calculate the position to place the image on the canvas, considering drag offset
        start_x = int((display_width - new_width) // 2 + self.drag_offset[0])
        start_y = int((display_height - new_height) // 2 + self.drag_offset[1])

        if self.is_drawing_mode:
            # Adjust the click coordinates based on scaling and pan
            adjusted_x = (x - start_x) * (image_width / new_width)
            adjusted_y = (y - start_y) * (image_height / new_height)
            self.drawing_points = [(adjusted_x, adjusted_y)]
        elif self.is_measurement_mode:
            # Adjust the coordinates to original image coordinates
            adjusted_x = (x - start_x) * (image_width / new_width)
            adjusted_y = (y - start_y) * (image_height / new_height)
            if self.first_point is None:
                self.first_point = (adjusted_x, adjusted_y)
            else:
                self.second_point = (adjusted_x, adjusted_y)
                # Append the measurement to the list
                self.measurements.append((self.first_point, self.second_point))
                # Reset measurement points
                self.first_point = None
                self.second_point = None
                # Update display to show the new measurement
                self.display_image(self.cached_image)

        self.page.update()

    def on_pan_update(self, e):
        dx = e.local_x - self.start_drag_x
        dy = e.local_y - self.start_drag_y

        pan_scale_factor = 0.1  # Adjusted pan speed

        # Get the original image dimensions
        image_height, image_width = self.cached_image.shape[:2]
        image_aspect_ratio = image_width / image_height

        # Get the display area size
        display_width, display_height = 800, 800

        # Determine the scaled dimensions while maintaining the aspect ratio
        if display_width / display_height > image_aspect_ratio:
            new_height = int(display_height * self.scale_factor)
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_width = int(display_width * self.scale_factor)
            new_height = int(new_width / image_aspect_ratio)

        # Calculate the position to place the image on the canvas, considering drag offset
        start_x = int((display_width - new_width) // 2 + self.drag_offset[0])
        start_y = int((display_height - new_height) // 2 + self.drag_offset[1])

        if self.is_drawing_mode:
            # Adjust the coordinates to account for zoom and pan
            adjusted_x = (e.local_x - start_x) * (image_width / new_width)
            adjusted_y = (e.local_y - start_y) * (image_height / new_height)
            # Append the new point to the list of drawing points
            self.drawing_points.append((adjusted_x, adjusted_y))
            # Draw the line between the last two points
            if len(self.drawing_points) > 1:
                start_point = self.drawing_points[-2]
                end_point = self.drawing_points[-1]
                self.draw_line(start_point, end_point)
        elif self.is_dragging_mode:
            # Adjust the offset and limit the panning based on the image size
            self.drag_offset = (self.drag_offset[0] + dx * pan_scale_factor, self.drag_offset[1] + dy * pan_scale_factor)
            # Re-render the image for panning
            self.display_image(self.cached_image)
        elif self.is_adjusting_wl_ww:
            # Adjust WW/WL values
            wl_ww_scale_factor = 0.1  # Defines how fast WW/WL adjustments happen
            self.window_width += dx * wl_ww_scale_factor
            self.window_level -= dy * wl_ww_scale_factor
            # Re-render the image after adjusting WW/WW
            self.display_image(self.cached_image)
        elif self.is_measurement_mode and self.first_point is not None:
            # Live measurement update
            adjusted_x = (e.local_x - start_x) * (image_width / new_width)
            adjusted_y = (e.local_y - start_y) * (image_height / new_height)
            self.second_point = (adjusted_x, adjusted_y)
            # Update display to show the temporary measurement
            self.display_image(self.cached_image)

        # Update the starting point for the next pan update
        self.start_drag_x = e.local_x
        self.start_drag_y = e.local_y

        # Update the page to reflect the changes
        self.page.update()

    def redraw_drawing_buffer(self):
        """Redraws the drawing buffer from the list of drawn lines for the current image."""
        # Initialize the drawing buffer with 3 channels
        self.drawing_buffer = np.zeros((*self.cached_image.shape[:2], 3), dtype=np.uint8)

        # Draw all lines in the drawn_lines list for the current image
        for line in self.drawn_lines:
            start_point, end_point = line
            # Convert the points to integer tuples
            start_point_int = (int(start_point[0]), int(start_point[1]))
            end_point_int = (int(end_point[0]), int(end_point[1]))
            # Draw the line on the drawing buffer
            cv2.line(self.drawing_buffer, start_point_int, end_point_int, (0, 255, 0), 2)  # Green line

    def draw_line(self, start_point, end_point):
        """Draws a line between two points on the drawing buffer."""
        # Ensure the current image index has an entry in the dictionary
        if self.current_index not in self.image_drawn_lines:
            self.image_drawn_lines[self.current_index] = []

        # Add the line to the list of drawn lines for the current image
        self.image_drawn_lines[self.current_index].append((start_point, end_point))

        # Update the local drawn_lines list
        self.drawn_lines = self.image_drawn_lines[self.current_index]

        # Redraw the drawing buffer from scratch
        self.redraw_drawing_buffer()

        # Update the display with the new drawing
        self.display_image(self.cached_image)

    def on_pan_end(self, e):
        if self.is_drawing_mode:
            self.drawing_points = []  # Clear points after drawing ends
        elif self.is_measurement_mode and self.first_point is not None and self.second_point is not None:
            # Append the measurement to the list
            self.measurements.append((self.first_point, self.second_point))
            # Reset measurement points
            self.first_point = None
            self.second_point = None
            # Update display to show the finalized measurement
            self.display_image(self.cached_image)
        # Reset the drag variables when the user finishes the gesture
        self.start_drag_x = 0
        self.start_drag_y = 0


    def draw_freehand_line(self):
        # Start with the cached image to avoid interference
        self.drawing_buffer = self.cached_image.copy()

        # Convert to BGR if necessary
        if len(self.drawing_buffer.shape) == 2:  # If grayscale
            self.drawing_buffer = cv2.cvtColor(self.drawing_buffer, cv2.COLOR_GRAY2BGR)

        # Ensure there are at least two points to draw a line
        if len(self.drawing_points) < 2:
            return

        # Draw lines between consecutive points on the drawing buffer
        for i in range(1, len(self.drawing_points)):
            start_point = self.drawing_points[i - 1]
            end_point = self.drawing_points[i]

            # Draw the line between the points on the drawing buffer
            cv2.line(self.drawing_buffer, start_point, end_point, (0, 255, 0), 2)  # Green for drawing

        # Update the cached image with the drawing
        self.cached_image = self.drawing_buffer.copy()

        # Display the image with drawing
        self.display_image(self.drawing_buffer)
        self.page.update()

       
    def toggle_drag_mode(self, e):
        self.is_dragging_mode = not self.is_dragging_mode
        if self.is_dragging_mode:
            self.is_measurement_mode = False  # Disable measurement when panning is on
            self.toggle_drag_button.style = ft.ButtonStyle(bgcolor=ft.colors.GREEN)
            self.error_label.value = "Pan Mode Active"
        else:
            self.toggle_drag_button.style = ft.ButtonStyle(bgcolor=None)  # Reset to default
            self.error_label.value = "Pan Mode Inactive"
        self.page.update()

    def toggle_wl_ww_mode(self, e):
        self.is_adjusting_wl_ww = not self.is_adjusting_wl_ww
        if self.is_adjusting_wl_ww:
            self.toggle_wl_ww_button.style = ft.ButtonStyle(bgcolor=ft.colors.GREEN)
            self.error_label.value = "WL/WW Adjust Mode Active"
        else:
            self.toggle_wl_ww_button.style = ft.ButtonStyle(bgcolor=None)  # Reset to default
            self.error_label.value = "WL/WW Adjust Mode Inactive"
        self.page.update()
    
    key = "1234567890123456" 

    sbox = {
        0x00: 0x63, 0x01: 0x7C, 0x02: 0x77, 0x03: 0x7B, 0x04: 0xF2, 0x05: 0x6B, 0x06: 0x6F, 0x07: 0xC5,
        0x08: 0x30, 0x09: 0x01, 0x0A: 0x67, 0x0B: 0x2B, 0x0C: 0xFE, 0x0D: 0xD7, 0x0E: 0xAB, 0x0F: 0x76,
        0x10: 0xCA, 0x11: 0x82, 0x12: 0xC9, 0x13: 0x7D, 0x14: 0xFA, 0x15: 0x59, 0x16: 0x47, 0x17: 0xF0,
        0x18: 0xAD, 0x19: 0xD4, 0x1A: 0xA2, 0x1B: 0xAF, 0x1C: 0x9C, 0x1D: 0xA4, 0x1E: 0x72, 0x1F: 0xC0,
        0x20: 0xB7, 0x21: 0xFD, 0x22: 0x93, 0x23: 0x26, 0x24: 0x36, 0x25: 0x3F, 0x26: 0xF7, 0x27: 0xCC,
        0x28: 0x34, 0x29: 0xA5, 0x2A: 0xE5, 0x2B: 0xF1, 0x2C: 0x71, 0x2D: 0xD8, 0x2E: 0x31, 0x2F: 0x15,
        0x30: 0x04, 0x31: 0xC7, 0x32: 0x23, 0x33: 0xC3, 0x34: 0x18, 0x35: 0x96, 0x36: 0x05, 0x37: 0x9A,
        0x38: 0x07, 0x39: 0x12, 0x3A: 0x80, 0x3B: 0xE2, 0x3C: 0xEB, 0x3D: 0x27, 0x3E: 0xB2, 0x3F: 0x75,
        0x40: 0x09, 0x41: 0x83, 0x42: 0x2C, 0x43: 0x1A, 0x44: 0x1B, 0x45: 0x6E, 0x46: 0x5A, 0x47: 0xA0,
        0x48: 0x52, 0x49: 0x3B, 0x4A: 0xD6, 0x4B: 0xB3, 0x4C: 0x29, 0x4D: 0xE3, 0x4E: 0x2F, 0x4F: 0x84,
        0x50: 0x53, 0x51: 0xD1, 0x52: 0x00, 0x53: 0xED, 0x54: 0x20, 0x55: 0xFC, 0x56: 0xB1, 0x57: 0x5B,
        0x58: 0x6A, 0x59: 0xCB, 0x5A: 0xBE, 0x5B: 0x39, 0x5C: 0x4A, 0x5D: 0x4C, 0x5E: 0x58, 0x5F: 0xCF,
        0x60: 0xD0, 0x61: 0xEF, 0x62: 0xAA, 0x63: 0xFB, 0x64: 0x43, 0x65: 0x4D, 0x66: 0x33, 0x67: 0x85,
        0x68: 0x45, 0x69: 0xF9, 0x6A: 0x02, 0x6B: 0x7F, 0x6C: 0x50, 0x6D: 0x3C, 0x6E: 0x9F, 0x6F: 0xA8,
        0x70: 0x51, 0x71: 0xA3, 0x72: 0x40, 0x73: 0x8F, 0x74: 0x92, 0x75: 0x9D, 0x76: 0x38, 0x77: 0xF5,
        0x78: 0xBC, 0x79: 0xB6, 0x7A: 0xDA, 0x7B: 0x21, 0x7C: 0x10, 0x7D: 0xFF, 0x7E: 0xF3, 0x7F: 0xD2,
        0x80: 0xCD, 0x81: 0x0C, 0x82: 0x13, 0x83: 0xEC, 0x84: 0x5F, 0x85: 0x97, 0x86: 0x44, 0x87: 0x17,
        0x88: 0xC4, 0x89: 0xA7, 0x8A: 0x7E, 0x8B: 0x3D, 0x8C: 0x64, 0x8D: 0x5D, 0x8E: 0x19, 0x8F: 0x73,
        0x90: 0x60, 0x91: 0x81, 0x92: 0x4F, 0x93: 0xDC, 0x94: 0x22, 0x95: 0x2A, 0x96: 0x90, 0x97: 0x88,
        0x98: 0x46, 0x99: 0xEE, 0x9A: 0xB8, 0x9B: 0x14, 0x9C: 0xDE, 0x9D: 0x5E, 0x9E: 0x0B, 0x9F: 0xDB,
        0xA0: 0xE0, 0xA1: 0x32, 0xA2: 0x3A, 0xA3: 0x0A, 0xA4: 0x49, 0xA5: 0x06, 0xA6: 0x24, 0xA7: 0x5C,
        0xA8: 0xC2, 0xA9: 0xD3, 0xAA: 0xAC, 0xAB: 0x62, 0xAC: 0x91, 0xAD: 0x95, 0xAE: 0xE4, 0xAF: 0x79,
        0xB0: 0xE7, 0xB1: 0xC8, 0xB2: 0x37, 0xB3: 0x6D, 0xB4: 0x8D, 0xB5: 0xD5, 0xB6: 0x4E, 0xB7: 0xA9,
        0xB8: 0x6C, 0xB9: 0x56, 0xBA: 0xF4, 0xBB: 0xEA, 0xBC: 0x65, 0xBD: 0x7A, 0xBE: 0xAE, 0xBF: 0x08,
        0xC0: 0xBA, 0xC1: 0x78, 0xC2: 0x25, 0xC3: 0x2E, 0xC4: 0x1C, 0xC5: 0xA6, 0xC6: 0xB4, 0xC7: 0xC6,
        0xC8: 0xE8, 0xC9: 0xDD, 0xCA: 0x74, 0xCB: 0x1F, 0xCC: 0x4B, 0xCD: 0xBD, 0xCE: 0x8B, 0xCF: 0x8A,
        0xD0: 0x70, 0xD1: 0x3E, 0xD2: 0xB5, 0xD3: 0x66, 0xD4: 0x48, 0xD5: 0x03, 0xD6: 0xF6, 0xD7: 0x0E,
        0xD8: 0x61, 0xD9: 0x35, 0xDA: 0x57, 0xDB: 0xB9, 0xDC: 0x86, 0xDD: 0xC1, 0xDE: 0x1D, 0xDF: 0x9E,
        0xE0: 0xE1, 0xE1: 0xF8, 0xE2: 0x98, 0xE3: 0x11, 0xE4: 0x69, 0xE5: 0xD9, 0xE6: 0x8E, 0xE7: 0x94,
        0xE8: 0x9B, 0xE9: 0x1E, 0xEA: 0x87, 0xEB: 0xE9, 0xEC: 0xCE, 0xED: 0x55, 0xEE: 0x28, 0xEF: 0xDF,
        0xF0: 0x8C, 0xF1: 0xA1, 0xF2: 0x89, 0xF3: 0x0D, 0xF4: 0xBF, 0xF5: 0xE6, 0xF6: 0x42, 0xF7: 0x68,
        0xF8: 0x41, 0xF9: 0x99, 0xFA: 0x2D, 0xFB: 0x0F, 0xFC: 0xB0, 0xFD: 0x54, 0xFE: 0xBB, 0xFF: 0x16
    }

    
    def process_encryption(self, e, key=None):
        selected_mode = self.mode_selector.value  # Get the selected mode (Encryption or Decryption)
        process_patient_info = self.patient_info_checkbox.value  # Check if patient info is selected
        process_image = self.image_checkbox.value  # Check if image data is selected

        if not self.file_paths or self.current_index >= len(self.file_paths):
            self.error_label.value = "No DICOM file loaded or invalid index."
            self.page.update()
            return

        # If the key is not passed as an argument, use the default key
        if key is None:
            key = "1234567890123456"  # You can replace this with a dynamically entered key or input from the user

        # Get the file path of the currently displayed DICOM file
        dicom_file_path = self.file_paths[self.current_index]

        # Process based on the selected mode
        if selected_mode == "Encryption":
            self.encrypt_dicom_patient_info_and_image(
                dicom_file_path, dicom_file_path, key,
                encrypt_patient_info=process_patient_info, encrypt_image=process_image
            )
            self.error_label.value = f"Encryption complete. File updated."
        elif selected_mode == "Decryption":
            self.decrypt_dicom_patient_info_and_image(
                dicom_file_path, dicom_file_path, key,
                decrypt_patient_info=process_patient_info, decrypt_image=process_image
            )
            self.error_label.value = f"Decryption complete. File updated."
        else:
            self.error_label.value = "Invalid mode selected."

        # Reload the DICOM image to display the updated image
        self.load_dicom_image()

        # Regenerate the thumbnail after encryption or decryption
        self.load_thumbnails()  # Reload thumbnails to reflect changes

        self.page.update()
  
    def xor(self, data_bytes, key_bytes):
        return bytes([b1 ^ b2 for b1, b2 in zip(data_bytes, key_bytes)])

    def subbytes(self,data, sbox):   
        return bytes([sbox.get(byte) for byte in data])

    def shiftrows(self,data):
        order = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]    
        shiftrows_result = bytes([data[i] for i in order])    
        return shiftrows_result

    def multiply(self,a, b):
        if a == 0x02:
            result = b << 1
            if result & 0x100: 
                result ^= 0x1B  
            return result & 0xFF 

        elif a == 0x03:
            result = (b << 1) ^ b
            if result & 0x100:  
                result ^= 0x1B 
            return result & 0xFF

    def mixcolumns(self, shiftrows_result):
        out1 = self.multiply(0x02, shiftrows_result[0]) ^ self.multiply(0x03, shiftrows_result[1]) ^ shiftrows_result[2] ^ shiftrows_result[3]
        out2 = shiftrows_result[0] ^ self.multiply(0x02, shiftrows_result[1]) ^ self.multiply(0x03, shiftrows_result[2]) ^ shiftrows_result[3]
        out3 = shiftrows_result[0] ^ shiftrows_result[1] ^ self.multiply(0x02, shiftrows_result[2]) ^ self.multiply(0x03, shiftrows_result[3])
        out4 = self.multiply(0x03, shiftrows_result[0]) ^ shiftrows_result[1] ^ shiftrows_result[2] ^ self.multiply(0x02, shiftrows_result[3])
        out5 = self.multiply(0x02, shiftrows_result[4]) ^ self.multiply(0x03, shiftrows_result[5]) ^ shiftrows_result[6] ^ shiftrows_result[7]
        out6 = shiftrows_result[4] ^ self.multiply(0x02, shiftrows_result[5]) ^ self.multiply(0x03, shiftrows_result[6]) ^ shiftrows_result[7]
        out7 = shiftrows_result[4] ^ shiftrows_result[5] ^ self.multiply(0x02, shiftrows_result[6]) ^ self.multiply(0x03, shiftrows_result[7])
        out8 = self.multiply(0x03, shiftrows_result[4]) ^ shiftrows_result[5] ^ shiftrows_result[6] ^ self.multiply(0x02, shiftrows_result[7])
        out9 = self.multiply(0x02, shiftrows_result[8]) ^ self.multiply(0x03, shiftrows_result[9]) ^ shiftrows_result[10] ^ shiftrows_result[11]
        out10 = shiftrows_result[8] ^ self.multiply(0x02, shiftrows_result[9]) ^ self.multiply(0x03, shiftrows_result[10]) ^ shiftrows_result[11]
        out11 = shiftrows_result[8] ^ shiftrows_result[9] ^ self.multiply(0x02, shiftrows_result[10]) ^ self.multiply(0x03, shiftrows_result[11])
        out12 = self.multiply(0x03, shiftrows_result[8]) ^ shiftrows_result[9] ^ shiftrows_result[10] ^ self.multiply(0x02, shiftrows_result[11])
        out13 = self.multiply(0x02, shiftrows_result[12]) ^ self.multiply(0x03, shiftrows_result[13]) ^ shiftrows_result[14] ^ shiftrows_result[15]
        out14 = shiftrows_result[12] ^ self.multiply(0x02, shiftrows_result[13]) ^ self.multiply(0x03, shiftrows_result[14]) ^ shiftrows_result[15]
        out15 = shiftrows_result[12] ^ shiftrows_result[13] ^ self.multiply(0x02, shiftrows_result[14]) ^ self.multiply(0x03, shiftrows_result[15])
        out16 = self.multiply(0x03, shiftrows_result[12]) ^ shiftrows_result[13] ^ shiftrows_result[14] ^ self.multiply(0x02, shiftrows_result[15])
        mixcolumn_result = bytes([out1, out2, out3, out4, out5, out6, out7, out8,
                                out9, out10, out11, out12, out13, out14, out15, out16])
        return mixcolumn_result


    def subkeygen(self, key_bytes, round_num):
        def rotword(word):
            order = [1, 2, 3, 0] 
            return bytes([word[i] for i in order])

        rcon_values = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
        rcon = rcon_values[round_num-1]
        
        word3 = key_bytes[-4:]
        rotated_word = rotword(word3)
        substituted_word = self.subbytes(rotated_word, self.sbox)  # Using self.sbox and self.subbytes
        xorrcon = bytes([substituted_word[0] ^ rcon]) + substituted_word[1:]
        
        word4 = self.xor(xorrcon, key_bytes[:4])  # Using self.xor
        word5 = self.xor(word4, key_bytes[4:8])
        word6 = self.xor(word5, key_bytes[8:12])
        word7 = self.xor(word6, key_bytes[12:16])
        
        gensubkey = word4 + word5 + word6 + word7
        return gensubkey

    # ทำลูปปปป
    def perform_aes_encryption(self, block, key_bytes, rounds=10):
        current_state = self.xor(block, key_bytes)  # Ensure only two arguments

        for round in range(1, rounds):
            current_state = self.subbytes(current_state, self.sbox)
            current_state = self.shiftrows(current_state)
            if round < rounds:
                current_state = self.mixcolumns(current_state)  # No need to pass multiply as an argument
            key_bytes = self.subkeygen(key_bytes, round)
            current_state = self.xor(current_state, key_bytes)

        current_state = self.subbytes(current_state, self.sbox)
        current_state = self.shiftrows(current_state)
        key_bytes = self.subkeygen(key_bytes, rounds)
        ciphertext = self.xor(current_state, key_bytes)  # Ensure only two arguments

        return ciphertext

    def encrypt(self, plaintext, key_bytes, rounds=10):
        blocksize = 16
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()  # Ensure plaintext is bytes
        
        padlength = blocksize - (len(plaintext) % blocksize)
        padding = bytes([padlength]) * padlength
        paddeddata = plaintext + padding
        data_bytes = paddeddata

        num_blocks = len(data_bytes) // blocksize
        ciphertext_full = bytes()

        for blocknum in range(num_blocks):
            block = data_bytes[blocknum * blocksize:(blocknum + 1) * blocksize]
            # Call self.perform_aes_encryption instead of passing it as an argument
            ciphertext_block = self.perform_aes_encryption(block, key_bytes, rounds)
            ciphertext_full += ciphertext_block

        return ciphertext_full

    def encrypt_dicom_patient_info_and_image(self, dicom_file_path, output_file_path, key, encrypt_patient_info=True, encrypt_image=True):
        # Read the DICOM file
        dicom_data = dcmread(dicom_file_path)

        # Convert the key to bytes
        key_bytes = key.encode()

        if encrypt_patient_info:
            if hasattr(dicom_data, 'PatientID'):
                # Encrypt the Patient ID
                patient_id = dicom_data.PatientID.encode()  # Convert string to bytes
                encrypted_patient_id = self.encrypt(patient_id, key_bytes)  # Encrypt
                dicom_data.PatientID = encrypted_patient_id.hex()  # Store as hex

            if hasattr(dicom_data, 'PatientName'):
                # Encrypt the Patient Name
                patient_name = dicom_data.PatientName.encode()  # Convert string to bytes
                encrypted_patient_name = self.encrypt(patient_name, key_bytes)  # Encrypt
                dicom_data.PatientName = encrypted_patient_name.hex()  # Store as hex

            if hasattr(dicom_data, 'PatientSex'):
                # Encrypt the Patient Sex
                patient_sex = dicom_data.PatientSex.encode()  # Convert string to bytes
                encrypted_patient_sex = self.encrypt(patient_sex, key_bytes)  # Encrypt
                dicom_data.PatientSex = encrypted_patient_sex.hex()  # Store as hex

        if encrypt_image and hasattr(dicom_data, 'PixelData'):
            # Encrypt the PixelData (raw image data)
            pixel_data = dicom_data.PixelData  # Raw pixel data as bytes
            encrypted_pixel_data = self.encrypt(pixel_data, key_bytes)  # Encrypt pixel data
            dicom_data.PixelData = encrypted_pixel_data  # Store encrypted bytes in PixelData

        # Save the modified DICOM file with the encrypted fields
        try:
            dcmwrite(output_file_path, dicom_data)
            print(f"Encryption complete. File updated: {output_file_path}")
        except Exception as e:
            print(f"Error writing encrypted DICOM file: {str(e)}")


    def decrypt(self, ciphertext, key, is_pixel_data=False):
        blocksize = 16

        # Ensure key is encoded to bytes (only if it’s not already bytes)
        if isinstance(key, str):
            key_bytes = key.encode()
        else:
            key_bytes = key

        # Use PyCryptodome's AES for decryption
        aes_cipher = PyCryptoAES.new(key_bytes, PyCryptoAES.MODE_ECB)  # Use ECB mode for simplicity
        decrypted_data = aes_cipher.decrypt(ciphertext)

        # Only unpad if the data is not PixelData (PixelData is binary and should not be padded/unpadded)
        if not is_pixel_data:
            try:
                decrypted_data = unpad(decrypted_data, blocksize)
            except ValueError as e:
                raise ValueError("Incorrect decryption or padding.")

        return decrypted_data

    def decrypt_dicom_patient_info_and_image(self, dicom_file_path, output_file_path, key, decrypt_patient_info=True, decrypt_image=True):
        dicom_data = dcmread(dicom_file_path)
        key_bytes = key.encode()

        if decrypt_patient_info:
            if hasattr(dicom_data, 'PatientID'):
                try:
                    encrypted_patient_id = bytes.fromhex(dicom_data.PatientID)
                    decrypted_patient_id = self.decrypt(encrypted_patient_id, key_bytes)
                    dicom_data.PatientID = decrypted_patient_id.decode()
                except Exception as e:
                    print(f"Error decrypting patient information: {str(e)}")

            if hasattr(dicom_data, 'PatientName'):
                try:
                    encrypted_patient_name = bytes.fromhex(str(dicom_data.PatientName))  # Convert PersonName to string
                    decrypted_patient_name = self.decrypt(encrypted_patient_name, key_bytes)
                    dicom_data.PatientName = decrypted_patient_name.decode()
                except Exception as e:
                    print(f"Error decrypting patient name: {str(e)}")

            if hasattr(dicom_data, 'PatientSex'):
                try:
                    encrypted_patient_sex = bytes.fromhex(dicom_data.PatientSex)
                    decrypted_patient_sex = self.decrypt(encrypted_patient_sex, key_bytes)
                    dicom_data.PatientSex = decrypted_patient_sex.decode()
                except Exception as e:
                    print(f"Error decrypting patient sex: {str(e)}")

        if decrypt_image and hasattr(dicom_data, 'PixelData'):
            try:
                encrypted_pixel_data = dicom_data.PixelData
                decrypted_pixel_data = self.decrypt(encrypted_pixel_data, key_bytes, is_pixel_data=True)
                dicom_data.PixelData = decrypted_pixel_data
            except Exception as e:
                print(f"Error decrypting image data: {str(e)}")

        try:
            dcmwrite(output_file_path, dicom_data)
            print(f"Decryption complete. File updated: {output_file_path}")
        except Exception as e:
            print(f"Error writing decrypted DICOM file: {str(e)}")


def main(page: ft.Page):
    viewer = DICOMViewer(page)
    # Ensure the page receives keyboard events
    page.on_keyboard_event = viewer.on_keyboard_event
    page.update()

ft.app(target=main)

