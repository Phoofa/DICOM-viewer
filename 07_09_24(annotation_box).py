import os
import pydicom
import numpy as np
import flet as ft
import cv2
import base64

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
        self.show_annotations = False  # Variable to track if annotations should be shown
        self.page.window.maximized = True  # Updated to use the new way of setting the window maximized

        # Create UI elements
        self.create_ui()

    def create_ui(self):
        self.error_label = ft.Text("")
        self.image_display = ft.Image(width=700, height=700)  # Adjust the size here
        self.gesture_detector = ft.GestureDetector(
            content=ft.Container(
                content=self.image_display,
                width=1200,
                height=1200,
            ),
            on_pan_start=self.on_pan_start,
            on_pan_update=self.on_pan_update,
            on_pan_end=self.on_pan_end,
        )

        self.next_button = ft.ElevatedButton("Next", on_click=self.show_next)
        self.prev_button = ft.ElevatedButton("Previous", on_click=self.show_previous)
        self.zoom_in_button = ft.ElevatedButton("Zoom In", on_click=self.zoom_in)
        self.zoom_out_button = ft.ElevatedButton("Zoom Out", on_click=self.zoom_out)
        self.toggle_drag_button = ft.ElevatedButton("Toggle Drag Mode", on_click=self.toggle_drag_mode)
        self.toggle_wl_ww_button = ft.ElevatedButton("Toggle WL/WW Adjust Mode", on_click=self.toggle_wl_ww_mode)
        self.toggle_annotation_button = ft.ElevatedButton("Toggle Annotation", on_click=self.toggle_annotation)

        self.wl_label = ft.Text(f"Window Level: {self.window_level}")
        self.ww_label = ft.Text(f"Window Width: {self.window_width}")

        # Create file pickers
        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.page.overlay.append(self.file_picker)
        
        self.directory_picker = ft.FilePicker(on_result=self.on_directory_selected)
        self.page.overlay.append(self.directory_picker)

        # Create a menu bar with combined rotate/flip options
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

        # Add elements to the page
        self.page.add(
            ft.Row([
                ft.Container(
                    content=self.thumbnail_list,
                    width=200,
                    height=1200,
                ),
                ft.VerticalDivider(width=1, thickness=2),
                ft.Column([
                    ft.Row([
                        self.menubar,
                        self.prev_button,
                        self.next_button,
                        self.zoom_in_button,
                        self.zoom_out_button,
                        self.toggle_drag_button,
                        self.toggle_wl_ww_button,
                        self.toggle_annotation_button  # Add toggle annotation button
                    ], alignment=ft.MainAxisAlignment.START),  # Align buttons to the start
                    self.error_label,
                    ft.Container(
                        content=self.gesture_detector,
                        alignment=ft.alignment.center,
                        expand=True
                    ),
                    ft.Column([self.wl_label, self.ww_label]),
                ], expand=True),
                ft.VerticalDivider(width=1, thickness=2),
                ft.Container(content=self.annotation_box, width=300, height=1200)  # Annotation box on the right
            ], expand=True)
        )

    def toggle_annotation(self, e):
        """Toggle the visibility of annotations."""
        self.show_annotations = not self.show_annotations
        if self.show_annotations:
            self.add_annotations_to_box()  # Show the annotations
        else:
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
    
    def show_image_from_file(self, file_path):
        # Find the index of the file path in the original file_paths list
        if file_path in self.file_paths:
            self.current_index = self.file_paths.index(file_path)
            self.load_dicom_image()  # Load the selected DICOM image

    def load_thumbnails(self):
        self.thumbnail_list.controls.clear()
        self.series_file_mapping = []  # List to store file paths corresponding to thumbnails

        series_dict = {}

        for file_path in self.file_paths:
            dicom_data = pydicom.dcmread(file_path)

            # Check if the file belongs to a series
            if hasattr(dicom_data, 'SeriesInstanceUID'):
                series_uid = dicom_data.SeriesInstanceUID
                if series_uid not in series_dict:
                    series_dict[series_uid] = file_path

        # Create thumbnails for the first image in each series
        for idx, file_path in enumerate(series_dict.values()):
            dicom_data = pydicom.dcmread(file_path)
            image = dicom_data.pixel_array
            thumbnail = self.generate_thumbnail(image)

            # Extract metadata for the thumbnail
            series_date = getattr(dicom_data, 'SeriesDate', 'Unknown Date')
            series_time = getattr(dicom_data, 'SeriesTime', 'Unknown Time')
            series_description = getattr(dicom_data, 'SeriesDescription', 'Unknown Series')
            modality = getattr(dicom_data, 'Modality', 'Unknown Modality')

            # Format the date as DD/MM/YYYY
            if series_date != 'Unknown Date' and len(series_date) == 8:
                formatted_date = f"{series_date[6:8]}/{series_date[4:6]}/{series_date[0:4]}"
            else:
                formatted_date = "Unknown Date"

            # Format the time as HH:MM:SS
            if series_time != 'Unknown Time' and len(series_time) >= 6:
                formatted_time = f"{series_time[0:2]}:{series_time[2:4]}:{series_time[4:6]}"
            else:
                formatted_time = "Unknown Time"

            # Combine date and time
            date_time_text = f"Date: {formatted_date} {formatted_time}"

            # Create a layout for the thumbnail including metadata
            thumbnail_image = ft.Image(
                src_base64=thumbnail,
                width=150,
                height=150,
                fit=ft.ImageFit.CONTAIN,
                border_radius=ft.border_radius.all(10)
            )

            # Create text fields for the metadata
            series_description_text = ft.Text(f"Series: {series_description}", size=12)
            modality_text = ft.Text(f"Type: {modality}", size=12)

            # Container with image and metadata text
            thumbnail_container = ft.Container(
                content=ft.Column([
                    thumbnail_image,
                    ft.Text(date_time_text, size=12),  # Date and time text
                    series_description_text,
                    modality_text
                ]),
                on_click=lambda e, file_path=file_path: self.show_image_from_file(file_path),
                padding=ft.padding.all(10),
                border_radius=ft.border_radius.all(10),
                border=ft.border.all(1, ft.colors.WHITE),  # Set a border using border parameter
            )

            self.thumbnail_list.controls.append(thumbnail_container)
            self.series_file_mapping.append(file_path)  # Add the file path to the mapping list

        self.page.update()


    def load_dicom_image(self):
        if self.file_paths:
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])
                self.cached_image = dicom_data.pixel_array

                # Set window width and level
                if hasattr(dicom_data, 'WindowWidth') and hasattr(dicom_data, 'WindowCenter'):
                    self.window_width = dicom_data.WindowWidth[0] if isinstance(dicom_data.WindowWidth, pydicom.multival.MultiValue) else dicom_data.WindowWidth
                    self.window_level = dicom_data.WindowCenter[0] if isinstance(dicom_data.WindowCenter, pydicom.multival.MultiValue) else dicom_data.WindowCenter
                else:
                    # Compute default window width and level based on image statistics
                    min_val = np.min(self.cached_image)
                    max_val = np.max(self.cached_image)
                    self.window_width = max_val - min_val
                    self.window_level = (max_val + min_val) / 2

                self.dicom_data = dicom_data  # Save the DICOM data for annotations
                self.display_image(self.cached_image)

                if self.show_annotations:
                    self.add_annotations_to_box()  # Show annotations in the box

                # Clear error message
                self.error_label.value = ""
                self.page.update()
            except Exception as e:
                # Display an error message if there is an issue
                self.error_label.value = f"Error: {str(e)}"
                self.page.update()

    def display_image(self, image_array):
        # Apply window width and window level adjustments
        adjusted_image = self.apply_window_level(image_array, self.window_width, self.window_level)

        # Get the display area size
        display_width, display_height = 1200, 1200  # Assuming the display area is 1200x1200

        # Calculate the aspect ratio of the original image
        image_height, image_width = adjusted_image.shape
        image_aspect_ratio = image_width / image_height

        # Determine the new dimensions while maintaining the aspect ratio
        if display_width / display_height > image_aspect_ratio:
            new_height = int(display_height * self.scale_factor)
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_width = int(display_width * self.scale_factor)
            new_height = int(new_width / image_aspect_ratio)

        # Resize the image to the new dimensions
        resized_image = cv2.resize(adjusted_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create an empty canvas with the size of the display area
        canvas = np.zeros((display_height, display_width), dtype=np.uint8)

        # Calculate the position to center the image on the canvas
        start_x = int((display_width - new_width) // 2 + self.drag_offset[0])
        start_y = int((display_height - new_height) // 2 + self.drag_offset[1])

        # Clip the start position if the dragged image is out of bounds
        start_x = max(start_x, -new_width + 1)  # Ensures at least 1 pixel of the image remains visible
        start_y = max(start_y, -new_height + 1)

        end_x = int(start_x + new_width)
        end_y = int(start_y + new_height)

        # Ensure the dimensions are within bounds of the canvas
        if start_x < 0:
            resized_image = resized_image[:, abs(start_x):]
            start_x = 0

        if start_y < 0:
            resized_image = resized_image[abs(start_y):, :]
            start_y = 0

        if end_x > display_width:
            resized_image = resized_image[:, :display_width - start_x]
            end_x = display_width

        if end_y > display_height:
            resized_image = resized_image[:display_height - start_y, :]
            end_y = display_height

        # Place the resized and possibly clipped image onto the canvas
        canvas[start_y:end_y, start_x:end_x] = resized_image

        # Add annotations if toggled on
        if self.show_annotations:
            self.add_annotations_to_box()

        # Convert the image to base64
        _, buf = cv2.imencode('.jpg', canvas)
        image_base64 = base64.b64encode(buf).decode('utf-8')

        # Update the image display in Flet
        self.image_display.src_base64 = image_base64

        # Update WL and WW labels
        self.wl_label.value = f"Window Level: {self.window_level}"
        self.ww_label.value = f"Window Width: {self.window_width}"

        self.page.update()

    def add_annotations_to_box(self):
        """Move DICOM patient information to the annotation box."""
        self.annotation_box.controls.clear()  # Clear previous annotations

        # Extract patient information for annotation
        patient_name = getattr(self.dicom_data, 'PatientName', 'Unknown Patient')
        patient_id = getattr(self.dicom_data, 'PatientID', 'Unknown ID')
        patient_sex = getattr(self.dicom_data, 'PatientSex', 'Unknown Sex')
        study_description = getattr(self.dicom_data, 'StudyDescription', 'Unknown Study')
        series_description = getattr(self.dicom_data, 'SeriesDescription', 'Unknown Series')
        study_date = getattr(self.dicom_data, 'StudyDate', 'Unknown Date')
        study_time = getattr(self.dicom_data, 'StudyTime', 'Unknown Time')

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

        self.page.update()

    def apply_window_level(self, image, window_width, window_level):
        # Convert window width and level to float
        window_width = float(window_width)
        window_level = float(window_level)

        # Calculate minimum and maximum pixel values based on window width and level
        min_pixel_value = window_level - (window_width / 2)
        max_pixel_value = window_level + (window_width / 2)

        # Apply the windowing
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

    def on_pan_update(self, e):
        dx = e.local_x - self.start_drag_x
        dy = e.local_y - self.start_drag_y

        if self.is_adjusting_wl_ww:
            self.window_width += dx
            self.window_level -= dy
        elif self.is_dragging_mode:
            self.drag_offset = (self.drag_offset[0] + dx, self.drag_offset[1] + dy)

        self.start_drag_x = e.local_x
        self.start_drag_y = e.local_y
        self.display_image(self.cached_image)

    def on_pan_end(self, e):
        self.start_drag_x = 0
        self.start_drag_y = 0


    def toggle_drag_mode(self, e):
        self.is_dragging_mode = not self.is_dragging_mode
        self.error_label.value = "Drag Mode Active" if self.is_dragging_mode else "Drag Mode Inactive"
        self.page.update()

    def toggle_wl_ww_mode(self, e):
        self.is_adjusting_wl_ww = not self.is_adjusting_wl_ww
        self.error_label.value = "WL/WW Adjust Mode Active" if self.is_adjusting_wl_ww else "WL/WW Adjust Mode Inactive"
        self.page.update()

def main(page: ft.Page):
    DICOMViewer(page)

ft.app(target=main)
