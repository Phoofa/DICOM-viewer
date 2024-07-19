import os
import pydicom
import numpy as np
import flet as ft
from io import BytesIO
import cv2
import base64
import flet.canvas as cv

class DICOMViewer:
    def __init__(self, page):
        self.page = page
        self.page.theme_mode = ft.ThemeMode.DARK
        self.file_paths = []  # List to store DICOM file paths
        self.current_index = 0  # Index of the currently displayed DICOM file
        self.is_flipped_horizontal = False  # Variable to track the horizontal flip state
        self.is_flipped_vertical = False
        self.rotation_angle = 0  # Variable to track the rotation angle
        self.contrast_value = 1.0
        self.brightness_value = 0
        self.scale_factor = 1.0  # Scale factor for zoom
        self.zoom_center = (0.5, 0.5)  # Center point for zoom (normalized coordinates)
        self.is_drawing_mode = False  # Variable to track drawing mode state
        self.is_dragging_mode = False  # Variable to track dragging mode state
        self.is_dragging = False  # Variable to track if currently dragging
        self.drag_offset = (0, 0)  # Offset for dragging
        self.draw_points = []  # List to store points for drawing
        self.cached_image = None  # Cached image for efficient dragging
        page.window.maximized = True  # Updated to use the new way of setting the window maximized
        # Create UI elements
        self.create_ui()

    def create_ui(self):
        self.error_label = ft.Text("")
        self.image_display = ft.Image(width=1200, height=1200)  # Adjust the size here
        self.gesture_detector = ft.GestureDetector(
            content=ft.Container(
                content=self.image_display,
                width=1200,
                height=1200,
            ),
            on_pan_start=self.on_pointer_down,
            on_pan_end=self.on_pointer_up,
            on_pan_update=self.on_pointer_move,
        )

        self.next_button = ft.ElevatedButton("Next", on_click=self.show_next)
        self.prev_button = ft.ElevatedButton("Previous", on_click=self.show_previous)
        self.zoom_in_button = ft.ElevatedButton("Zoom In", on_click=self.zoom_in)
        self.zoom_out_button = ft.ElevatedButton("Zoom Out", on_click=self.zoom_out)
        self.rotate_button_clockwise = ft.ElevatedButton("Rotate Clockwise", on_click=lambda _: self.rotate_image(clockwise=False))
        self.rotate_button_counterclockwise = ft.ElevatedButton("Rotate Counterclockwise", on_click=lambda _: self.rotate_image(clockwise=True))
        self.flip_horizontal_button = ft.ElevatedButton("Flip Horizontal", on_click=lambda _: self.flip_image(horizontal=True))
        self.flip_vertical_button = ft.ElevatedButton("Flip Vertical", on_click=lambda _: self.flip_image(vertical=True))
        self.toggle_draw_button = ft.ElevatedButton("Toggle Draw Mode", on_click=self.toggle_draw_mode)
        self.toggle_drag_button = ft.ElevatedButton("Toggle Drag Mode", on_click=self.toggle_drag_mode)

        self.contrast_label = ft.Text("Contrast:")
        self.contrast_slider = ft.Slider(min=-100, max=100, value=0, on_change=self.adjust_image, width=300)
        self.brightness_label = ft.Text("Brightness:")
        self.brightness_slider = ft.Slider(min=-100, max=100, value=0, on_change=self.adjust_image, width=300)

        # Create file pickers
        self.file_picker = ft.FilePicker(on_result=self.on_file_selected)
        self.page.overlay.append(self.file_picker)
        
        self.directory_picker = ft.FilePicker(on_result=self.on_directory_selected)
        self.page.overlay.append(self.directory_picker)

        # Create a menu bar with browse options
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
                )
            ]
        )

        # Create the drawing canvas
        self.canvas = cv.Canvas()

        # Add elements to the page
        self.page.add(
            ft.Row([
                self.menubar,
                self.prev_button,
                self.next_button,
                self.zoom_in_button,
                self.zoom_out_button,
                self.rotate_button_clockwise,
                self.rotate_button_counterclockwise,
                self.flip_horizontal_button,
                self.flip_vertical_button,
                self.toggle_draw_button,
                self.toggle_drag_button
            ], alignment=ft.MainAxisAlignment.START),  # Align buttons to the start
            self.error_label,
            ft.Container(
                content=self.gesture_detector,
                alignment=ft.alignment.center,
                expand=True
            ),
            ft.Column([self.contrast_label, self.contrast_slider]),
            ft.Column([self.brightness_label, self.brightness_slider]),
            self.canvas
        )

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
            self.current_index = 0
            self.load_dicom_image()
        else:
            self.error_label.value = "No DICOM file selected."
            self.page.update()

    def load_dicom_image(self):
        if self.file_paths:
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])
                self.cached_image = dicom_data.pixel_array
                self.display_image(self.cached_image)

                # Clear error message
                self.error_label.value = ""
                self.page.update()
            except Exception as e:
                # Display an error message if there is an issue
                self.error_label.value = f"Error: {str(e)}"
                self.page.update()

    def display_image(self, image_array):
        # Apply brightness and contrast adjustments
        adjusted_image = self.adjust_brightness_contrast(image_array, self.contrast_value, self.brightness_value)

        # Get the dimensions of the adjusted image
        height, width = adjusted_image.shape

        # Calculate the cropping region based on the zoom factor and center point
        center_x, center_y = int(self.zoom_center[0] * width), int(self.zoom_center[1] * height)
        half_width = int(width / (2 * self.scale_factor))
        half_height = int(height / (2 * self.scale_factor))

        # Ensure the cropping region is within the image boundaries
        start_x = max(center_x - half_width, 0)
        end_x = min(center_x + half_width, width)
        start_y = max(center_y - half_height, 0)
        end_y = min(center_y + half_height, height)

        # Crop the image to the calculated region
        cropped_image = adjusted_image[start_y:end_y, start_x:end_x]

        # Apply drag offset to center the image properly
        drag_x, drag_y = self.drag_offset
        if drag_x != 0 or drag_y != 0:
            rows, cols = cropped_image.shape
            M = np.float32([[1, 0, drag_x], [0, 1, drag_y]])
            cropped_image = cv2.warpAffine(cropped_image, M, (cols, rows))

        # Resize the cropped image back to the original display size
        cropped_image = cv2.resize(cropped_image, (1200, 1200), interpolation=cv2.INTER_LINEAR)

        # Convert the image to base64
        _, buf = cv2.imencode('.jpg', cropped_image)
        image_base64 = base64.b64encode(buf).decode('utf-8')

        # Update the image display in Flet
        self.image_display.src_base64 = image_base64
        self.page.update()

        # Draw existing points
        self.draw_existing_points()

    def adjust_brightness_contrast(self, image, contrast, brightness):
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted

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

    def adjust_image(self, e):
        if self.file_paths:
            try:
                # Read the DICOM file
                dicom_data = pydicom.dcmread(self.file_paths[self.current_index])

                # Update contrast and brightness values
                self.contrast_value = (self.contrast_slider.value / 100.0)
                self.brightness_value = self.brightness_slider.value * 2.55

                # Adjust and display the image using OpenCV
                self.display_image(dicom_data.pixel_array)
                self.cached_image = dicom_data.pixel_array
                self.error_label.value = ""
                self.page.update()
            except Exception as e:
                # Display an error message if there is an issue
                self.error_label.value = f"Error: {str(e)}"
                self.page.update()

    def toggle_draw_mode(self, e):
        self.is_drawing_mode = not self.is_drawing_mode
        if self.is_drawing_mode:
            self.canvas.on_click = self.start_drawing
            self.canvas.on_drag = self.continue_drawing
        else:
            self.canvas.on_click = None
            self.canvas.on_drag = None

    def toggle_drag_mode(self, e):
        self.is_dragging_mode = not self.is_dragging_mode
        self.error_label.value = "Drag Mode Active" if self.is_dragging_mode else "Drag Mode Inactive"
        self.page.update()

    def on_pointer_down(self, e):
        if self.is_dragging_mode:
            self.is_dragging = True
            self.start_drag_x = e.local_x
            self.start_drag_y = e.local_y

    def on_pointer_up(self, e):
        if self.is_dragging:
            self.is_dragging = False

    def on_pointer_move(self, e):
        if self.is_dragging:
            dx = e.local_x - self.start_drag_x
            dy = e.local_y - self.start_drag_y
            self.drag_offset = (self.drag_offset[0] + dx, self.drag_offset[1] + dy)
            self.start_drag_x = e.local_x
            self.start_drag_y = e.local_y
            self.display_image(self.cached_image)  # Update the image display with the cached image and drag offset

    def start_drawing(self, e):
        if self.is_drawing_mode:
            self.draw_points.append((e.x, e.y))
            self.canvas.add(cv.Line(start=cv.Point(e.x, e.y), end=cv.Point(e.x, e.y), paint=ft.Paint(stroke_width=2, color="red")))
            self.page.update()

    def continue_drawing(self, e):
        if self.is_drawing_mode:
            start_point = self.draw_points[-1]
            self.draw_points.append((e.x, e.y))
            self.canvas.add(cv.Line(start=cv.Point(*start_point), end=cv.Point(e.x, e.y), paint=ft.Paint(stroke_width=2, color="red")))
            self.page.update()

    def draw_existing_points(self):
        for i in range(1, len(self.draw_points)):
            start_point = self.draw_points[i - 1]
            end_point = self.draw_points[i]
            self.canvas.add(cv.Line(start=cv.Point(*start_point), end=cv.Point(*end_point), paint=ft.Paint(stroke_width=2, color="red")))
        self.page.update()

def main(page: ft.Page):
    DICOMViewer(page)

ft.app(target=main)
