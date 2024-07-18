import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import flet as ft
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import cv2
import base64
import flet.canvas as cv

class DICOMViewer:
    def __init__(self, page):
        self.page = page
        self.file_paths = []  # List to store DICOM file paths
        self.current_index = 0  # Index of the currently displayed DICOM file
        self.is_flipped_horizontal = False  # Variable to track the horizontal flip state
        self.is_flipped_vertical = False
        self.rotation_angle = 0  # Variable to track the rotation angle
        self.contrast_value = 1.0
        self.brightness_value = 0
        self.is_drawing_mode = False  # Variable to track drawing mode state
        self.draw_points = []  # List to store points for drawing
        page.window_maximized = True
        # Create UI elements
        self.create_ui()

    def create_ui(self):
        self.error_label = ft.Text("")
        self.image_display = ft.Image(width=600, height=600)
        self.next_button = ft.ElevatedButton("Next", on_click=self.show_next)
        self.prev_button = ft.ElevatedButton("Previous", on_click=self.show_previous)
        self.zoom_in_button = ft.ElevatedButton("Zoom In", on_click=self.zoom_in)
        self.zoom_out_button = ft.ElevatedButton("Zoom Out", on_click=self.zoom_out)
        self.rotate_button_clockwise = ft.ElevatedButton("Rotate Clockwise", on_click=lambda _: self.rotate_image(clockwise=False))
        self.rotate_button_counterclockwise = ft.ElevatedButton("Rotate Counterclockwise", on_click=lambda _: self.rotate_image(clockwise=True))
        self.flip_horizontal_button = ft.ElevatedButton("Flip Horizontal", on_click=lambda _: self.flip_image(horizontal=True))
        self.flip_vertical_button = ft.ElevatedButton("Flip Vertical", on_click=lambda _: self.flip_image(vertical=True))
        self.toggle_draw_button = ft.ElevatedButton("Toggle Draw Mode", on_click=self.toggle_draw_mode)

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
                self.toggle_draw_button
            ], alignment=ft.MainAxisAlignment.START),  # Align buttons to the start
            self.error_label,
            ft.Container(
                content=self.image_display,
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
                self.display_image(dicom_data.pixel_array)

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

        # Create figure and axes within the main thread context
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(adjusted_image, cmap=plt.cm.gray)
        ax.axis('off')

        canvas = FigureCanvas(fig)
        buf = BytesIO()

        canvas.print_jpeg(buf)
        buf.seek(0)

        # Convert the image buffer to base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Update the image display in Flet
        self.image_display.src_base64 = image_base64
        self.page.update()
        plt.close(fig)  # Close the figure after use

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
        pass  # Implement zoom in functionality

    def zoom_out(self, e):
        pass  # Implement zoom out functionality

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

                # Display the rotated image using matplotlib
                self.display_image(rotated_image)
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

                # Display the flipped image using matplotlib
                self.display_image(flipped_image)
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
                self.contrast_value = 1 + (self.contrast_slider.value / 100.0)
                self.brightness_value = self.brightness_slider.value * 2.55

                # Adjust and display the image using matplotlib
                self.display_image(dicom_data.pixel_array)
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
