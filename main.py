import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from full_code import *

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Studio")
        self.root.geometry("1300x700")
        self.root.configure(bg="#FCF6F5") 

        # Button and font styles
        self.button_bg = "#990011"
        self.button_fg = "#FFFFFF"
        self.font = ("Arial", 12, "bold")

        # Frames
        self.control_frame = tk.Frame(root, bg="#FCF6F5")
        self.control_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor="nw")

        self.image_frame = tk.Frame(root, bg="#FFFFFF", bd=2, relief=tk.SOLID)
        self.image_frame.pack(side=tk.RIGHT, padx=20, pady=20, expand=True, fill=tk.BOTH)

        # Upload button
        self.upload_button = tk.Button(
            self.control_frame,
            text="Upload Photo",
            command=self.upload_photo,
            bg=self.button_bg,
            fg=self.button_fg,
            font=self.font,
            relief="flat",  # Make the button's edges flat
            width=20,
            pady=10,  
            height=2 
        )
        self.upload_button.pack(pady=10)

        # Functionality frame
        self.functionality_frame = tk.Frame(self.control_frame, bg="#FCF6F5")
        self.functionality_frame.pack(pady=20)

        # Image attributes
        self.original_image = None
        self.processed_image = None

        # Halftoning dropdown menu 
        self.halftoning_var = tk.StringVar(self.control_frame)
        self.halftoning_var.set("Halftoning")
        
        # Simple edge dropdown menu
        self.simple_edge_detection_var = tk.StringVar(self.control_frame)
        self.simple_edge_detection_var.set("Simple Edge")

        # Advanced edge dropdown menu
        self.advanced_edge_detection_var = tk.StringVar(self.control_frame)
        self.advanced_edge_detection_var.set("Advanced Edge")

        # Filtering dropdown menu
        self.filtering_var = tk.StringVar(self.control_frame)
        self.filtering_var.set("Filtering")

        # Operations dropdown menu
        self.operation_var = tk.StringVar(self.control_frame)
        self.operation_var.set("Operations")

        # Histogram segementation dropdown menu
        self.segmentation_var = tk.StringVar(self.control_frame)
        self.segmentation_var.set("Segmentation")



    def upload_photo(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            # Load the image by read function from utils
            self.original_image = read_image(file_path)
            self.display_images()

            # Add functionality buttons
            self.add_functionality_buttons()

    def add_functionality_buttons(self):
        for widget in self.functionality_frame.winfo_children():
            widget.destroy()

        self.create_function_button("Threshold", self.threshold)
        self.create_function_button("Histogram", self.histogram)

        
        # Halftoning dropdown menu 
        halftoning_menu = tk.OptionMenu(
            self.functionality_frame,
            self.halftoning_var,
            "Simple Halftoning", "Advanced Halftoning"
        )
        halftoning_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1  # Set fixed height to match buttons
        )
        halftoning_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        halftoning_menu.pack(pady=10)

        # Trigger halftoning function based on selected option
        self.halftoning_var.trace("w", self.on_halftoning_select)

        # Simple Edge detection dropdown menu 
        simple_edge_detection_menu = tk.OptionMenu(
            self.functionality_frame,
            self.simple_edge_detection_var,
            "Sobel", "Prewitt", "Kirsch"
        )
        simple_edge_detection_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1 
        )
        simple_edge_detection_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu
        simple_edge_detection_menu.pack(pady=10)

        # Trigger edge detection function based on selected option
        self.simple_edge_detection_var.trace("w", self.on_simple_edge_detection_select)


        # Advanced Edge detection dropdown menu
        advanced_edge_detection_menu = tk.OptionMenu(
            self.functionality_frame,
            self.advanced_edge_detection_var,
            "Homogeneity", "Difference", "Gaussian", "Contrast-based", "Variance-based", "Range-based"
        )
        advanced_edge_detection_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1  # Set fixed height to match buttons
        )
        advanced_edge_detection_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        advanced_edge_detection_menu.pack(pady=10)

        # Trigger edge detection function based on selected option
        self.advanced_edge_detection_var.trace("w", self.on_advanced_edge_detection_select)

        # filtering dropdown menu
        filtering_menu = tk.OptionMenu(
            self.functionality_frame,
            self.filtering_var,
            "High-pass", "Low-pass", "Median"
        )
        filtering_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1  # Set fixed height to match buttons
        )
        filtering_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        filtering_menu.pack(pady=10)

        # Trigger filtering function based on selected option
        self.filtering_var.trace("w", self.on_filtering_select)

        # Operations dropdown menu
        Op_menu = tk.OptionMenu(
            self.functionality_frame,
            self.operation_var,
            "Add", "Subtract","Invert"
        )
        Op_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=1
        )
        Op_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu 
        Op_menu.pack(pady=10)

        # Trigger filtering function based on selected option
        self.operation_var.trace("w", self.on_operation_select)

        # histogram segementation dropdown menu
        segmentation_menu = tk.OptionMenu(
            self.functionality_frame,
            self.segmentation_var,
            "Manual", "Peak", "Valley", "Adaptive"
        )
        segmentation_menu.config(
            bg=self.button_bg, fg=self.button_fg, font=self.font, width=20,
            height=2  # Set fixed height to match buttons
        )
        segmentation_menu["menu"].config(
            bg=self.button_bg, fg=self.button_fg
        )  # Style the dropdown menu (no hover)
        segmentation_menu.pack(pady=10)

        # Trigger edge detection function based on selected option
        self.segmentation_var.trace("w", self.on_segmentation_select)





    def create_function_button(self, text, command):
        button = tk.Button(
            self.functionality_frame,
            text=text,
            command=command,
            bg=self.button_bg,
            fg=self.button_fg,
            font=self.font,
            relief="flat",  # Flat button style
            width=20,
            pady=10,  
            height=1  
        )
        button.pack(pady=10)

    def on_halftoning_select(self, *args):
        halftoning_type = self.halftoning_var.get()
        if halftoning_type == "Simple Halftoning":
            self.simple_halftoning()
        elif halftoning_type == "Advanced Halftoning":
            self.advanced_halftoning()
    
    def on_simple_edge_detection_select(self, *args):
        edge_detection_type = self.simple_edge_detection_var.get()
        if edge_detection_type == "Sobel":
            self.simple_edge_sobel()
        elif edge_detection_type == "Prewitt":
            self.simple_edge_prewitt()
        elif edge_detection_type == "Kirsch":
            self.simple_edge_kirsch()
    
    def on_advanced_edge_detection_select(self, *args):
        edge_detection_type = self.advanced_edge_detection_var.get()
        if edge_detection_type == "Homogeneity":
            self.advanced_edge_homogeneity()
        elif edge_detection_type == "Difference":
            self.advanced_edge_difference()
        elif edge_detection_type == "Gaussian":
            self.advanced_edge_differenceofGaussians()
        elif edge_detection_type == "Contrast-based":
            self.advanced_edge_contrastBased()
        elif edge_detection_type == "Variance-based":
            self.advanced_edge_varianceBased()
        else:
            self.advanced_edge_rangeBased()
        
    def on_filtering_select(self, *args):
        filtering_type = self.filtering_var.get()
        if filtering_type == "High-pass":
            self.high_bass_filtering()
        elif filtering_type == "Low-pass":
            self.low_bass_filtering()
        elif filtering_type == "Median":
            self.median_filtering()
    
    def on_operation_select(self, *args):
        op_type = self.operation_var.get()
        if op_type == "Add":
            self.add_image()
        elif op_type == "Subtract":
            self.subtract_image()
        elif op_type == "Invert":
            self.Invert_image()
    
    def on_segmentation_select(self, *args):
        segmentation_type = self.segmentation_var.get()
        if segmentation_type == "Manual":
            self.manual_segmentation()
        elif segmentation_type == "Peak":
            self.histogram_peaks_segmentation()
        elif segmentation_type == "Valley":
            self.histogram_valleys_segmentation()
        else:
            self.histogram_adaptive_segmentation()



    def display_images(self,technique=None):
        # Clear the image frame
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Display the original image
        if self.original_image is not None:
            original_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_img)
            original_resized = original_pil.resize((400, 400))
            original_photo = ImageTk.PhotoImage(original_resized)
            original_label = tk.Label(self.image_frame, text="Original Image", bg="#FFFFFF", font=self.font)
            original_label.grid(row=0, column=0, padx=10, pady=5)
            original_image_label = tk.Label(self.image_frame, image=original_photo, bg="#FFFFFF")
            original_image_label.image = original_photo
            original_image_label.grid(row=1, column=0, padx=10, pady=5)

        # Display the processed image
        if self.processed_image is not None:
            processed_pil = Image.fromarray(self.processed_image)
            processed_resized = processed_pil.resize((400, 400))
            processed_photo = ImageTk.PhotoImage(processed_resized)
            processed_label_text = technique if technique else "Processed Image"
            processed_label = tk.Label(self.image_frame, text=processed_label_text, bg="#FFFFFF", font=self.font)
            processed_label.grid(row=0, column=1, padx=10, pady=5)
            processed_image_label = tk.Label(self.image_frame, image=processed_photo, bg="#FFFFFF")
            processed_image_label.image = processed_photo
            processed_image_label.grid(row=1, column=1, padx=10, pady=5)



    def threshold(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the threshold function from image_utils
        threshold_value = threshold(self.original_image)

        # Display threshold value
        messagebox.showinfo("Threshold Applied", f"Threshold value: {threshold_value:.2f}")

    def simple_halftoning(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the halftoning function from image_utils
        processed_image = simple_halftoning(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Simple Halftoning")

    def advanced_halftoning(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced halftoning function from image_utils
        processed_image = advanced_halftoning(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Advanced Halftoning")
    
    def histogram(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the histogram function from utils
        processed_image = histogram(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Histogram")
    
    def simple_edge_sobel(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the simeple edge sobel function from utils
        processed_image = simple_edge_sobel(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images("Sobel Operator")    

    def simple_edge_prewitt(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the simple edge prewitt function from utils
        processed_image = simple_edge_prewitt(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Prewitt Operator")

    def simple_edge_kirsch(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

    # Call the Kirsch edge detection function
        processed_image, direction = simple_edge_kirsch(self.original_image)

    # Update the processed image
        self.processed_image = processed_image

    # Optional: Store or process direction data for further use
        self.edge_direction = direction
        print("Edge Direction (Numerical Indices):")
  # Save direction for potential visualization or analysis
        print(direction)
    # Refresh images
        self.display_images(technique="Kirsch Operator")

    def advanced_edge_homogeneity(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge homogeneity function from image_utils
        processed_image = advanced_edge_homogeneity(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Homogeneity Operator")

    def advanced_edge_difference(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge difference function from utils
        processed_image = advanced_edge_difference(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Difference Operator")
    
    def advanced_edge_differenceofGaussians(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge difference function from utils
        processed_image = advanced_edge_differenceofGaussians(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Difference of Gaussians")
    
    def advanced_edge_contrastBased(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced edge contrast-based function from utils
        processed_image = advanced_edge_contrastBased(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Contrast-Based")

    def advanced_edge_rangeBased(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image = advanced_edge_rangeBased(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Range-based")

    def advanced_edge_varianceBased(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image = advanced_edge_varianceBased(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique='Variance-Based')

    def high_bass_filtering(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the high bass function from utils
        processed_image = high_bass_filtering(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="High-Pass Filter")

    def low_bass_filtering(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the high bass function from utils
        processed_image = low_bass_filtering(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Low-Bass Filter")
        
    def median_filtering(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the median function from utils
        processed_image = median_filtering(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Median Filter")
    
    def add_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the add image function from utils
        processed_image = add_image(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Add Image")
    
    def subtract_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the subtract image function from utils
        processed_image = subtract_image(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Subtract Image")

    def Invert_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the subtract image function from utils
        processed_image = invert_image(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Inverted Image")

    def histogram_peaks_segmentation(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image, clusters = histogram_peaks_segmentation(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Peak Segmentation")
    
    def manual_segmentation(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image = manual_segmentation(self.original_image,low_thresh=30,high_thresh=100)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Manual Segmentation")

    def histogram_valleys_segmentation(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image = histogram_valleys_segmentation(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Valley Segmentation")

    def histogram_adaptive_segmentation(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Call the advanced histogram function from utils
        processed_image = histogram_adaptive_segmentation(self.original_image)

        # Update the processed image
        self.processed_image = processed_image

        # Refresh images
        self.display_images(technique="Adaptive Segmentation")

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
