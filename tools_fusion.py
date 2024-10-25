import tkinter as tk


def button_click(message):
    label.config(text=message)


# Create the main window
root = tk.Tk()
root.title("TOOL FUSION")

# Set window size to half of the screen
window_width = root.winfo_screenwidth() // 2
window_height = root.winfo_screenheight() // 2
root.geometry(f"{window_width}x{window_height}")

# Color Palette
background_color = "#242424"
text_color = "#FFFFFF"
button_color = "#3C90D8"
button_hover_color = "#539EC9"

# Label for Normal Tools
normal_tools_label = tk.Label(root, text="Normal Tools", font=("Helvetica", 14), fg=text_color, bg=background_color)
normal_tools_label.grid(row=0, column=0, columnspan=4, pady=10, sticky='nsew')

# Buttons for Normal Tools
button1 = tk.Button(root, text="Video to Audio Conv", command=lambda: button_click1(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button1.grid(row=1, column=0, padx=10, pady=5, sticky='nsew')


# video to Audio convertor
def button_click1():
    import os
    import tkinter as tk
    from tkinter import filedialog
    from moviepy.editor import VideoFileClip


    class VideoToAudioConverter:
        def __init__(self, root):
            self.root = root
            self.root.title("Video to Audio Converter")

            self.create_widgets()

        def create_widgets(self):
            self.label = tk.Label(self.root, text="Select a video file:")
            self.label.pack()

            self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_file)
            self.browse_button.pack()

            self.convert_button = tk.Button(self.root, text="Convert", command=self.convert_video_to_audio)
            self.convert_button.pack()

        def browse_file(self):
            self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])

        def convert_video_to_audio(self):
            if hasattr(self, 'video_path') and self.video_path:
                video_clip = VideoFileClip(self.video_path)
                audio_clip = video_clip.audio

                audio_extension = os.path.splitext(self.video_path)[0] + ".mp3"

                audio_clip.write_audiofile(audio_extension, codec='mp3')

                video_clip.close()
                audio_clip.close()

                tk.messagebox.showinfo("Conversion Complete", f"Audio file saved to {audio_extension}")

    if __name__ == "__main__":
        root = tk.Tk()
        app = VideoToAudioConverter(root)
        root.mainloop()


button2 = tk.Button(root, text="Language Translator", command=lambda: button_click2(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button2.grid(row=1, column=1, padx=10, pady=5, sticky='nsew')


# language translator

def button_click2():
    from translate import Translator
    import tkinter as tk
    from tkinter import ttk, messagebox

    class TranslatorApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Text Translator")

            # Language dictionary
            self.languages = {"bn": "Bangla", "en": "English", "ko": "Korean", "fr": "French", "de": "German",
                              "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese", 'la': "Latin",
                              "ms": "Malay", "ne": "Nepali", "ru": "Russian", "ar": "Arabic", "zh": "Chinese",
                              "es": "Spanish"}

            # GUI Elements
            tk.Label(root, text="Select Destination Language:").pack(pady=10)
            self.language_combobox = ttk.Combobox(root, values=list(self.languages.values()))
            self.language_combobox.pack(pady=10)

            tk.Label(root, text="Enter Text to Translate:").pack(pady=10)
            self.input_text = tk.Entry(root, width=40)
            self.input_text.pack(pady=10)

            tk.Button(root, text="Translate", command=self.translate_text).pack(pady=20)

            # Translation Output
            tk.Label(root, text="Translated Output:").pack(pady=10)
            self.output_text = tk.Text(root, height=10, width=60, wrap=tk.WORD)
            self.output_text.pack()

        def translate_text(self):
            dest_language = self.get_language_code()
            if dest_language:
                text_to_translate = self.input_text.get()
                translator = Translator(to_lang=dest_language)

                try:
                    translation = translator.translate(text_to_translate)

                    # Clearing previous content and displaying the new translation
                    self.output_text.delete(1.0, tk.END)
                    self.output_text.insert(tk.END, f"{self.languages[dest_language]} translation: {translation}\n")

                except Exception as e:
                    messagebox.showerror("Error", f"Error during translation: {str(e)}")

        def get_language_code(self):
            selected_language = self.language_combobox.get()
            for code, name in self.languages.items():
                if selected_language == name:
                    return code
            else:
                messagebox.showerror("Error", "Please select a valid destination language.")
                return None

    if __name__ == "__main__":
        root = tk.Tk()
        app = TranslatorApp(root)
        root.mainloop()


button3 = tk.Button(root, text="QR Code Genterator", command=lambda: button_click3(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button3.grid(row=1, column=2, padx=10, pady=5, sticky='nsew')

#qr code generatoer
def button_click3():
    import tkinter as tk
    from tkinter import filedialog
    import qrcode

    class QRCodeGenerator:
        def __init__(self, root):
            self.root = root
            self.root.title("QR Code Generator")

            self.create_widgets()

        def create_widgets(self):
            self.label = tk.Label(self.root, text="Enter data to encode:")
            self.label.pack()

            self.entry = tk.Entry(self.root, width=40)
            self.entry.pack()

            self.generate_button = tk.Button(self.root, text="Generate QR Code", command=self.generate_qr_code)
            self.generate_button.pack()

        def generate_qr_code(self):
            data = self.entry.get()

            if data:
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(data)
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")

                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])

                if file_path:
                    img.save(file_path)
                    tk.messagebox.showinfo("QR Code Generator", f"QR code saved to {file_path}")

    if __name__ == "__main__":
        root = tk.Tk()
        app = QRCodeGenerator(root)
        root.mainloop()


button4 = tk.Button(root, text="Text to Speech Generator", command=lambda: button_click4(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button4.grid(row=1, column=3, padx=10, pady=5, sticky='nsew')

#Text to Speech Generator
def button_click4():
    from gtts import gTTS
    import tkinter as tk
    from tkinter import filedialog

    def convert_text_to_speech():
        try:
            # Ask the user to choose the input text file
            input_text_file = filedialog.askopenfilename(title="Select Text File", filetypes=[("Text files", "*.txt")])

            # Check if a file is selected
            if not input_text_file:
                return

            # Ask the user to choose the output audio file
            output_audio_file = filedialog.asksaveasfilename(title="Save As", defaultextension=".mp3",
                                                             filetypes=[("Audio files", "*.mp3")])

            # Check if a file is selected
            if not output_audio_file:
                return

            # Read text from the input file
            with open(input_text_file, 'r', encoding='utf-8') as file:
                text_content = file.read()

            # Convert text to speech
            tts = gTTS(text_content, lang='en')
            tts.save(output_audio_file)

            print(f"Conversion successful.")
        except Exception as ex:
            print(f'Error: {str(ex)}')

    # Create the main GUI window
    root = tk.Tk()
    root.title("Text to Speech Converter")

    # Create "Choose File" button
    choose_file_button = tk.Button(root, text="Choose Text File", command=convert_text_to_speech)
    choose_file_button.pack(pady=20)

    # Start the GUI main loop
    root.mainloop()


button5 = tk.Button(root, text="File Convertor", command=lambda: button_click5(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button5.grid(row=2, column=0, padx=10, pady=5, sticky='nsew')

#file convertor
def button_click5():
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox
    import pandas as pd
    import img2pdf
    import json
    import xmltodict
    from PIL import Image
    from io import BytesIO

    class FileConverterApp:
        def __init__(self, master):
            self.master = master
            master.title("File Converter")

            # Create conversion type dropdown
            self.conversion_label = tk.Label(master, text="Select conversion type:")
            self.conversion_label.pack()

            self.conversion_var = tk.StringVar()
            self.conversion_choices = ["Image to PDF", "Excel to CSV", "JSON to CSV", "XML to JSON", "JPEG to PNG"]
            self.conversion_dropdown = tk.OptionMenu(master, self.conversion_var, *self.conversion_choices)
            self.conversion_dropdown.pack()

            # Create file input button
            self.file_button = tk.Button(master, text="Select File", command=self.select_file)
            self.file_button.pack()

            # Create convert button
            self.convert_button = tk.Button(master, text="Convert", command=self.convert)
            self.convert_button.pack()

            # Create reverse button
            self.reverse_button = tk.Button(master, text="Reverse", command=self.reverse)
            self.reverse_button.pack()

        def select_file(self):
            file_path = filedialog.askopenfilename()
            if file_path:
                messagebox.showinfo("File Selected", f"Selected File: {file_path}")
                self.file_path = file_path
            else:
                messagebox.showwarning("No File Selected", "Please select a file for conversion.")

        def convert(self):
            conversion_type = self.conversion_var.get()

            if not hasattr(self, 'file_path'):
                messagebox.showwarning("No File Selected", "Please select a file for conversion.")
                return

            if conversion_type == 'Image to PDF':
                self.convert_img_to_pdf()

            elif conversion_type == 'Excel to CSV':
                self.convert_excel_to_csv()

            elif conversion_type == 'JSON to CSV':
                self.convert_json_to_csv()

            elif conversion_type == 'XML to JSON':
                self.convert_xml_to_json()

            elif conversion_type == 'JPEG to PNG':
                self.convert_jpeg_to_png()

        def reverse(self):
            conversion_type = self.conversion_var.get()

            if not hasattr(self, 'file_path'):
                messagebox.showwarning("No File Selected", "Please select a file for reverse conversion.")
                return

            if conversion_type == 'Image to PDF':
                self.reverse_pdf_to_img()

            elif conversion_type == 'Excel to CSV':
                self.reverse_csv_to_excel()

            elif conversion_type == 'JSON to CSV':
                self.reverse_csv_to_json()

            elif conversion_type == 'XML to JSON':
                self.reverse_json_to_xml()

            elif conversion_type == 'JPEG to PNG':
                self.reverse_png_to_jpeg()

        def convert_img_to_pdf(self):
            img = Image.open(self.file_path)
            pdf_bytes = img2pdf.convert(img.filename)
            img.close()
            self.save_file(pdf_bytes, "converted_file.pdf")

        def convert_excel_to_csv(self):
            df = pd.read_excel(self.file_path)
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            self.save_file(csv_bytes, "converted_file.csv")

        def convert_json_to_csv(self):
            df = pd.read_json(self.file_path)
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            self.save_file(csv_bytes, "converted_file.csv")

        def convert_xml_to_json(self):
            with open(self.file_path) as xml_data:
                json_data = json.dumps(xmltodict.parse(xml_data.read()), indent=2)
            self.save_file(json_data.encode('utf-8'), "converted_file.json")

        def convert_jpeg_to_png(self):
            img = Image.open(self.file_path)
            png_bytes = BytesIO()
            img.save(png_bytes, format='PNG')
            img.close()
            self.save_file(png_bytes.getvalue(), "converted_file.png")

        def reverse_pdf_to_img(self):
            with open(self.file_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
                images = img2pdf.convert_from_bytes(pdf_data)
                if images:
                    img = images[0]
                    self.save_file(img.tobytes(), "reversed_file.jpg")
                else:
                    messagebox.showwarning("Reverse Conversion Failed", "No images found in the PDF.")

        def reverse_csv_to_excel(self):
            df = pd.read_csv(self.file_path)
            excel_bytes = df.to_excel(index=False).saveformat('xlsx')
            self.save_file(excel_bytes, "reversed_file.xlsx")

        def reverse_csv_to_json(self):
            df = pd.read_csv(self.file_path)
            json_data = df.to_json(indent=2)
            self.save_file(json_data.encode('utf-8'), "reversed_file.json")

        def reverse_json_to_xml(self):
            with open(self.file_path) as json_file:
                json_data = json.load(json_file)
                xml_data = xmltodict.unparse(json_data, pretty=True)
            self.save_file(xml_data.encode('utf-8'), "reversed_file.xml")

        def reverse_png_to_jpeg(self):
            img = Image.open(self.file_path)
            jpeg_bytes = BytesIO()
            img.convert('RGB').save(jpeg_bytes, format='JPEG')
            img.close()
            self.save_file(jpeg_bytes.getvalue(), "reversed_file.jpeg")

        def save_file(self, data, file_name):
            file_path = filedialog.asksaveasfilename(defaultextension=".*", filetypes=[("All Files", "*.*")],
                                                     initialfile=file_name)
            if file_path:
                with open(file_path, 'wb') as output_file:
                    output_file.write(data)
                    messagebox.showinfo("Conversion Successful", f"File saved as {file_path}")

    if __name__ == "__main__":
        root = tk.Tk()
        app = FileConverterApp(root)
        root.mainloop()


button6 = tk.Button(root, text="Image Resizer", command=lambda: button_click6(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button6.grid(row=2, column=1, padx=10, pady=5, sticky='nsew')


# Image Resizer

def button_click6():
    import cv2
    import tkinter as tk
    from tkinter import filedialog

    class ImageResizerApp:
        def __init__(self, master):
            self.master = master
            self.master.title("Image Resizer App")

            self.btn_browse = tk.Button(master, text="Browse Image", command=self.browse_image)
            self.btn_browse.pack(pady=10)

            self.k = 5  # default scaling ratio

            self.btn_save = tk.Button(master, text="Save Resized Image", command=self.save_resized_image,
                                      state=tk.DISABLED)
            self.btn_save.pack(pady=10)

            self.resized_image_path = ""

            # Labels to display details
            self.label_original_size = tk.Label(master, text="Original Image Size: N/A")
            self.label_original_size.pack()

            self.label_resized_size = tk.Label(master, text="Resized Image Size: N/A")
            self.label_resized_size.pack()

            self.label_resized_path = tk.Label(master, text="Resized Image Path: N/A")
            self.label_resized_path.pack()

        def browse_image(self):
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

            if file_path:
                self.resize_image(file_path)

        def resize_image(self, file_path):
            img = cv2.imread(file_path)

            if img is not None:
                original_size = f"Original Image Size: {img.shape[1]} x {img.shape[0]}"
                self.label_original_size.config(text=original_size)

                width = int(img.shape[1] / self.k)
                height = int(img.shape[0] / self.k)

                scaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                resized_size = f"Resized Image Size: {width} x {height}"
                self.label_resized_size.config(text=resized_size)

                cv2.imshow("Resized Output", scaled)
                cv2.waitKey(500)
                cv2.destroyAllWindows()

                # Save the resized image
                self.resized_image_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                                       filetypes=[("JPEG files", "*.jpg")])
                cv2.imwrite(self.resized_image_path, scaled)
                print(f"Resized image saved at: {self.resized_image_path}")

                # Update the label with the resized image path
                self.label_resized_path.config(text=f"Resized Image Path: {self.resized_image_path}")

                # Enable the "Save Resized Image" button
                self.btn_save.config(state=tk.NORMAL)
            else:
                print("Error: Unable to read the image.")

        def save_resized_image(self):
            # Additional functionality to save the resized image
            if self.resized_image_path:
                print(f"Image saved to: {self.resized_image_path}")
            else:
                print("Error: No resized image to save.")

    if __name__ == "__main__":
        root = tk.Tk()
        app = ImageResizerApp(root)
        root.mainloop()


button7 = tk.Button(root, text="YouTube Video Downloader", command=lambda: button_click7(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button7.grid(row=2, column=2, padx=10, pady=5, sticky='nsew')


# youtube Video Downloader

def button_click7():
    import tkinter as tk
    from tkinter import filedialog
    from pytube import YouTube

    class YouTubeVideoDownloader:
        def __init__(self, root):
            self.root = root
            self.root.title("YouTube Video Downloader")

            self.create_widgets()

        def create_widgets(self):
            self.label = tk.Label(self.root, text="Enter YouTube Video URL:")
            self.label.pack()

            self.url_entry = tk.Entry(self.root, width=50)
            self.url_entry.pack()

            self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_location)
            self.browse_button.pack()

            self.download_button = tk.Button(self.root, text="Download Video", command=self.download_video)
            self.download_button.pack()

        def browse_location(self):
            self.download_path = filedialog.askdirectory()

        def download_video(self):
            video_url = self.url_entry.get()

            if not video_url:
                tk.messagebox.showwarning("Warning", "Please enter a valid YouTube video URL.")
                return

            try:
                youtube = YouTube(video_url)
                video = youtube.streams.filter(progressive=True, file_extension="mp4").first()

                if self.download_path:
                    video.download(self.download_path)
                    tk.messagebox.showinfo("Download Complete", f"Video downloaded to {self.download_path}")
                else:
                    tk.messagebox.showwarning("Warning", "Please choose a download location.")

            except Exception as e:
                tk.messagebox.showerror("Error", f"Error: {str(e)}")

    if __name__ == "__main__":
        root = tk.Tk()
        app = YouTubeVideoDownloader(root)
        root.mainloop()


# image to pencil_sketch

button8 = tk.Button(root, text="Image to PencilSketch Conv", command=lambda: button_click8(), bg=button_color, fg=text_color,
                    padx=10, pady=5, activebackground=button_hover_color)
button8.grid(row=2, column=3, padx=10, pady=5, sticky='nsew')
#image to pencilSketch Conv

def button_click8():
    import tkinter as tk
    from tkinter import filedialog
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance

    def convert_to_pencil_sketch(input_image, output_image):
        try:

            # Open the image
            with Image.open(input_image) as img:
                # Convert to grayscale
                grayscale_img = ImageOps.grayscale(img)

                # Invert the image
                inverted_img = ImageOps.invert(grayscale_img)

                # Apply blur to enhance pencil-like effect
                blurred_img = inverted_img.filter(ImageFilter.GaussianBlur(radius=5))

                # Adjust contrast for better results
                enhancer = ImageEnhance.Contrast(blurred_img)
                final_img = enhancer.enhance(2.0)

                # Save the result
                final_img.save(output_image)

            print(
                f"Conversion successful. Image '{input_image}' has been converted to a pencil sketch '{output_image}'.")
        except Exception as ex:
            print(f'Error: {str(ex)}')

    def choose_image():
        # Ask the user to choose the input image
        input_image = filedialog.askopenfilename(title="Select Image File",
                                                 filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        # Check if a file is selected
        if not input_image:
            return

        # Ask the user to choose the output image
        output_image = filedialog.asksaveasfilename(title="Save As", defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png")])

        # Check if a file is selected
        if not output_image:
            return

        # Convert image to pencil sketch
        convert_to_pencil_sketch(input_image, output_image)

    # Create the main GUI window
    root = tk.Tk()
    root.title("Image to Pencil Sketch Converter")

    # Create "Choose Image" button
    choose_image_button = tk.Button(root, text="Choose Image", command=choose_image)
    choose_image_button.pack(pady=20)

    # Start the GUI main loop
    root.mainloop()


# Label for AI Tools
ai_tools_label = tk.Label(root, text="AI Tools", font=("Helvetica", 14), fg=text_color, bg=background_color)
ai_tools_label.grid(row=3, column=0, columnspan=4, pady=10, sticky='nsew')

# Buttons for AI Tools
button9 = tk.Button(root, text="Text to Img Generator", command=lambda: button_click9(), bg=button_color, fg=text_color, padx=10,
                    pady=5, activebackground=button_hover_color)
button9.grid(row=4, column=0, padx=10, pady=5, sticky='nsew')


# text to img generation
def button_click9():
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
    from PIL import Image, ImageTk
    from diffusers import StableDiffusionPipeline

    class TextToImageGenerator:
        def __init__(self, root):
            self.root = root
            self.root.title("Text to Image Generator")

            self.create_widgets()

        def create_widgets(self):
            self.label = tk.Label(self.root, text="Enter Text:")
            self.label.pack()

            self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=5)
            self.text_area.pack(padx=10, pady=10)

            self.generate_button = tk.Button(self.root, text="Generate Image", command=self.generate_image)
            self.generate_button.pack(pady=10)

            self.image_label = tk.Label(self.root, text="Generated Image:")
            self.image_label.pack()

            self.canvas = tk.Canvas(self.root, width=400, height=400)
            self.canvas.pack()

        def generate_image(self):
            text_prompt = self.text_area.get("1.0", tk.END).strip()

            if not text_prompt:
                messagebox.showwarning("Warning", "Please enter text.")
                return

            try:
                # Replace 'YOUR_AUTH_TOKEN' with your OpenAI authorization token
                authorization_token = "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"
                model_id = "CompVis/stable-diffusion-v1-4"

                pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=authorization_token)
                image = pipe(text_prompt, guidance_scale=8.5).images[0]

                self.display_image(image)

            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")

        def display_image(self, image):
            image = image.convert("RGB")  # Ensure the image is in RGB mode
            img = ImageTk.PhotoImage(image)
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

    if __name__ == "__main__":
        root = tk.Tk()
        app = TextToImageGenerator(root)
        root.mainloop()


button10 = tk.Button(root, text="Face Emotion Dectection", command=lambda: button_click10(), bg=button_color, fg=text_color, padx=10,
                     pady=5, activebackground=button_hover_color)
button10.grid(row=4, column=1, padx=10, pady=5, sticky='nsew')


# face emotion dectection
def button_click10():
    import sys

    import cv2
    from keras.models import model_from_json
    import numpy as np

    json_file = open("emotiondetector.json", 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("emotionddectector.h5")
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_file)

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'netural', 5: 'sad', 6: "surprise"}
    while True:
        i, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        try:
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                            (0, 0, 255))
            cv2.imshow("output", im)
            cv2.waitKey(27)
        except cv2.error:
            pass
        key = cv2.waitKey(1)
        if key == ord('h'):
            sys.exit(0)


button11 = tk.Button(root, text="Voice Recoginizer", command=lambda: button_click11(), bg=button_color, fg=text_color, padx=10,
                     pady=5, activebackground=button_hover_color)
button11.grid(row=4, column=2, padx=10, pady=5, sticky='nsew')


# small virtual assistance
def button_click11():
    import tkinter as tk
    from tkinter import scrolledtext
    import pyttsx3
    import speech_recognition as sr

    class VirtualAssistant:
        def __init__(self, master):
            self.master = master
            master.title("Virtual Assistant")

            self.output_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10)
            self.output_text.pack(padx=10, pady=10)

            self.listen_button = tk.Button(master, text="Listen", command=self.listen_command)
            self.listen_button.pack(pady=5)

            self.clear_button = tk.Button(master, text="Clear", command=self.clear_output)
            self.clear_button.pack(pady=5)

            self.quit_button = tk.Button(master, text="Quit", command=master.destroy)
            self.quit_button.pack(pady=5)

            self.engine = pyttsx3.init()
            self.recognizer = sr.Recognizer()

        def speak(self, text):
            self.engine.say(text)
            self.engine.runAndWait()

        def listen_command(self):
            try:
                with sr.Microphone() as source:
                    self.output_text.insert(tk.END, "Listening...\n")
                    self.output_text.update_idletasks()
                    audio_data = self.recognizer.listen(source, timeout=5)
                    self.output_text.insert(tk.END, "Recognizing...\n")
                    self.output_text.update_idletasks()

                command = self.recognizer.recognize_google(audio_data).lower()
                self.output_text.insert(tk.END, f"Command: {command}\n")
                self.output_text.update_idletasks()

                # Process the recognized command (add your logic here)
                self.process_command(command)

            except sr.UnknownValueError:
                self.output_text.insert(tk.END, "Sorry, I could not understand the command.\n")
                self.output_text.update_idletasks()
            except sr.RequestError as e:
                self.output_text.insert(tk.END, f"Error with the speech recognition service: {e}\n")
                self.output_text.update_idletasks()

        def process_command(self, command):
            # Add your logic to process the recognized command here
            # For example, you can have different if statements for different commands