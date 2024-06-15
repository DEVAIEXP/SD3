import csv
import datetime
import os
import platform
from PIL import Image, PngImagePlugin

def read_image_metadata(image_path):
    if image_path is None or not os.path.exists(image_path):
        return "File does not exist or path is None."
    
    
    last_modified_timestamp = os.path.getmtime(image_path)

    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')
    with Image.open(image_path) as img:
        metadata = img.info
        metadata_str = f"Last Modified Date: {last_modified_date}\n"
        for key, value in metadata.items():
            metadata_str += f"{key}: {value}\n"
            
    return metadata_str

    
def save_image_with_metadata(image, filename, metadata):
    meta_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        meta_info.add_text(key, str(value))
    image.save(filename, "PNG", pnginfo=meta_info)   



def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')
        
def load_styles():
    styles = {"No Style": ("", "")}
    try:
        with open('styles.csv', mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) == 3:
                    styles[row[0]] = (row[1], row[2])
    except Exception as e:
        print(f"Failed to load styles from CSV: {e}")
    return styles