import os
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

device = "cuda" if torch.cuda.is_available() else "cpu"
converter = PdfConverter(artifact_dict=create_model_dict())

def parse_and_save(pdf_path, output_root="parsed_output"):
    filename = os.path.basename(pdf_path)
    file_stem = os.path.splitext(filename)[0]
    
    book_output_folder = os.path.join(output_root, file_stem)
    images_folder = os.path.join(book_output_folder, "images")
    
    os.makedirs(images_folder, exist_ok=True)
    
    print(f"processing: {filename}")
    
    rendered = converter(pdf_path) # trigger pipeline of models to transform pdf into text
    text, _, images = text_from_rendered(rendered) # separates the result into clean markdown text and a dictionary of images found in the document
    
    for img_filename, image in images.items():
        save_path = os.path.join(images_folder, img_filename)
        image.save(save_path)

    md_path = os.path.join(book_output_folder, f"{file_stem}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"saved to: {book_output_folder}")

def main():
    input_folder = "books"
    output_base = "parsed_output"

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

    print(f"found {len(pdf_files)} pdf(s). starting batch processing")

    for pdf_file in pdf_files: # take one file per time
        full_pdf_path = os.path.join(input_folder, pdf_file)
        parse_and_save(full_pdf_path, output_base)

    print("batch processing complete")

if __name__ == "__main__":
    main()