
import os
from pdf2image import convert_from_path
import PyPDF2
import cv2
from matplotlib import pyplot as plt
import streamlit as st
from io import BytesIO
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import pytesseract
from pytesseract import image_to_string
from transformers import BartForConditionalGeneration, BartTokenizer
import textwrap

# # Set the path to the Poppler executables (replace with your actual path)
# poppler_path = r'C:\\Users\\Vishal Pahuja\\Downloads\\Release-23.08.0-0\\poppler-23.08.0\\Library\\bin'
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# # Add the Poppler path to the system's PATH environment variable
# import os
# os.environ['PATH'] += os.pathsep + poppler_path

# # Load the BART model and tokenizer
# model_name = "facebook/bart-large-cnn"  # You can use a different BART model if needed
# model = BartForConditionalGeneration.from_pretrained(model_name)
# tokenizer = BartTokenizer.from_pretrained(model_name)

# def convert_pdf_to_img(pdf_file):
#     """
#     @desc: this function converts a PDF into Image
    
#     @params:
#         - pdf_file: the file to be converted
    
#     @returns:
#         - an iterable containing image format of all the pages of the PDF
#     """
#     return convert_from_bytes(pdf_file.read())

# def preprocess_image(image):
#     """
#     @desc: this function preprocesses the image before OCR
    
#     @params:
#         - image: the input image
    
#     @returns:
#         - the preprocessed image
#     """
#     # Example preprocessing steps (you can customize these as needed)
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
#     # Apply thresholding to enhance text
#     _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Convert NumPy array back to PIL image
#     preprocessed_image = Image.fromarray(thresholded_image)
    
#     return preprocessed_image

# def convert_image_to_text(image):
#     """
#     @desc: this function extracts text from an image
    
#     @params:
#         - image: the image to extract the content
    
#     @returns:
#         - the textual content of the image
#     """
#     text = image_to_string(image)
#     return text

# def get_text_from_any_pdf(pdf_file):
#     """
#     @desc: this function is our final system combining the previous functions
    
#     @params:
#         - pdf_file: the original PDF File
    
#     @returns:
#         - the textual content of ALL the pages
#     """
#     images = convert_pdf_to_img(pdf_file)
#     final_text = ""
    
#     for pg, img in enumerate(images):
#         # Preprocess the image
#         preprocessed_img = preprocess_image(img)
        
#         # Extract text from the preprocessed image
#         text = convert_image_to_text(preprocessed_img)
        
#         # Append text to the final result
#         final_text += text
        
#         # You can optionally add a separator between pages if needed
#         # final_text += f"--- Page {pg + 1} ---\n{text}\n"
    
#     return final_text

# def summarize_text_with_bart(text):
#     input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
#     summary_ids = model.generate(input_ids, max_length=550, min_length=520, length_penalty=2.0, num_beams=4, early_stopping=True)

#     summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summarized_text

# # def format_summarized_text(summarized_text):
# #     # Split the text into paragraphs or sentences, depending on your preference
# #     # Example: split by sentences
# #     sentences = summarized_text.split('. ')
    
# #     # Reconstruct the text with line breaks
# #     formatted_text = '\n'.join(sentences)
    
# #     return formatted_text
# # import textwrap

# def format_summarized_text(summarized_text):
#     # Split the text into paragraphs or sentences, depending on your preference
#     # Example: split by sentences
#     max_line_width=100
#     sentences = summarized_text.split('. ')
    
#     # Initialize a list to store the wrapped lines
#     wrapped_lines = []
    
#     for sentence in sentences:
#         # Wrap each sentence at word boundaries to avoid cutting words
#         lines = textwrap.wrap(sentence, width=max_line_width)
        
#         # Add the wrapped lines to the list
#         wrapped_lines.extend(lines)
    
#     # Reconstruct the text with line breaks
#     formatted_text = '\n'.join(wrapped_lines)
    
#     return formatted_text



# # Streamlit UI
# st.title("PDF Summarizer")

# # Upload PDF file
# pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if pdf_file is not None:
#     # Process the uploaded PDF
#     extracted_text = get_text_from_any_pdf(pdf_file)
    
#     # Summarize the extracted text with BART
#     summarized_text = summarize_text_with_bart(extracted_text)
    
#     # Format and display the summarized text
#     formatted_summarized_text = format_summarized_text(summarized_text)
#     st.subheader("Summarized Text")
#     st.text(formatted_summarized_text)


# import streamlit as st
# import cv2
# from matplotlib import pyplot as plt
# import pytesseract
# from PIL import Image
# import os
import streamlit as st
import cv2
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
import os
import uuid 
from pdf2image import convert_from_path

poppler_path = r'C:\\Users\\Vishal Pahuja\\Downloads\\Release-23.08.0-0\\poppler-23.08.0\\Library\\bin'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Function to perform image processing
 # Import the uuid module to generate unique filenames

# Function to generate a unique filename
def generate_unique_filename():
    unique_id = str(uuid.uuid4())
    return f"temp2/processed_{unique_id}.jpg"

# Function to perform image processing
def process_image(img_path):
    img = cv2.imread(img_path)
    
    # Your image processing steps here
    inverted_image = cv2.bitwise_not(img)
    processed_image_path = generate_unique_filename()
    # cv2.imwrite(processed_image_path, inverted_image)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("temp2/gray.jpg", gray_image)

    thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    cv2.imwrite("temp2/bw_image.jpg", im_bw)

    inverted_image = cv2.bitwise_not(img)
    cv2.imwrite("temp2/inverted.jpg", inverted_image)

    # display("temp2/inverted.jpg")

    """## 02: Rescaling"""





    """## 03: Binarization"""

    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = grayscale(img)
    cv2.imwrite("temp2/gray.jpg", gray_image)

    # display("temp2/gray.jpg")

    thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    cv2.imwrite("temp2/bw_image.jpg", im_bw)

    # display("temp2/bw_image.jpg")

    """## 04: Noise Removal"""

    def noise_removal(image):
        import numpy as np
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)

    no_noise = noise_removal(im_bw)
    cv2.imwrite("temp2/no_noise.jpg", no_noise)

    # display("temp2/no_noise.jpg")

    """## Dilation and Erosion"""

    def thin_font(image):
        import numpy as np
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    eroded_image = thin_font(no_noise)
    cv2.imwrite("temp2/eroded_image.jpg", eroded_image)

    # display("temp2/eroded_image.jpg")

    def thick_font(image):
        import numpy as np
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    dilated_image = thick_font(no_noise)
    cv2.imwrite("temp2/dilated_image.jpg", dilated_image)

    # display("temp2/dilated_image.jpg")
    is_color_image = len(img.shape) == 3 and img.shape[2] == 3

    is_color_image = len(img.shape) == 3 and img.shape[2] == 3

    if is_color_image:
            # Color image preprocessing steps
            inverted_image = cv2.bitwise_not(img)
            text_image = noise_removal(inverted_image)
    else:
            # Grayscale image preprocessing steps
            gray_image = grayscale(img)
            text_image = noise_removal(gray_image)



    """## 07: Removing Borders"""

    # display("temp2/no_noise.jpg")

    def remove_borders(image):
        contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
        cnt = cntsSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y+h, x:x+w]
        return (crop)

    no_borders = remove_borders(no_noise)
    cv2.imwrite("temp2/no_borders.jpg", no_borders)
    # display('temp2/no_borders.jpg')

    """## 08: Missing Borders"""

    color = [255, 255, 255]
    top, bottom, left, right = [150]*4

    image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite("temp2/image_with_border.jpg", image_with_border)

    cv2.imwrite(processed_image_path, image_with_border)

    # ... other processing steps ...

    return processed_image_path  # Return the path to the processed image

# Function to perform OCR
def perform_ocr(processed_image_path):
    img = Image.open(processed_image_path)
    ocr_result = pytesseract.image_to_string(img)
    return ocr_result

poppler_path = r'C:\\Users\\Vishal Pahuja\\Downloads\\Release-23.08.0-0\\poppler-23.08.0\\Library\\bin'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Add the Poppler path to the system's PATH environment variable
os.environ['PATH'] += os.pathsep + poppler_path

# Load the BART model and tokenizer
# You can use a different BART model if needed
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)


def convert_pdf_to_img(pdf_file):
    """
    @desc: this function converts a PDF into Image

    @params:
        - pdf_file: the file to be converted

    @returns:
        - an iterable containing image format of all the pages of the PDF
    """
    return convert_from_path(pdf_file)


def preprocess_image(image):
    """
    @desc: this function preprocesses the image before OCR

    @params:
        - image: the input image

    @returns:
        - the preprocessed image
    """
    # Example preprocessing steps (you can customize these as needed)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text
    _, thresholded_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert NumPy array back to PIL image
    preprocessed_image = Image.fromarray(thresholded_image)

    return preprocessed_image


def convert_image_to_text(image):
    """
    @desc: this function extracts text from an image

    @params:
        - image: the image to extract the content

    @returns:
        - the textual content of the image
    """
    text = image_to_string(image)
    return text


def get_text_from_any_pdf(pdf_file):
    """
    @desc: this function is our final system combining the previous functions

    @params:
        - pdf_file: the original PDF File

    @returns:
        - the textual content of ALL the pages
    """
    images = convert_pdf_to_img(pdf_file)
    final_text = ""

    for pg, img in enumerate(images):
        # Preprocess the image
        preprocessed_img = preprocess_image(np.array(img))

        # Extract text from the preprocessed image
        text = convert_image_to_text(preprocessed_img)

        # Append text to the final result
        final_text += text

        # You can optionally add a separator between pages if needed
        # final_text += f"--- Page {pg + 1} ---\n{text}\n"

    return final_text


# Example usage:

# Example usage:
pdf_file_path = "C:\\Users\\Vishal Pahuja\\Documents\\Vishal\\python\\Operations Management.pdf"
extracted_text = get_text_from_any_pdf(pdf_file_path)
print(extracted_text)

def main():
    st.title("Image Processing and OCR App")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Create a folder to store processed images
        os.makedirs("temp2", exist_ok=True)

        # Save the uploaded image
        image_path = "temp2/uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            st.text("Processing Image...")
            processed_image_path = process_image(image_path)
            st.image(processed_image_path, caption="Processed Image", use_column_width=True)

            ocr_result = perform_ocr(processed_image_path)
            st.header("Extracted Text:")
            st.write(ocr_result)

    if file_type.startswith('image'):
       # Display the uploaded image
       st.title("Text Extraction from Images and PDFs")

# Upload image or PDF file
        
        os.makedirs("temp2", exist_ok=True)
        # st.title("Image Processing and OCR App")
        uploaded_image = st.file_uploader("Upload an image or PDF", type=[
                                 "jpg", "jpeg", "png", "pdf"])
        
        # Save the uploaded image with a unique filename
        image_path = generate_unique_filename()
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            st.text("Processing Image...")
            processed_image_path = process_image(image_path)
            st.image(processed_image_path, caption="Processed Image", use_column_width=True)

            ocr_result = perform_ocr(processed_image_path)
            st.header("Extracted Text:")
            st.write(ocr_result)

    elif file_type == 'application/pdf':
        # Convert the PDF to images
        pdf_images = convert_from_bytes(
            uploaded_file.read(), poppler_path=poppler_path)

        # Initialize a placeholder for extracted text from all PDF pages
        pdf_text = ""

        # Perform OCR on each page of the PDF
        for page_num, pdf_image in enumerate(pdf_images, 1):
            st.header(f"Page {page_num}")
            st.image(
                pdf_image, caption=f"Page {page_num}", use_column_width=True)

            # Perform OCR on the PDF page image
            page_text = pytesseract.image_to_string(pdf_image)

            # Append page text to the overall text
            pdf_text += page_text + "\n"

        # Display the extracted text from the entire PDF
        st.header("Extracted Text from PDF")
        st.write(pdf_text)