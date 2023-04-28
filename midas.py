import torch
import cv2
import urllib.request
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Define a function to process the input image
def process_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        prediction = prediction - prediction.min()  # set the minimum value to 0
        prediction = prediction / prediction.max()  # set the maximum value to 1
        prediction = (prediction * 255).cpu().numpy().astype(np.uint8)  # convert to 8-bit grayscale
        prediction = cv2.applyColorMap(prediction, cv2.COLORMAP_HOT)  # apply color map
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)  # convert to RGB
    output = prediction
    return output




# Set up the Streamlit app
st.title("MiDaS Depth Estimation App")

# Create an input file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If the user has uploaded an image, display the input and output images
if uploaded_file is not None:
    # Load the image and process it
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    output = process_image(image)
    
    # Display the input and output images
    st.subheader("Input Image")
    st.image(image, clamp=True,caption="Input Image", use_column_width=True)
    st.subheader("Output Image")
    st.image(output, clamp=True,caption="Depth Map", use_column_width=True)
