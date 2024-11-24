import io
import pandas as pd
from PIL import Image
import streamlit as st
from model import detect_classes

st.set_page_config(
    page_title="Side Menu Example",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* Change the font in the sidebar */
    .css-1d391kg {  /* Sidebar class */
        font-family: Ubuntu;  /* Change to desired font */
        font-size: 16px;  /* Adjust font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Choose Weights")
option = st.sidebar.selectbox(
    "Choose weights:",
    ["Last", "Best"]
)

st.markdown("<h1 style='font-size: 38px; font-family:Ubuntu ; text-align:center;'>Object Detection with YOLOv10 on BCCD Dataset</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:18px; font-family:Ubuntu ; text-align:center;'>The following application performs object detection, i.e. in this case blood cell type detection between the classes RBC, WBC and Platelets, draws bounding boxes around them and displays the confidence score and class of each box within it. It also displays the overall precison and overall recall of the model as well as precison and recall on each class.</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:18px; font-family:Ubuntu ; text-align:center;'>The model has two weights, Best and Last(default).</br> Toggle between the weights by going to the sidebar and choosing the weight that suits you best.</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:20px; font-family:Ubuntu ; text-align:center;'>Upload a sample image to perfrom object detection on the image</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stButton {
        display: flex;
        justify-content: center;  /* Center horizontally */
    }
    .stButton > button {
        width: 200px;  /* Set button width */
        height: 50px;  /* Set button height */
        font-size: 24px;  /* Adjust font size */
        font-weight:bold;
        background-color: #4f1323;  /* Button background color */
        color: white;  /* Text color */
        border: 0.1px solid;  /* Remove border */
        border-color: white;
        border-radius: 8px;  /* Rounded corners */
        cursor: pointer;  /* Pointer cursor on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stImage {
        border: 1px solid white;
        padding: 5px;
        display: inline-block;
    }
    .stSubheader{
        text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def display_images_metrics(uploaded_image, processed_image, class_metrics, overall_metrics):
    st.markdown("<h1 style='font-size:20px; font-family:Ubuntu ; text-align:center;'></br>Results</h1>", unsafe_allow_html=True)
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(uploaded_image, caption="Original Image", width = 600,use_container_width=False)
    with img_col2:
        st.image(processed_image, caption="Processed Image with class detections", width = 600,use_container_width=False)

    st.markdown("<h1 style='font-size:20px; font-family:Ubuntu ; text-align:center;'></br>Model Metrics</h1>", unsafe_allow_html=True)

    key_mapping = {0: "RBC", 1: "WBC", 2: "Platelets"}
    mapped_class_metrics = {key_mapping[k]: v for k, v in class_metrics.items()}
    df_class_metrics = pd.DataFrame.from_dict(mapped_class_metrics, orient="index").reset_index()
    df_class_metrics.rename(columns={"index": "Classes"}, inplace=True)

    df_overall_metrics = pd.DataFrame(overall_metrics, index=['overall'])

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.markdown("<h1 style='font-size:20px; font-family:Ubuntu ; text-align:center;'>Class Metrics</h1>", unsafe_allow_html=True)
        st.table(df_class_metrics)
    with metrics_col2:
        st.markdown("<h1 style='font-size:20px; font-family:Ubuntu ; text-align:center;'>Overall Metrics</h1>", unsafe_allow_html=True)
        st.table(df_overall_metrics)
        
def download_processed_image(image, filename):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    st.markdown(
        """
        <style>
        .stDownloadButton > button {
            font-size: 24px;  /* Adjust font size */
            height: 50px;
            font-weight:bold;
            background-color: #4f1323;  /* Button background color */ 
            color: white;  /* Text color */
            border: 0.1px solid;  /* Remove border */
            border-color: white;
            border-radius: 8px;  /* Rounded corners */
            cursor: pointer;  /* Pointer cursor on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        label="⤓ Download Processed Image",
        data=img_byte_arr,
        file_name=filename[:-4],
        mime="image/jpeg",
        use_container_width=True
    )

class_metrics, overall_metrics = None, None
uploaded_file = st.file_uploader(".", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    file_name = uploaded_file.name
    uploaded_image = Image.open(uploaded_file)
    st.write("Image uploaded successfully!")
    if st.button("🔍 Detect"):
        if option == "Last":
            class_metrics, overall_metrics, processed_img = detect_classes(uploaded_image,"./results/",file_name[:-4],'last')
            display_images_metrics(uploaded_image, processed_img, class_metrics, overall_metrics)
            download_processed_image(processed_img,f"processed_{file_name}.jpg")
        elif option == "Best":
            class_metrics, overall_metrics, processed_img = detect_classes(uploaded_image,"./results/",file_name[:-4],'best')
            display_images_metrics(uploaded_image, processed_img, class_metrics, overall_metrics)
            download_processed_image(processed_img,f"processed_{file_name}.jpg")
    else:
        pass
