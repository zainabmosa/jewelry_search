import os
import numpy as np
import torch
import torch.nn as nn
import faiss
import streamlit as st

from torchvision import models, transforms
from PIL import Image


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Jewelry Visual Search",
    page_icon="💎",
    layout="wide"
)


# ==========================================
# Paths
# ==========================================

MODEL_PATH = "mobilenetv2_embeddings.pth"
IMAGE_PATHS_PATH = "image_paths.npy"
FAISS_INDEX_PATH = "jewelry.index"

# ==========================================
# Settings
# ==========================================

TOP_K = 25


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# Load MobileNetV2
# ==========================================

@st.cache_resource
def load_model():

    model = models.mobilenet_v2(
        weights=None
    )

    # Remove classification head
    model.classifier = nn.Identity()

    # Load saved weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


# ==========================================
# Load FAISS Index
# ==========================================

@st.cache_resource
def load_index():

    return faiss.read_index(
        FAISS_INDEX_PATH
    )


# ==========================================
# Load Image Paths
# ==========================================

@st.cache_data
def load_image_paths():

    return np.load(
        IMAGE_PATHS_PATH,
        allow_pickle=True
    ).tolist()


# ==========================================
# Image Preprocessing
# ==========================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==========================================
# Extract Embedding
# ==========================================

def extract_embedding(
    image,
    model
):

    image = image.convert(
        "RGB"
    )

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    )

    image = image.to(
        device
    )

    with torch.no_grad():

        embedding = model(
            image
        )

    embedding = (
        embedding
        .cpu()
        .numpy()
        .astype("float32")
    )

    faiss.normalize_L2(
        embedding
    )

    return embedding


# ==========================================
# Load Resources
# ==========================================

model = load_model()

index = load_index()

image_paths = load_image_paths()


# ==========================================
# UI
# ==========================================

st.title(
    "💎 Jewelry Visual Search Engine"
)

st.write(
    "Upload a jewelry image or use your camera "
    "to find visually similar products."
)


# ==========================================
# Sidebar
# ==========================================

st.sidebar.header(
    "Search Settings"
)

threshold = st.sidebar.slider(
    "Similarity Threshold",

    min_value=0.0,

    max_value=1.0,

    value=0.55,

    step=0.05
)


# ==========================================
# Upload Image
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a jewelry image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ==========================================
# Camera
# ==========================================

camera_file = st.camera_input(
    "Or take a photo"
)


# ==========================================
# Select Input
# ==========================================

image_file = None

if uploaded_file is not None:

    image_file = uploaded_file

elif camera_file is not None:

    image_file = camera_file


# ==========================================
# Search
# ==========================================

if image_file is not None:

    query_image = Image.open(
        image_file
    ).convert("RGB")


    # Show uploaded image

    st.subheader(
        "Your Image"
    )

    st.image(
        query_image,
        width=300
    )


    # Extract embedding

    query_embedding = extract_embedding(
        query_image,
        model
    )


    # Search Top 25

    similarities, indices = index.search(
        query_embedding,
        TOP_K
    )


    similarities = similarities[0]

    indices = indices[0]


    # Apply threshold

    results = []

    for similarity, image_index in zip(
        similarities,
        indices
    ):

        if similarity >= threshold:

            results.append(
                (
                    float(similarity),
                    int(image_index)
                )
            )


    # ======================================
    # Display Results
    # ======================================

    if len(results) == 0:

        st.warning(
            "No sufficiently similar jewelry "
            "was found. Try another image "
            "or lower the similarity threshold."
        )

    else:

        st.subheader(
            f"Top {len(results)} Similar Products"
        )

        columns = st.columns(
            5
        )

        for position, (
            similarity,
            image_index
        ) in enumerate(results):

            with columns[
                position % 5
            ]:

                image_path = os.path.join(
                    ".",
                    image_paths[
                        image_index
                    ]
                )

                try:

                    result_image = Image.open(
                        image_path
                    ).convert(
                        "RGB"
                    )

                    st.image(
                        result_image,
                        use_container_width=True
                    )

                    st.caption(
                        f"Similarity: "
                        f"{similarity:.3f}"
                    )

                except Exception as e:

                    st.error(
                        f"Could not load image: {e}"
                    )
