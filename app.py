import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import faiss

from PIL import Image
from torchvision import models, transforms


# -------------------------
# App settings
# -------------------------

st.set_page_config(
    page_title="Jewelry Finder",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Jewelry Visual Search")
st.write("Upload a jewelry photo and find similar items from the catalog.")


# -------------------------
# Paths and settings
# -------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

TOP_K = 25
THRESHOLD = 0.55


# -------------------------
# Load model and search files
# -------------------------

@st.cache_resource
def load_search_engine():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load MobileNetV2
    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(
        weights=weights
    )

    # Remove classification layer
    model.classifier = nn.Identity()

    model = model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # File paths
    embeddings_path = os.path.join(
        DATA_DIR,
        "embeddings.npy"
    )

    paths_path = os.path.join(
        DATA_DIR,
        "image_paths.npy"
    )

    index_path = os.path.join(
        DATA_DIR,
        "jewelry.index"
    )

    # Load saved files
    embeddings = np.load(
        embeddings_path
    )

    image_paths = np.load(
        paths_path,
        allow_pickle=True
    )

    index = faiss.read_index(
        index_path
    )

    return (
        model,
        transform,
        index,
        image_paths,
        device
    )


# -------------------------
# Create query embedding
# -------------------------

def get_image_embedding(
    image,
    model,
    transform,
    device
):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        embedding = model(tensor).cpu().numpy().astype(
            "float32"
        )

    # Normalize embedding
    faiss.normalize_L2(embedding)

    return embedding


# -------------------------
# Find the correct image path
# -------------------------

def get_full_image_path(saved_path):

    saved_path = str(saved_path)

    # 1. If it is already a correct path
    if os.path.exists(saved_path):
        return saved_path

    # 2. Normal relative path
    possible_path = os.path.join(
        PROJECT_DIR,
        saved_path
    )

    if os.path.exists(possible_path):
        return possible_path

    # 3. If the path was saved from Google Colab
    # Example:
    # /content/drive/MyDrive/jewelry_search/data/images/...
    marker = "data/images/"

    normalized_path = saved_path.replace(
        "\\",
        "/"
    )

    if marker in normalized_path:

        image_part = normalized_path.split(
            marker,
            1
        )[1]

        possible_path = os.path.join(
            DATA_DIR,
            "images",
            image_part
        )

        if os.path.exists(possible_path):
            return possible_path

    # If nothing works
    return None


# -------------------------
# Load search engine
# -------------------------

try:

    (
        model,
        transform,
        index,
        image_paths,
        device
    ) = load_search_engine()

except Exception as e:

    st.error("Could not load the search files.")
    st.code(str(e))
    st.stop()


# -------------------------
# Upload options
# -------------------------

tab1, tab2 = st.tabs([
    "📁 Upload Image",
    "📷 Use Camera"
])


with tab1:

    uploaded_file = st.file_uploader(
        "Choose a jewelry image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


with tab2:

    camera_file = st.camera_input(
        "Take a photo"
    )


# Choose uploaded image
query_file = (
    uploaded_file
    if uploaded_file is not None
    else camera_file
)


# -------------------------
# Search
# -------------------------

if query_file is not None:

    query_image = Image.open(
        io.BytesIO(
            query_file.getvalue()
        )
    ).convert("RGB")


    st.subheader("Your Image")

    st.image(
        query_image,
        width=300
    )


    if st.button(
        "🔍 Find Similar Jewelry",
        use_container_width=True
    ):

        with st.spinner(
            "Searching..."
        ):

            # Create query embedding
            query_embedding = get_image_embedding(
                query_image,
                model,
                transform,
                device
            )

            # Search FAISS
            similarities, indices = index.search(
                query_embedding,
                TOP_K
            )


        similarities = similarities[0]
        indices = indices[0]


        # Filter by threshold
        valid_results = [

            (
                int(image_index),
                float(score)
            )

            for image_index, score in zip(
                indices,
                similarities
            )

            if image_index != -1
            and score >= THRESHOLD

        ]


        st.divider()


        # -------------------------
        # No results
        # -------------------------

        if len(valid_results) == 0:

            st.warning(
                "No similar jewelry was found above "
                "the similarity threshold."
            )


        # -------------------------
        # Show results
        # -------------------------

        else:

            st.subheader(
                f"✨ Similar Results ({len(valid_results)})"
            )

            cols = st.columns(5)

            shown_results = 0
            missing_images = []


            for rank, (
                image_index,
                score
            ) in enumerate(valid_results):


                # Get saved path
                saved_path = image_paths[
                    image_index
                ]


                # Find correct image
                full_path = get_full_image_path(
                    saved_path
                )


                # If image exists
                if full_path is not None:

                    try:

                        result_image = Image.open(
                            full_path
                        ).convert("RGB")


                        with cols[
                            shown_results % 5
                        ]:

                            st.image(
                                result_image,
                                caption=(
                                    f"#{shown_results + 1} "
                                    f"| Similarity: "
                                    f"{score:.3f}"
                                ),
                                use_container_width=True
                            )


                        shown_results += 1


                    except Exception as e:

                        missing_images.append(
                            str(saved_path)
                        )


                else:

                    missing_images.append(
                        str(saved_path)
                    )


            # Show warning if images are missing
            if shown_results == 0:

                st.error(
                    "The search worked, but the image "
                    "files could not be found."
                )

                st.write(
                    "The first saved image path is:"
                )

                if len(valid_results) > 0:

                    first_index = valid_results[0][0]

                    st.code(
                        str(
                            image_paths[
                                first_index
                            ]
                        )
                    )


            elif len(missing_images) > 0:

                st.info(
                    f"Showing {shown_results} images. "
                    f"Some image files could not be found."
                )


            else:

                st.success(
                    f"Found {shown_results} similar jewelry items!"
                )


else:

    st.info(
        "👆 Upload a jewelry image or use "
        "the camera to start searching."
    )


# -------------------------
# Footer
# -------------------------

st.divider()

st.caption(
    "Built using MobileNetV2, embeddings, "
    "FAISS, and Streamlit."
)
