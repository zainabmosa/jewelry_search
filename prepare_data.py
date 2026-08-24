import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import faiss

from PIL import Image
from torchvision import models, transforms

st.set_page_config(
    page_title="Jewelry Finder",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Jewelry Visual Search")
st.write("Upload a jewelry photo and find similar items from the catalog.")


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

TOP_K = 25
THRESHOLD = 0.55


@st.cache_resource
def load_search_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Remove the classification head
    model.classifier = nn.Identity()

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")
    paths_path = os.path.join(DATA_DIR, "image_paths.npy")
    index_path = os.path.join(DATA_DIR, "jewelry.index")

    embeddings = np.load(embeddings_path)
    image_paths = np.load(paths_path, allow_pickle=True)
    index = faiss.read_index(index_path)

    return model, transform, index, image_paths, device


def get_image_embedding(image, model, transform, device):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor).cpu().numpy().astype("float32")

    faiss.normalize_L2(embedding)

    return embedding


# Load model and search files
try:
    model, transform, index, image_paths, device = load_search_engine()
except Exception as e:
    st.error("Could not load the search files.")
    st.code(str(e))
    st.stop()


tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a jewelry image",
        type=["jpg", "jpeg", "png", "webp"]
    )

with tab2:
    camera_file = st.camera_input("Take a photo")


query_file = uploaded_file if uploaded_file is not None else camera_file


if query_file is not None:

    query_image = Image.open(
        io.BytesIO(query_file.getvalue())
    ).convert("RGB")

    st.subheader("Your Image")
    st.image(query_image, width=300)

    if st.button("🔍 Find Similar Jewelry", use_container_width=True):

        with st.spinner("Searching..."):
            query_embedding = get_image_embedding(
                query_image,
                model,
                transform,
                device
            )

            similarities, indices = index.search(
                query_embedding,
                TOP_K
            )

        similarities = similarities[0]
        indices = indices[0]

        valid_results = [
            (int(i), float(score))
            for i, score in zip(indices, similarities)
            if i != -1 and score >= THRESHOLD
        ]

        st.divider()

        if len(valid_results) == 0:
            st.warning(
                "No similar jewelry was found above the similarity threshold."
            )

        else:
            st.subheader(f"✨ Similar Results ({len(valid_results)})")

            cols = st.columns(5)

            for rank, (image_index, score) in enumerate(valid_results):

                rel_path = str(image_paths[image_index])
                full_path = os.path.join(PROJECT_DIR, rel_path)

                if os.path.exists(full_path):
                    result_image = Image.open(full_path).convert("RGB")

                    with cols[rank % 5]:
                        st.image(
                            result_image,
                            caption=f"#{rank + 1} | Similarity: {score:.3f}",
                            use_container_width=True
                        )

else:
    st.info("👆 Upload a jewelry image or use the camera to start searching.")


st.divider()
st.caption("Built using MobileNetV2, embeddings, FAISS, and Streamlit.")
