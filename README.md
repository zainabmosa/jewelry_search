# 💎 Jewelry Visual Search Engine

## About the Project

In this project, I built a Jewelry Visual Search Engine using the **Tanishq Jewellery Dataset**.

The main idea is that the user can upload a jewelry image, and the system will find similar jewelry images from the dataset.

I used **MobileNetV2** to extract image features and **FAISS** to search for similar images.

---

## 🎯 Project Goal

The goal of this project is to create a simple visual search system.

Instead of searching using text, the user can upload an image of jewelry, and the system will show visually similar items.

---

## 🔍 How It Works

The project works in the following steps:

1. Load the jewelry images.
2. Use MobileNetV2 to extract features from each image.
3. Convert the image features into embeddings.
4. Save the embeddings and image paths.
5. Create a FAISS index.
6. Upload a query image.
7. Search for similar jewelry images.
8. Display the Top similar results.

---

## 🤖 Model Used

I used **MobileNetV2** with pretrained weights.

The classification layer was removed, and the model was used as a feature extractor.

This allows the system to compare images based on their visual features.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* MobileNetV2
* FAISS
* NumPy
* Pillow
* Streamlit

---

## 🖥️ Streamlit Application

I created a Streamlit application where the user can:

* Upload a jewelry image
* Use the camera to take a photo
* Search for similar jewelry
* View similar results from the dataset

### 🚀 Try the App

[Open Jewelry Visual Search App](https://jewelrysearch1.streamlit.app/?utm_source=chatgpt.com)

---

## 📁 Project Structure

```text
jewelry_search/
│
├── jewelry_visual_search.ipynb
├── prepare_data.py
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── images/
    ├── embeddings.npy
    ├── image_paths.npy
    └── jewelry.index
```

---

## ⚙️ Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

First, prepare the data and create the embeddings:

```bash
python prepare_data.py
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🔎 Visual Search

The application uses the uploaded image as a query.

The image is converted into an embedding using MobileNetV2.

Then FAISS compares the query embedding with the saved embeddings and returns similar jewelry images.

The system displays the **Top 25 similar results**.

---

## 👩‍💻 Author

**Zainab Mohammed**

**Data Science**
