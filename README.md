# 💎 Jewelry Visual Search Engine

A **Computer Vision-based Visual Search Engine** that allows users to upload a jewelry image or capture one using their camera and retrieve visually similar jewelry products.

The system uses **MobileNetV2** to extract visual features from jewelry images and **FAISS** to perform fast similarity search based on image embeddings.

## 🚀 Live Demo

🔗 **Streamlit App:**
https://jewelrysearch1.streamlit.app/

## 📌 Project Overview

Traditional product search usually depends on keywords, product names, or descriptions. This project provides an alternative approach using **image-based similarity search**.

Users can upload a jewelry image, and the system will:

1. Process the uploaded image.
2. Extract its visual features using MobileNetV2.
3. Convert the image into a numerical embedding.
4. Search for similar embeddings using FAISS.
5. Return the most visually similar jewelry products.

## 🎯 Objectives

* Build a visual search engine for jewelry products.
* Use deep learning for image feature extraction.
* Implement similarity search using FAISS.
* Allow users to search using uploaded images.
* Support camera-based image input.
* Deploy the application using Streamlit.

## 🗂️ Dataset

The dataset contains **490 jewelry images**, organized into two main categories:

* Necklace
* Ring

The images are used to generate deep-learning embeddings for visual similarity search.

## 🧠 Methodology

The project follows the pipeline below:

```text
Jewelry Images
       ↓
Image Preprocessing
       ↓
MobileNetV2
       ↓
Feature Embeddings
       ↓
L2 Normalization
       ↓
FAISS Index
       ↓
Similarity Search
       ↓
Top Similar Jewelry Images
```

### 1. Image Preprocessing

Each image is:

* Converted to RGB.
* Resized to `224 × 224`.
* Converted to a PyTorch tensor.
* Normalized using ImageNet mean and standard deviation.

### 2. Feature Extraction

**MobileNetV2** is used as a feature extractor.

The original classification layer is removed, allowing the network to generate feature embeddings instead of classification predictions.

### 3. Embeddings

Each jewelry image is represented by a numerical feature vector.

The embeddings are normalized using FAISS L2 normalization before indexing.

### 4. Similarity Search

The generated embeddings are stored in a **FAISS IndexFlatIP** index.

The system uses inner-product similarity on normalized embeddings to identify visually similar jewelry images.

The application retrieves the **Top 25** results and applies a configurable similarity threshold.

## 🛠️ Technologies Used

| Technology  | Purpose                                   |
| ----------- | ----------------------------------------- |
| Python      | Main programming language                 |
| PyTorch     | Deep learning and feature extraction      |
| Torchvision | MobileNetV2 model and image preprocessing |
| FAISS       | Vector similarity search                  |
| NumPy       | Numerical data and embeddings             |
| Pillow      | Image processing                          |
| Streamlit   | Web application and deployment            |
| GitHub      | Source code and project storage           |

## 📁 Project Structure

```text
jewelry_search/
│
├── app.py
├── prepare_data.py
├── requirements.txt
├── README.md
│
├── embeddings.npy
├── image_paths.npy
├── jewelry.index
├── mobilenetv2_embeddings.pth
│
└── data/
    └── images/
        ├── necklace/
        └── ring/
```

### File Description

| File                         | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `app.py`                     | Main Streamlit application                     |
| `prepare_data.py`            | Generates image embeddings and the FAISS index |
| `requirements.txt`           | Required Python libraries                      |
| `embeddings.npy`             | Stored image embeddings                        |
| `image_paths.npy`            | Paths corresponding to the image embeddings    |
| `jewelry.index`              | FAISS similarity-search index                  |
| `mobilenetv2_embeddings.pth` | Saved MobileNetV2 feature-extraction model     |
| `data/images/`               | Jewelry image dataset                          |

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/zainabmosa/jewelry_search.git
```

Move into the project directory:

```bash
cd jewelry_search
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application Locally

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 🔍 How to Use

1. Open the application.
2. Upload a jewelry image using the file uploader, or take a picture using the camera option.
3. The image is processed by MobileNetV2.
4. The application searches the FAISS index.
5. The most visually similar jewelry products are displayed.
6. Adjust the **Similarity Threshold** from the sidebar to control how similar the returned images need to be.

## 📊 Search Settings

The application provides a similarity threshold ranging from `0.0` to `1.0`.

The default threshold is:

```text
0.55
```

A higher threshold returns fewer but more visually similar results, while a lower threshold allows more results with weaker similarity.

## ☁️ Deployment

The application is deployed using **Streamlit** and connected to the GitHub repository.

### Live Application

https://jewelrysearch1.streamlit.app/

## 🔮 Future Improvements

Possible future improvements include:

* Adding more jewelry categories and products.
* Improving similarity accuracy using a fine-tuned model.
* Adding product metadata such as price, material, and product ID.
* Displaying product names and additional information.
* Implementing advanced filtering.
* Using a more powerful vision model for improved feature representations.
* Adding a database for product information.

## 👩🏻‍💻 Author

**Zainab Mohamed Moosa**

General Assembly
---

### ⭐ Project Summary

This project demonstrates the use of **Deep Learning, Computer Vision, Feature Embeddings, Vector Similarity Search, and Streamlit Deployment** to build an end-to-end jewelry visual search application.
