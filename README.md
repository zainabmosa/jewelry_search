# 💎 Jewelry Visual Search Engine

## 🚀 Live Demo

🔗 **Streamlit App:**
https://jewelrysearch1.streamlit.app/

## About the Project

In this project, I built a visual search engine using the Tanishq Jewellery Dataset.

The user can upload a jewelry image, and the system finds visually similar jewelry from the dataset.

---

## What I Used

- MobileNetV2 for feature extraction
- Transfer Learning
- Image embeddings
- FAISS for similarity search
- Streamlit for the web application

---

## How It Works

1. The images are loaded from the dataset.
2. MobileNetV2 extracts an embedding from each image.
3. The embeddings are saved.
4. FAISS is used to search for similar images.
5. The user uploads an image.
6. The app shows the top similar results.

---

## 📁 Project Structure

```text
jewelry_search/
│
├── jewelry_visual_search_student_style.ipynb
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

## ▶️ Run the Project

First, install the required libraries:

```bash
pip install -r requirements.txt
```

If you need to create the embeddings and FAISS index:

```bash
python prepare_data.py
```

Then run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🔍 Search Results

The app:

- Lets the user upload an image or use the camera
- Displays the uploaded image
- Searches for the top 25 similar items
- Uses a similarity threshold to avoid showing unrelated results

---

## 📊 Quality Check

I also added a simple category-based Precision@5 check in the notebook.

---

## 👩‍💻 Author

**Zainab Mohammed**  
**Data Science**
