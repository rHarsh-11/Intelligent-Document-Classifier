
# 📄 Intelligent Document Classifier (20 Newsgroups NLP Project)

An **NLP-powered document classification system** that automatically categorizes text into one of 20 predefined topics using **TF-IDF** and **Logistic Regression**.  
The project uses the **20 Newsgroups dataset** and includes a **Streamlit web app** for real-time classification.

---

## 🚀 Features
- **Text Preprocessing** (lowercasing, tokenization, stopword removal)
- **TF-IDF Vectorization** for feature extraction
- **Multi-class Logistic Regression** for classification
- **Evaluation Metrics**: Precision, Recall, F1-score
- **Confusion Matrix Visualization**
- **Streamlit UI** for interactive predictions
- **Custom Input Prediction** for any text

---

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:**
  - `scikit-learn` (ML & evaluation)
  - `nltk` (tokenization, stopwords)
  - `pandas`, `numpy` (data handling)
  - `matplotlib`, `seaborn` (visualization)
  - `streamlit` (web UI)
  - `joblib` (model saving/loading)

---

## 📂 Project Structure
```

intelligent\_document\_classifier/
│── models/
│   ├── document\_classifier\_model.pkl
│   ├── tfidf\_vectorizer.pkl
│── notebooks/
│   ├── Regression_model.ipynb
│   ├── Model_test.ipynb
│── app.py                  # Streamlit application
│── requirements.txt
│── .gitignore
│── README.md

````

---

## 📊 Dataset
We use the **[20 Newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)** from Scikit-learn,  
which contains ~18,000 documents across 20 categories:
- Religion, Politics, Science, Sports, Computers, and more.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rHarsh-11/Intelligent-Document-Classifier
cd Intelligent-Document-Classifier
````

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK Data

```bash
python -m nltk.downloader stopwords punkt
```

---

## ▶️ Running the Jupyter Notebook

```bash
jupyter notebook notebooks/document_classifier.ipynb
```

---

## 🌐 Running the Streamlit App

```bash
streamlit run app.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📈 Example Output

* **Prediction:**

```
Predicted Category: rec.sport.hockey
```

---

Do you want me to also create the **`requirements.txt`** right now so that once you push this with README, the Streamlit Cloud deployment will work without missing dependencies? That’s the next logical step.
```
