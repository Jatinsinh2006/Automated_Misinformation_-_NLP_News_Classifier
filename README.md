# 📰 Fake News Detection

An end-to-end Machine Learning + NLP project that classifies news articles as **Fake** or **Real**, using the ISOT Fake News Dataset, TF-IDF vectorization, and classical ML models — deployed as an interactive Streamlit web app.

## 📂 Dataset

**[ISOT Fake News Dataset (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)**

Contains two CSV files:
- `Fake.csv` — fake news articles
- `True.csv` — real news articles

> Download both files from Kaggle and place them inside the `data/` folder before running `train.py`.

## 📁 Project Structure

```
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── processed_news.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── model_accuracy.png
│   └── project_report.pdf
│
├── streamlit_app.py
├── train.py
├── predict.py
├── preprocessing.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 🛠️ Technologies Used

- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, Streamlit, Pickle
- **ML Models:** TF-IDF Vectorizer, Logistic Regression, Multinomial Naive Bayes, SVM
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 🔄 Workflow

```
Dataset Collection → Data Cleaning → EDA → Text Preprocessing
    (lowercase, remove punctuation/numbers, tokenize, stopword removal, stemming)
    → TF-IDF Vectorization → Train-Test Split
    → Model Training (LR, Naive Bayes, SVM)
    → Model Evaluation → Save Best Model (.pkl)
    → Streamlit Deployment → User enters News → Prediction (Fake / Real)
```

## 🚀 Getting Started

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd Fake-News-Detection
pip install -r requirements.txt
```

### 2. Add the dataset

Download `Fake.csv` and `True.csv` from Kaggle and place them in `data/`.

### 3. Train the models

```bash
python train.py
```

This will preprocess the data, train all three models, evaluate them, save the best-performing model (`models/best_model.pkl`), the TF-IDF vectorizer (`models/tfidf_vectorizer.pkl`), and generate reports in `reports/`.

### 4. Predict from the command line

```bash
python predict.py "Some news article text here..."
```

### 5. Run the web app

```bash
streamlit run streamlit_app.py
```

## 📊 Evaluation

Models are compared on Accuracy, Precision, Recall, and F1-Score. The best-performing model (by F1-score) is automatically selected and saved for deployment. See `reports/classification_report.txt` and `reports/model_accuracy.png` after training.

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.