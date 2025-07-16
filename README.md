# 🧠 Telugu Mental Health Sentiment Analysis using MuRIL & NLP

A machine learning project for detecting **mental health sentiment** in Telugu language texts. This includes classification into emotions such as `stress`, `normal`, `personality disorder`, `bipolar`, `anxiety`, `suicide` and `depression`. The model is built using **MuRIL embeddings**, handcrafted features, and TF-IDF vectors to detect emotional state from social media posts or user input.

---

## 📌 Project Objective

To analyze Telugu text data and classify mental health-related sentiments, supporting early detection of emotional distress using Natural Language Processing and machine learning techniques.

---

## 💡 Features

- 📜 Preprocessing of Telugu text (`cleaning`, `tokenization`)
- 🤖 Sentence embeddings using **Google's MuRIL (Multilingual BERT for Indian languages)**
- 🛠️ Handcrafted features: word counts, uniqueness, repeated words
- 🧮 TF-IDF feature extraction
- 🧠 Classification using ML models like `LightGBM`, `SVM`, `Random Forest`
- 📊 Evaluation using confusion matrix, accuracy, F1-score, and classification report
- 💾 Resumable embedding generation with checkpointing
- 🔍 Supports `stress`, `normal`, `personality disorder`, `bipolar`, `anxiety`, `suicide` and `depression` categories

---

## 🧱 Tech Stack

| Area              | Tools & Libraries                        |
|-------------------|------------------------------------------|
| Language          | Python                                   |
| NLP Embeddings    | [MuRIL (Google)](https://huggingface.co/google/muril-base-cased) |
| ML Libraries      | Scikit-learn, LightGBM, XGBoost, SVM     |
| NLP Tools         | HuggingFace Transformers, iNLTK (for Indian languages) |
| Data Handling     | Pandas, NumPy                            |
| Feature Storage   | Sparse matrices, `.npz`, `.npy`, CSV     |
| Visualizations    | Seaborn, Matplotlib                      |
| Notebook Platform | Jupyter Notebook                         |

---

## 📊 Evaluation Metrics

- **Accuracy**: Overall performance
- **Precision & Recall**: Especially useful in imbalanced classes
- **F1-score**: Balanced view of precision and recall
- **Confusion Matrix**: Visual analysis of misclassifications

---

## ⚙️ How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/telugu-nlp-sentiment-analysis.git
    cd telugu-nlp-sentiment-analysis
    ```

2. (Optional) Create a virtual environment and activate it:
    ```bash
    conda create -n nlp_env python=3.10
    conda activate nlp_env
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

5. Open `nlp_sentiment_analysis.ipynb` and run all cells.

---

## 📝 Dataset

- Telugu text-based dataset for mental health analysis
- Contains labels like `happy`, `sad`, `stress`, `anxious`, `normal`
- (Private or sourced from trusted mental health research corpora)
- **Note**: Dataset not included in the repo due to privacy concerns

---

## 🚧 Limitations

- Requires a GPU for faster MuRIL embedding generation
- Dataset availability in Telugu is limited; room for more annotated data
- Performance can be improved with more domain-specific data

---

## 🔮 Future Work

- Use fine-tuned LLMs (e.g., IndicBERT, MuRIL fine-tuned)
- Integrate attention-based architectures like BiLSTM+Attention
- Build a web app or chatbot for real-time sentiment analysis
- Translate English datasets to Telugu using multilingual models for data augmentation

---

## 🙋‍♀️ Author

**B Jyothisha**  
> Student | Developer | AI/ML Enthusiast

---

## ⭐️ Show Your Support

If you found this helpful:

- 🌟 Star this repo
- 🍴 Fork and contribute
- 🧠 Use it in your own mental health NLP projects

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).


