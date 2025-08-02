# 🔄 Customer Churn Prediction

ML-powered app to predict telecom customer churn with SHAP-based explanations and real-time Streamlit interface.

## 📖 About

This project predicts the **likelihood of a customer churning** (leaving a telecom service) using a Gradient Boosting model trained on telecom customer data. It includes an interactive **Streamlit web application** for:

* Real-time single/batch predictions
* Visual analytics
* SHAP-based feature explanations

The model was trained using exploratory data analysis (EDA), preprocessing, and threshold tuning in the `churn_eda_and_model.ipynb` notebook.

## ✨ Features

* 📊 Predict churn for individual or bulk customer data
* 🔍 SHAP visualization for **explainable AI**
* 🧠 Optimized decision threshold for balanced performance
* 📈 Visual breakdown of churn by key customer segments
* 💾 Downloadable prediction outputs

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend & Model:** Scikit-learn, LightGBM
* **EDA & Visualization:** Pandas, Seaborn, Matplotlib
* **Model Explainability:** SHAP
* **Others:** Joblib, Numpy, Imbalanced-learn

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/shubhruia/customer_churn_prediction.git
cd customer-churn-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Project Structure

```
├── app.py                     # Streamlit app
├── churn_eda_and_model.ipynb  # Model training & EDA notebook
├── model/
│   ├── churn_model.pkl        # Trained ML model
│   ├── feature_columns.pkl    # Features used by the model
│   └── best_threshold.txt     # Optimal probability threshold
├── requirements.txt
```

> ⚠️ Ensure the model files (`.pkl`, `.txt`) are in the `model/` folder as expected by `app.py`.

### 5. Run the App

```bash
streamlit run app.py
```

## 🚀 How to Use

1. Launch the app with Streamlit.
2. Use the **Single Prediction** tab to manually enter customer info.
3. Upload a CSV in the **Batch Prediction** tab for bulk scoring.
4. Use the **Visual Analysis** tab for churn insights and EDA plots.
5. Get model explanations with **SHAP** for each individual prediction.

## 🧪 Model Overview

* **Algorithm:** Gradient Boosting Classifier (via LightGBM)
* **Optimized Threshold:** \~0.48
* **Test Set Performance:**

  * ROC-AUC: 89.6%
  * Accuracy: 82.7%
  * Precision (Churn): 80%
  * Recall (Churn): 88%

## 📊 Example Use Cases

* Proactively retain high-risk customers
* Estimate customer lifetime value impact of churn
* Visualize which services impact churn the most

## 🤝 Contributing

Contributions are welcome!

1. Fork this repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## 📜 License

Distributed under the **MIT License**.
See [LICENSE](LICENSE) for details.

## 📬 Contact

* **Shubh Ruia:** [LinkedIn](https://www.linkedin.com/in/shubh-ruia/)
* **Project Link:** [GitHub](https://github.com/shubhruia/customer-churn-prediction)

---

Let me know if you want badges, a demo video/gif, or dataset references included.
