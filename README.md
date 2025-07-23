# 🏈 CFB SP+ Power Ratings & Spread Predictor

This web app uses machine learning to project **2025 SP+ power ratings** for all 133 FBS college football teams, based on previous years' advanced stats. It also allows you to compare two teams and generate a **projected point spread**, including **home field advantage** and a **95% confidence interval**.

Built with **Streamlit** and powered by a custom **ridge regression model** trained on 2023–2024 season data.

---

## 🔮 Features

- Predicts **2025 SP+ ratings** for all FBS teams  
- Allows you to compare two teams head-to-head  
- Adjusts for **home field advantage (+2.5 points)**  
- Displays **projected spread** and **95% confidence interval**  
- Clean, fast, and mobile-friendly interface  

---

## 📊 How It Works

1. The model is trained on:
   - 2024 SP+ rating
   - 2024 returning production %
   - 2024 returning roster size
   - 2024 total production rating
   - 2024 strength of schedule

2. It uses **Ridge Regression** from `scikit-learn` to reduce overfitting.

3. Predictions are **rescaled** between -10 and +30 to match typical SP+ ranges.

---

## 📁 Files Included

- `app.py` – Streamlit app  
- `CFB.csv` – 2023 stats and 2024 actual SP+ (used for training)  
- `CFB25.csv` – 2024 stats used to predict 2025 SP+  
- `requirements.txt` – Python dependencies (pandas, sklearn, streamlit)

---

## 🚀 Try It Live

Once deployed to [Streamlit Cloud](https://streamlit.io/cloud), your friends can access the app through a shareable link like:

