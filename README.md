# telco-churn-prediction
 A machine learning project to predict customer churn using Python, Scikit-learn, and XGBoost.

# Telco Customer Churn Prediction

An end-to-end machine learning project to predict customer churn and identify key drivers to inform retention strategies.

![Project Banner](https://i.imgur.com/2s9V5cO.png)

## 1. Business Problem

Customer churn is a critical challenge for subscription-based businesses like telecommunications companies. The loss of customers directly impacts revenue and market share. This project aims to solve this problem by building a predictive model that can identify customers who are at a high risk of churning.

The key objectives are:
* To build a reliable machine learning model that accurately predicts customer churn.
* To identify the most significant factors and customer behaviors that contribute to churn.
* To provide actionable, data-driven insights that the business can use to develop effective customer retention campaigns.

---

## 2. Data

The analysis is based on a publicly available **Telco Customer Churn** dataset from Kaggle. This dataset contains demographic information, account details, and services subscribed to by over 7,000 customers.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Features:** Include customer gender, tenure, contract type, payment method, monthly charges, and more.
* **Target Variable:** `Churn` (Yes / No).

---

## 3. Methodology

The project follows a standard data science workflow:

1.  **Data Cleaning & Preprocessing:** Handled missing values (e.g., converting empty 'TotalCharges' to numeric) and removed irrelevant columns like `customerID`.

2.  **Exploratory Data Analysis (EDA):** Created visualizations to understand the relationships between different features and customer churn. This step revealed initial insights into customer behavior.

3.  **Feature Engineering:** Converted all categorical features (e.g., 'Contract', 'PaymentMethod') into a numerical format using one-hot encoding. Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to ensure models performed optimally.

4.  **Model Training & Comparison:** Trained two different classification models:
    * A baseline **Logistic Regression** model.
    * A more advanced **XGBoost Classifier**.

5.  **Evaluation & Interpretation:** Evaluated the models based on key classification metrics (Accuracy, Precision, Recall, F1-score) and a confusion matrix. The feature importances of the best-performing model were analyzed to determine the key drivers of churn.

---

## 4. Key Findings & Insights

The analysis and the final model revealed several crucial insights:

* ### Insight 1: Contract Length is the Biggest Loyalty Factor
    Customers on **month-to-month contracts** are significantly more likely to churn. Conversely, those on **one or two-year contracts** are far more loyal.
    

* ### Insight 2: High-Risk Services & Payment Methods
    Subscribers with **Fiber Optic internet service** and those who pay by **Electronic Check** showed the highest churn rates. This may point to issues with service reliability, pricing, or the payment experience.

* ### Insight 3: Tenure Matters
    Customer **tenure** is a strong predictor of churn. Newer customers are at a much higher risk of leaving, while long-term customers are more likely to stay.



---

## 5. Model Performance

The **XGBoost Classifier** was selected as the final model due to its superior performance over the baseline.

* **Accuracy:** ~79% on the unseen test set.
* **Key Insight:** The model demonstrated a strong ability to correctly identify customers who would *not* churn (high precision for the 'No' class) and a reasonable ability to catch customers who *would* churn.

---

## 6. Technologies Used

* **Python 3**
* **Pandas:** For data manipulation and cleaning.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For data preprocessing (`StandardScaler`), model training (`LogisticRegression`), and evaluation (`classification_report`, `confusion_matrix`).
* **XGBoost:** For training the high-performance gradient boosting model.
* **Matplotlib & Seaborn:** For data visualization.
* **Jupyter Notebook / VS Code:** For the development environment.
