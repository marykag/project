# Food Hazard Detection Challenge

The **Food Hazard Detection Challenge** involves predicting two key categories, **hazard-category** and **product-category**, based on the provided dataset. The objective is to evaluate and compare machine learning models using both short-text (title) and long-text (text) features for accurate classification. Below is a description of the dataset and analysis tasks.

---

## Dataset Overview

This project performs multi-label classification on text data using machine learning algorithms. It includes:
- **Data preprocessing**: Loading and cleaning text data, feature extraction using TF-IDF.
- **Model Building**: Training multiple classification algorithms (e.g., Logistic Regression, Decision Trees, SVM).
- **Hyperparameter Tuning**: Using GridSearchCV/RandomizedSearchCV to optimize hyperparameters.
- **Model Evaluation**: Evaluating model performance using metrics like precision, recall, F1-score, and AUC-ROC.
- **Model Comparison**: Comparing the results of different models.
- **Visualization**: Plotting evaluation metrics for easy comparison.


The dataset includes the following columns:

| **Column Name**       | **Description**                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------------|
| `year`                | The year of the recall notification.                                                                     |
| `month`               | The month of the recall notification.                                                                    |
| `day`                 | The day of the recall notification.                                                                      |
| `country`             | The country issuing the recall.                                                                          |
| `title`               | A short description or title of the recall notification.                                                 |
| `text`                | Detailed information about the recall, including product name, problem description, and quantities.      |
| `hazard-category`     | The general category of the hazard (e.g., *biological*, *chemical*).                                      |
| `product-category`    | The general category of the product (e.g., *meat, egg and dairy products*).                               |
| `hazard`              | The specific hazard associated with the product (e.g., *listeria monocytogenes*).                        |
| `product`             | The specific product involved in the recall (e.g., *smoked sausage*).                                    |

---

## Task Description

### Objective:
The goal is to develop machine learning models that predict the **product-category** and **hazard-category** and the **producy** and **hazard** using:
1. **Short Texts (Title)**: Analyze the predictive power of concise information.
2. **Long Texts (Text)**: Leverage detailed descriptions for enhanced accuracy.

### Steps to Perform:
1. **Data Preprocessing**:
   - Tokenize and preprocess the `title` and `text` columns.
   - Perform stopword removal, lemmatization, or stemming.
   - Use techniques like **TF-IDF** to convert text data into numerical representations.

2. **Model Training**:
   - Train separate models for the `title` and `text` data using various machine learning algorithms.
   - Explore both basic (e.g., Logistic Regression, Naive Bayes) and advanced (e.g., Random Forest, Gradient Boosting) models.

3. **Evaluation**:
   - Measure performance using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
   - Compare results to determine which approach (short-text vs. long-text) provides better predictions.

4. **Submission**:
   - Submit predictions to the leaderboard for evaluation.
   - Document the results, including the best-performing model, and report the score and rank.

---

## Exploratory Data Analysis

### Sample Data:
Below is a preview of the dataset:

| Year | Month | Day | Country | Title                          | Text                                                                                                                        | Hazard-Category | Product-Category     | Hazard                 | Product        | 
|------|-------|-----|---------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-----------------|----------------------|------------------------|----------------|
| 1994 | 1     | 7   | US      | Recall Notification: FSIS-024-94 | Case Number: 024-94; Date Opened: 07/01/1994; Date Closed: 09/22/1994; Recall Class: 1; Press Release: Y; Product: SMOKED CHICKEN SAUSAGE; Problem: BACTERIA; Description: LISTERIA; Total Pounds Recalled: 2,894; Pounds Recovered: 2,894 | Biological     | Meat, egg and dairy  | Listeria monocytogenes | Smoked Sausage |
| 1994 | 3     | 10  | US      | Recall Notification: FSIS-033-94 | Case Number: 033-94; Date Opened: 10/03/1994; Date Closed: 01/19/1995; Recall Class: 1; Press Release: Y; Product: WIENERS; Problem: BACTERIA; Description: LISTERIA; Total Pounds Recalled: 5,500; Pounds Recovered: 4,568              | Biological     | Meat, egg and dairy  | Listeria spp           | Sausage        |

### Key Insights:
- The dataset contains structured fields for date and location, and unstructured text fields (`title`, `text`).
- Labels (`hazard-category`, `product-category`, `hazard`, `product`) are categorical, ideal for classification tasks.
- The `text` column provides significantly more details compared to the `title`, which may result in better predictive performance.

---

## Benchmarking Approaches

1. **Short Text Analysis (Title)**:
   - Focus on extracting features from the `title` column.
   - Use simple & advanced models to test the power of concise recall descriptions.

2. **Long Text Analysis (Text)**:
   - Utilize the detailed information in the `text` column.
   - Apply simple & advanced feature extraction and machine learning techniques to maximize prediction accuracy.


**This approach ensures a comprehensive evaluation of the dataset and provides insights into the predictive power of short vs. long textual descriptions for food hazard detection.**
