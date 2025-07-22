# Churn Prediction & Segmentation For Retention Strategy For Ecommerce

<img width="1024" height="366" alt="image" src="https://github.com/user-attachments/assets/94ea100c-aab3-432f-b117-174fbd0ca8e0" />

## 📑 Table of Contents
1. 🌱 [Background & Overview](#background-&-overview)
2. 🔍 [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. 📊 [Train & Apply Churn Prediction Model](#train-apply-churn-prediction-model)
4. 💡 [Key Findings and Recommendations for Retention](#key-findings-and-recommendations-for-retention)
5. 🤖 [Create A Model For Predicting Churn](#create-a-model-for-predicting-churn)
6. 🧑‍💻 [Customer Segmentation Using Clustering](#customer-segmentation-using-clustering)

## 📌 Background & Overview

### **🎯 Objective**

This project aims to predict and segment churned users for an e-commerce business in order to create effective retention strategies. By applying Machine Learning and Python, this project seeks to:

✔️ Identify key behaviors and patterns among churned customers.

✔️ Build a predictive model to anticipate customer churn.

✔️ Segment churned users to tailor retention offers and special promotions.


**❓ What Business Question Will It Solve?**

✔️ Which factors drive customer churn in e-commerce?

✔️ How can churned users be predicted in advance and addressed proactively?

✔️ How can we develop an accurate churn prediction model?

✔️ How can churned users be segmented to enable targeted promotional strategies?


**👤 Who Is This Project For?**

✔️ Data Analysts & Business Analysts — To uncover insights on churn behavior and inform retention planning.

✔️ Marketing & Customer Retention Teams — To design and launch data-driven retention campaigns.

✔️ Decision-makers & Stakeholders — To help reduce churn rates and increase customer lifetime value.

## 📂 **Dataset Description & Data Structure**

### 📌 **Data Source**  
**Source:** The dataset is obtained from the e-commerce company's database.  
**Size:** The dataset contains 5,630 rows and 20 columns.  
**Format:** .xlxs file format.

### 📊 **Data Structure & Relationships**

1️⃣ **Tables Used:**  
The dataset contains only **1 table** with customer and transaction-related data.

2️⃣ **Table Schema & Data Snapshot**  
**Table: Customer Churn Data**

<details>
  <summary>Click to expand the table schema</summary>

| **Column Name**              | **Data Type** | **Description**                                              |
|------------------------------|---------------|--------------------------------------------------------------|
| CustomerID                   | INT           | Unique identifier for each customer                          |
| Churn                        | INT           | Churn flag (1 if customer churned, 0 if active)              |
| Tenure                       | FLOAT         | Duration of customer's relationship with the company (months)|
| PreferredLoginDevice         | OBJECT        | Device used for login (e.g., Mobile, Desktop)                 |
| CityTier                     | INT           | City tier (1: Tier 1, 2: Tier 2, 3: Tier 3)                   |
| WarehouseToHome              | FLOAT         | Distance between warehouse and customer's home (km)         |
| PreferredPaymentMode         | OBJECT        | Payment method preferred by customer (e.g., Credit Card)     |
| Gender                       | OBJECT        | Gender of the customer (e.g., Male, Female)                  |
| HourSpendOnApp               | FLOAT         | Hours spent on app or website in the past month              |
| NumberOfDeviceRegistered     | INT           | Number of devices registered under the customer's account   |
| PreferedOrderCat             | OBJECT        | Preferred order category for the customer (e.g., Electronics)|
| SatisfactionScore            | INT           | Satisfaction rating given by the customer                    |
| MaritalStatus                | OBJECT        | Marital status of the customer (e.g., Single, Married)       |
| NumberOfAddress              | INT           | Number of addresses registered by the customer               |
| Complain                     | INT           | Indicator if the customer made a complaint (1 = Yes)         |
| OrderAmountHikeFromLastYear  | FLOAT         | Percentage increase in order amount compared to last year   |
| CouponUsed                   | FLOAT         | Number of coupons used by the customer last month            |
| OrderCount                   | FLOAT         | Number of orders placed by the customer last month           |
| DaySinceLastOrder            | FLOAT         | Days since the last order was placed by the customer        |
| CashbackAmount               | FLOAT         | Average cashback received by the customer in the past month  |

</details>

## ⚒️ Main Process
### **Data Preprocessing**

📌 Import Necessary Libraries

```ruby
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
```

📌 Load the data

```ruby
import os
os.chdir("/content/drive/MyDrive/DAC 1 on 1 /ML/K35_Pham Thanh Tung_Project 3")
df = pd.read_excel("churn_prediction.xlsx")
df
```

📌 Before diving into analysis, let's take a quick look at the first few rows of the dataset to examine its structure and key features

<img width="1847" height="672" alt="image" src="https://github.com/user-attachments/assets/82640dfc-91c8-4fb7-ab3b-cf9ad79dbe8d" />

**💡 Data Understanding**

📌 Before performing any analysis or modeling, I carried out several steps to preprocess the data:

**📝 Check Dataset Structure**

<img width="675" height="657" alt="image" src="https://github.com/user-attachments/assets/d88b5d57-3e31-4d5a-941d-82e26ca83c35" />

After checking the general structure of the dataset, this gave me an overview of the number of rows, columns, and data types for each feature, along with summary statistics. The dataset contains 5,630 rows and 20 columns, with a mix of numeric and categorical variables.

**📝 Check for Missing Values** 

<img width="1848" height="619" alt="image" src="https://github.com/user-attachments/assets/580b3f06-ae16-47fd-8f67-844bab770458" />

Missing values were detected in multiple columns. The columns with missing values are:

   - `Tenure` - 264 missing values
   - `WarehouseToHome` - 251 missing values
   - `HourSpendOnApp` - 255 missing values
   - `OrderAmountHikeFromlastYear` - 265 missing values
   - `CouponUsed` - 256 missing values
   - `OrderCount` - 258 missing values
   - `DaySinceLastOrder` - 307 missing values

**📝 Missing Value Handling**  

The dataset contained missing values in several columns. The missing values were **handled by replacing them with the median**, which prepared the data for further analysis and modeling.

```ruby
# Define the list of columns with missing values
cols_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

# Replace missing columns with median
for col in cols_missing:
    # Fill missing values in each column with the median of that column
    df[col].fillna(value= df[col].median(), inplace=True)
```

<img width="866" height="596" alt="image" src="https://github.com/user-attachments/assets/66e6ddfb-5225-431d-8373-483c8afe1770" />

**📝 Check for Duplicates**  

Aftering checkeing for duplicate rows in the dataset and found that there were no duplicate entries.

<img width="561" height="176" alt="image" src="https://github.com/user-attachments/assets/17a77e1a-23a6-4d14-b8f9-09b64fe28f6d" />

**📝 Merging Columns with Similar Data but Different Names**  

The dataset contains words with the **same meaning** but written differently. These should be **standardized into a single form**.

<img width="1145" height="653" alt="image" src="https://github.com/user-attachments/assets/745e1e5e-c840-422a-bc55-fbf398bed6fd" />

### **Q1. What are the patterns/behavior of churned users? What are your suggestions to the company to reduce churned users.**

**📝 Outlier Detection**

<img width="854" height="714" alt="image" src="https://github.com/user-attachments/assets/9f77718f-bfd8-4c26-843d-ff8ea1afc787" />

- Categorical variables like `PreferredLoginDevice`, `Gender`, `MaritalStatus`, and others were analyzed through count plots, which gave insights into the distribution of each category.
  
- Continuous variables like `Tenure`, `SatisfactionScore`, `CashbackAmount`, and others were examined using boxplots. It was observed that the outliers in these columns. However, **outliers** reasonable and **should be kept** because they represent **distinguishing characteristics for predicting churn**.

**📝 Encoding**

After preprocessing the dataset, encoding was applied to the categorical features.

<img width="1853" height="649" alt="image" src="https://github.com/user-attachments/assets/7131fcd1-8b3a-446b-a9ab-c1ac9449012a" />

The `CustomerID` column was dropped since it is a unique identifier and does not contribute to the prediction model.













