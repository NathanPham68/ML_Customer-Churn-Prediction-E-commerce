# Churn Prediction & Segmentation For Retention Strategy For Ecommerce

<img width="1024" height="366" alt="image" src="https://github.com/user-attachments/assets/94ea100c-aab3-432f-b117-174fbd0ca8e0" />

## 📑 Table of Contents
1. 🌱 [Background & Overview](#background-&-overview)
2. 🔍 [Dataset Description & Data Structure](#dataset-description-&-data-structure)
3. 📊 [Main Process](#main-process)
4. 💡 [Q1. What are the patterns/behavior of churned users?](#q1.-what-are-the-patterns/behavior-of-churned-users?)
5. 🤖 [Q2. Build the Machine Learning model for predicting churned users](#q2.-build-the-machine-learning-model-for-predicting-churned-users)
6. 🧑‍💻 [Q3. Based on the behaviors of churned users, the company would like to offer some special promotions for them](#q3.-based-on-the-behaviors-of-churned-users,-the-company-would-like-to-offer-some-special-promotions-for-them)

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

<img width="1017" height="603" alt="image" src="https://github.com/user-attachments/assets/b335ca3d-a383-4503-ad24-97cfe04afe6e" />

Show Feature Importance

```ruby
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x.columns, clf_ranf.feature_importances_):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True)

importances = importances.reset_index()

# Create bar chart
plt.figure(figsize=(8, 8))
plt.barh(importances.tail(20)['index'][:20], importances.tail(20)['Gini-importance'])

plt.title('Feature Important')

# Show plot
plt.show()
```

<img width="1136" height="707" alt="image" src="https://github.com/user-attachments/assets/0ef3e0a9-76fc-4757-bf28-cc6e91ff6b3d" />

As Feature Importance show, we can see these features can have high relation with target columns:
* Tenure
* Cashback amount
* Distance from warehouse to home
* Complain
* Days since Last order

&rarr; We will analyse and visualize these features for more insights.

**📝 Analyse features from initial Random Forest model**

* Tenure
* Cashback amount
* Distance from warehouse to home
* Complain
* Days since Last order

```ruby
def count_percentage(df, column, target, count):
    '''
    This function to create the table calculate the percentage of fraud/non-fraud
    transaction on total transaction group by category values
    '''

    # Create 2 dataframes of fraud and non-fraud
    fraud = df[df[target]==1].groupby(column)[[count]].count().reset_index().sort_values(ascending=False, by = count)
    not_fraud = df[df[target]==0].groupby(column)[[count]].count().reset_index().sort_values(ascending=False, by = count)

    #Merge 2 dataframe into one:
    cate_df = fraud.merge(not_fraud, on = column , how = 'outer')
    cate_df = cate_df.fillna(0)
    cate_df.rename(columns = {count+'_x':'fraud',count+'_y':'not_fraud'}, inplace = True)

    #Caculate the percentage:
    cate_df['%'] = cate_df['fraud']/(cate_df['fraud']+cate_df['not_fraud'])
    cate_df = cate_df.sort_values(by='%', ascending=False)

    return cate_df
```

#### **1. Tenure**  New users are churned more than old users (tenure = 0 or 1)

<img width="1189" height="571" alt="image" src="https://github.com/user-attachments/assets/a1096eea-08da-4772-993f-2a7a27b8936d" />

<img width="1216" height="570" alt="image" src="https://github.com/user-attachments/assets/ba1add97-0b17-4313-8f79-b61463ba0ff0" />

#### **2. Warehouse to home**  Not significantly related

<img width="622" height="619" alt="image" src="https://github.com/user-attachments/assets/2d0aac9b-c891-4e13-afd4-532c1e3ed541" />

<img width="737" height="511" alt="image" src="https://github.com/user-attachments/assets/989691ce-2802-4273-84b2-b49fed960c6f" />

For both churn & not churn:
* The median, pt25, mean, pt75 is quite the same --> The centralize of data is the same
* For not churn, data has some outliers --> This can be not significant enough to consider it as an insight for not churn

&rarr; There're no strong evidences show that there different between churn and not churn for warehousetohome --> We will exclude this features when apply model for not being bias.

#### **3. Days since last order:** churn users with complain = 1 have higher days since orders than churned users with complain = 0

<img width="789" height="618" alt="image" src="https://github.com/user-attachments/assets/3e6f008c-37fa-42d7-8c42-3402198869de" />

From this chart, we see for churned users, they had orders recently (the day since last order less than not churned users) --> This quite strange, we should monitor more features for this insight (satisfaction_score, complain,..)

<img width="768" height="655" alt="image" src="https://github.com/user-attachments/assets/49c9b34d-2f57-49b0-923b-0c289a731f99" />

For churned users with complain = 1, they had daysincelastorder higher than churn users with compain = 0

#### **4. Cashback amount**  Churn users recevied cashback amount less than not churn users.

<img width="758" height="619" alt="image" src="https://github.com/user-attachments/assets/22fd52cd-7600-4e29-8ef9-42096dc2f956" />

Churn users recevied cashback amount less than not churn users.

#### **5. Complain** The number of users complain on churn is higher than not churn

<img width="803" height="595" alt="image" src="https://github.com/user-attachments/assets/1c058cd5-9c21-4474-82d9-c18486965553" />

#### **6. Conclusion & Suggestion**
1. Churned users usually are new users &rarr; Provide more promotion for new users, or increase the new users experience
2. Churned users usually receive less cashback than not churn &rarr; Increase the cashback ratio
3. Churned users complain more &rarr; deep dive what these churned users complain about, and provide the solution

### **Q2. Build the Machine Learning model for predicting churned users. (fine tuning)**

<img width="811" height="665" alt="image" src="https://github.com/user-attachments/assets/6ea62843-65b5-48bf-960d-b0219d61da33" />

<img width="763" height="608" alt="image" src="https://github.com/user-attachments/assets/d9572dc5-4276-4c29-91e6-7478b4ad1b02" />

**📝 Fine-tune the BEST model**

```ruby
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# 1. Khởi tạo mô hình cơ bản
rf_base = RandomForestClassifier(random_state=42)

# 2 Tạo grid tham số để RandomizedSearch
param_grid_rf = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# 3 RandomizedSearchCV với scoring là Recall
rf_finetune = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid_rf,
    n_iter=30,  # số vòng thử nghiệm
    cv=5,
    scoring='recall',
    verbose=1,
    random_state=42,
    n_jobs=-1  # tận dụng nhiều CPU core
)

# 4 Fit trên train set
rf_finetune.fit(X_train_scaled, y_train)

# 5 Kết quả tốt nhất
print("Best Params RF:", rf_finetune.best_params_)
print("Best CV Recall:", rf_finetune.best_score_)
```

**📝 Best RF score on test set**

<img width="684" height="661" alt="image" src="https://github.com/user-attachments/assets/8cac0355-e4fb-48ee-9a63-f12df75b2623" />

**📝 Check feature importance**

```ruby
# Tầm quan trọng của feature
importances = pd.Series(
    best_rf.feature_importances_,
    index=top_features
).sort_values(ascending=False)

print(importances)

# Vẽ biểu đồ
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance - Best RF Model")
plt.show()
```

<img width="847" height="469" alt="image" src="https://github.com/user-attachments/assets/ca119b5c-2993-4302-980d-1c363544d7cb" />

### **Q3. Based on the behaviors of churned users, the company would like to offer some special promotions for them. Please segment these churned users into groups. What are the differences between groups?**

<img width="746" height="552" alt="image" src="https://github.com/user-attachments/assets/1c2d9adf-8b41-4074-ae4d-0a074bd0738d" />

**📝 Apply PCA for dimensionality reduction & visualization**

<img width="763" height="343" alt="image" src="https://github.com/user-attachments/assets/7d6a9486-90f6-47d3-86e0-d127570a1597" />

**📝 Determine the appropriate number of Clusters: Elbow + Silhouette**

```ruby
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method
wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(pca_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, wcss, 'bo--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

<img width="694" height="401" alt="image" src="https://github.com/user-attachments/assets/85e66394-5c54-44bb-ad50-33a398277939" />

<img width="763" height="243" alt="image" src="https://github.com/user-attachments/assets/d2c030d4-e491-40aa-9669-4a2113e185a0" />

- PCA can not keep the significant meaning of the data (the sum of explained_variance_ratio is too small)
- When applying Elbow method, we see there're no clear elbow points.
- Our hypothesis is that the data is sporadic, which means there're no clearly common patterns between data, and we can not cluster them into groups.

**Our suggestions for next steps:**

* We can gather more data on churned users — either by collecting actual churn data or by using the supervised model above to predict churn and treat those predictions as ground truth for the clustering model.

* The business can roll out promotions to all churned users and track the results. These results can then be added as new features in the next model.

