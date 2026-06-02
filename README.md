# HDB Resale Price Prediction for Cataria Property Solutions

## Project Description

Cataria Property Solutions aims to improve the accuracy and consistency of HDB resale property valuations through machine learning. Traditional valuation methods often rely heavily on manual assessments and agent experience, which can lead to inconsistencies and inefficiencies.

This project develops a regression-based machine learning pipeline that predicts HDB resale prices using historical transaction data. The solution provides data-driven property valuations to support property consultants and improve decision-making for clients.

---

# Prerequisites & Installation

## Requirements

* Python 3.11
* Conda (Recommended)
* Git

## Required Libraries

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
pyyaml
joblib
```

## Installation

```bash
git clone <repository_url>
cd hdb-resale-price-prediction

conda create -n aiap_hdb python=3.11
conda activate aiap_hdb

pip install -r requirements.txt
```

---

# Pipeline Execution

## Run the Pipeline

```bash
python main.py
```

## Modify Configuration

Project parameters can be configured in:

```bash
./src/config.yaml
```

Examples of configurable settings include:

* Train-test split ratio
* Random seed
* Hyperparameter tuning settings
* Feature selection options
* Model configuration

After updating the configuration file, rerun `main.py` to execute the pipeline with the new settings.

---

# Pipeline Flow

## Step 1: Data Loading

Load raw HDB resale transaction data into the pipeline.

## Step 2: Data Cleaning

Perform data quality checks and preprocessing:

* Remove duplicate records
* Handle missing values
* Correct data types
* Remove irrelevant features
* Remove post-hoc features to prevent data leakage

## Step 3: Exploratory Data Analysis (EDA)

Analyse the dataset to identify:

* Feature distributions
* Outliers
* Correlations
* Potential data quality issues

## Step 4: Feature Engineering & Preprocessing

Prepare data for modelling through:

* Feature engineering
* Encoding categorical variables
* Feature scaling
* Train-test splitting

## Step 5: Model Training

Train and compare:

* Linear Regression
* Ridge Regression
* Lasso Regression

## Step 6: Hyperparameter Tuning

Optimise model performance using:

* Grid Search CV
* Randomized Search CV

## Step 7: Model Evaluation

Evaluate model performance using regression metrics.

## Step 8: Model Saving

Save the final selected model for deployment and future use.

---

# Key EDA Findings

Several important patterns were identified during exploratory data analysis.

Floor area was found to be one of the strongest predictors of resale price, with larger flats generally achieving higher transaction values. Location-related features such as town also showed significant influence, reflecting differences in demand across various regions in Singapore. Remaining lease demonstrated a positive relationship with resale price, where flats with longer lease durations tended to command higher prices.

In addition, several numerical features contained outliers and skewed distributions, requiring preprocessing and scaling before model training. These findings directly informed the feature engineering and model development process.

---

# Feature Handling

| Feature                | Treatment           | Reason                                                 |
| ---------------------- | ------------------- | ------------------------------------------------------ |
| town                   | One-Hot Encoding    | Convert location categories into numerical features    |
| flat_type              | Ordinal Encoding    | Preserve the natural ordering of flat types            |
| storey_range           | Ordinal Encoding    | Preserve the ordering of floor levels                  |
| floor_area_sqm         | Feature Scaling     | Standardise numerical values                           |
| remaining_lease        | Feature Engineering | Extract lease information for modelling                |
| remaining_lease_months | Feature Engineering | Convert lease duration into a numerical format         |
| year                   | Feature Engineering | Capture temporal effects in resale prices              |
| month                  | Feature Engineering | Capture seasonal transaction patterns                  |
| Missing Values         | Imputation          | Handle incomplete records while retaining observations |
| Duplicate Records      | Removed             | Improve data quality and prevent bias                  |
| Irrelevant Features    | Removed             | Reduce noise and improve model performance             |
| Post-hoc Features      | Removed             | Prevent data leakage                                   |

---

# Model Choices

## Linear Regression

Linear Regression was selected as the baseline model because it is simple, interpretable, and provides a benchmark for comparison.

## Ridge Regression

Ridge Regression was selected to reduce overfitting through L2 regularisation. It shrinks coefficient values while retaining all features, improving model generalisation.

## Lasso Regression

Lasso Regression was selected because it combines regularisation with feature selection by shrinking some coefficients to zero, reducing model complexity.

---

# Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

## Final Selected Model: Ridge Regression

| Metric   | Result  |
| -------- | ------- |
| R² Score | 0.865   |
| MAE      | $41,352 |

The tuned Ridge Regression model achieved the best overall performance. It explains approximately 86.5% of the variation in HDB resale prices while maintaining an average prediction error of approximately $41,352. These results demonstrate strong predictive performance and good generalisation to unseen data.

---

# Deployment Considerations

## Scalability

The Ridge Regression model is computationally efficient and can scale effectively to handle large volumes of valuation requests.

## Real-Time Performance

Prediction latency is low, making the model suitable for near real-time property valuation applications.

## Integration

The model can be integrated into Cataria Property Solutions' existing systems through APIs, dashboards, or web applications.

## Monitoring and Maintenance

Property market conditions may change over time, leading to data drift and reduced model performance. Continuous monitoring should be implemented to track performance and determine when retraining is required.

## Business Impact

This solution provides Cataria Property Solutions with a consistent, data-driven approach to HDB resale price estimation, helping improve valuation accuracy and support better business decisions.
