# Medical Insurance Charges Prediction

A machine learning project that predicts individual insurance `charges` from demographic and lifestyle attributes, comparing **Linear Regression** with **Polynomial Regression** using scikit-learn.

> Problem Statement and Objectives

The notebook frames a **supervised regression** task: predict the continuous target `charges` from the other columns in the dataset.

Objectives (derived from the notebook's workflow):

* Load and inspect the dataset, check for missing values and duplicates, and clean the data.
* Encode categorical features and standardize the inputs.
* Train a Linear Regression baseline and a Polynomial Regression model.
* Compare the models on a held-out test set using MSE, RMSE, MAE, and R².

## Dataset

* **Source:** Kaggle dataset `mirichoi0218/insurance`, downloaded in the notebook with `kagglehub`.
* **Size:** 1,338 rows × 7 columns (1,337 rows after removing 1 duplicate).
* **Missing values:** none.

|Column|Type|Role|
|-|-|-|
|`age`|integer|feature|
|`sex`|categorical|feature|
|`bmi`|float|feature|
|`children`|integer|feature|
|`smoker`|categorical (yes/no)|feature|
|`region`|categorical|feature|
|`charges`|float|**target**|

Currency/units of `charges` and the data collection method are not specified in the notebook.

## Data Preprocessing and Exploratory Analysis

The exploratory analysis in the notebook is limited to basic inspection:

1. Previewed the data (`head`), checked shape and dtypes (`info`).
2. Checked missing values (`isna().sum()`): none found.
3. Checked duplicates (`duplicated().sum()`): 1 found and removed with `drop\_duplicates()`.
4. One-hot encoded `sex`, `smoker`, and `region` (`pd.get\_dummies`, `drop\_first=True`), giving 8 input features.
5. Split the data 80/20 with `train\_test\_split(random\_state=42)`: 1,069 training and 268 test samples.
6. Standardized features with `StandardScaler`, fitted on the training set only and applied to the test set.

No distribution plots, correlation analysis, or outlier analysis are included in the notebook.

## Models

|Model|Details|
|-|-|
|Linear Regression|`LinearRegression` on standardized features (baseline)|
|Polynomial Regression|`PolynomialFeatures` + `LinearRegression`; degrees 1–4 compared, **degree 2** selected for the final comparison|

## Evaluation and Results

Metrics are computed on the 268-sample test set.

|Model|MSE|RMSE|MAE|R²|
|-|-|-|-|-|
|Linear Regression|35,478,020.68|5,956.34|4,177.05|0.8069|
|Polynomial (degree 2)|21,585,843.72|4,646.06|2,867.32|0.8825|

Polynomial degree comparison:

|Degree|Features|Train R²|Test R²|Test RMSE|
|-|-|-|-|-|
|1|8|0.7299|0.8069|5,956.34|
|2|44|0.8340|0.8825|4,646.06|
|3|164|0.8471|0.8709|4,870.27|
|4|494|0.8666|0.8003|6,057.86|

The notebook also plots actual vs. predicted charges for both models.

## Main Findings

* Polynomial Regression (degree 2) outperformed the linear baseline on every reported metric (test R² 0.8825 vs. 0.8069; RMSE 4,646 vs. 5,956).
* Test R² peaks at degree 2 and then declines, while training R² keeps rising. This indicates overfitting at higher degrees.
* In the linear model (standardized features), `smoker\_yes` has the largest coefficient (≈ 9,234), followed by `age` (≈ 3,473), `bmi` (≈ 1,928), and `children` (≈ 637). Coefficients for `sex\_male` and the region dummies are comparatively small.

## Limitations

* Only one train/test split (`random\_state=42`) is used; no cross-validation.
* The polynomial degree was chosen by looking at test-set results, so the reported test score may be slightly optimistic. A separate validation set or cross-validation would be more rigorous.
* Only two model types are compared; no regularization, tree-based, or other models are tested, and no hyperparameter tuning is performed.
* Limited EDA (no visualizations of feature distributions or relationships).
* Residual analysis and model assumptions are not examined.

## Libraries and Tools

* Python 3 (exact version not specified)
* NumPy, pandas, Matplotlib
* scikit-learn (`LinearRegression`, `PolynomialFeatures`, `StandardScaler`, `train\_test\_split`, metrics)
* kagglehub (dataset download)
* Jupyter Notebook / Google Colab (the notebook output indicates it was run on Colab)

Library versions are not specified in the notebook.

## Installation and Usage

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd <your-repository-folder>

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install numpy pandas matplotlib scikit-learn kagglehub notebook

# 4. Launch the notebook
jupyter notebook ML\_section\_task.ipynb
```

Run all cells in order. The dataset is downloaded automatically by `kagglehub` (internet access required). Alternatively, upload the notebook to Google Colab and run it there.

## Project Structure

```
├── ML\_section\_task.ipynb   # Analysis and models
└── README.md
```

