import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_pct=0.01, upper_pct=0.99, exclude_cols=None):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.exclude_cols = exclude_cols or []

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols_to_cap = [c for c in X_df.columns if c not in self.exclude_cols]
        self.cols_to_cap_ = cols_to_cap
        self.lower_bounds_ = X_df[cols_to_cap].quantile(self.lower_pct)
        self.upper_bounds_ = X_df[cols_to_cap].quantile(self.upper_pct)
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        outlier_mask = (
            X_df[self.cols_to_cap_].lt(self.lower_bounds_, axis=1) |
            X_df[self.cols_to_cap_].gt(self.upper_bounds_, axis=1)
        )
        num_outlier_values = outlier_mask.sum().sum()
        total_values = X_df[self.cols_to_cap_].shape[0] * X_df[self.cols_to_cap_].shape[1]
        pct_outlier_values = (num_outlier_values / total_values) * 100
        rows_with_outliers = outlier_mask.any(axis=1).sum()
        pct_rows = (rows_with_outliers / len(X_df)) * 100

        print("=" * 40)
        print("Outlier Report")
        print("=" * 40)
        print(f"Outlier values           : {num_outlier_values}")
        print(f"Percentage of values     : {pct_outlier_values:.2f}%")
        print(f"Rows with outliers       : {rows_with_outliers}")
        print(f"Percentage of rows       : {pct_rows:.2f}%")
        print("=" * 40)

        X_df[self.cols_to_cap_] = X_df[self.cols_to_cap_].clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1
        )
        return X_df

import statsmodels.api as sm
"""Iteratively drops the column with the highest VIF (above threshold),
    learned from TRAINING data only. Applies the same kept-column list to
    any future data (test set, new predictions)."""
class VIFDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=7.0, columns=None):
        self.threshold = threshold
        self.columns = columns

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            cols = self.columns if self.columns is not None else [f"col_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)

        cols = list(X_df.columns)

        while len(cols) > 1:
            X_with_const = sm.add_constant(X_df[cols])  # add intercept term
            vif_values = [
                variance_inflation_factor(X_with_const.values, i)
                for i in range(1, X_with_const.shape[1])
            ]  # skip index 0 = constant
            max_vif = max(vif_values)
            if max_vif > self.threshold:
                drop_col = cols[vif_values.index(max_vif)]
                print(f"Dropping '{drop_col}' due to VIF = {max_vif:.2f}")
                cols.remove(drop_col)
            else:
                break

        self.cols_to_keep_ = cols
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            cols = self.columns if self.columns is not None else [f"col_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)

        return X_df[self.cols_to_keep_]


app = Flask(__name__)

# Load the pipeline ONCE at startup — not per request
pipeline = joblib.load('simple_uber_pipeline.joblib')

REQUIRED_FIELDS = ['distance', 'jfk_dist', 'lga_dist', 'ewr_dist', 'nyc_dist', 'year']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    errors = []
    values = {}

    for field in REQUIRED_FIELDS:
        raw_value = request.form.get(field, '').strip()
        if raw_value == '':
            errors.append(f"'{field}' is required.")
            continue
        try:
            values[field] = float(raw_value)
        except ValueError:
            errors.append(f"'{field}' must be a number.")

    # Basic sanity range checks — catch obviously invalid input
    if 'distance' in values and (values['distance'] < 0 or values['distance'] > 200):
        errors.append("Distance must be between 0 and 200 km.")
    if 'year' in values and (values['year'] < 2009 or values['year'] > 2025):
        errors.append("Year must be a realistic ride year.")

    if errors:
        return render_template('index.html', errors=errors)

    # Build a single-row DataFrame — same column names/order the pipeline expects
    input_df = pd.DataFrame([values])[REQUIRED_FIELDS]

    prediction = pipeline.predict(input_df)[0]

    return render_template('result.html', prediction=round(prediction, 2), inputs=values)

if __name__ == '__main__':
    app.run(debug=True)