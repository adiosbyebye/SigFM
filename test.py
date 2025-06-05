import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from itertools import combinations
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sdgym.datasets import get_available_datasets
from sdv.datasets.demo import download_demo
from collections import Counter
import pickle
import gzip

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Let Pandas use full terminal width
pd.set_option('display.expand_frame_repr', False)  # Do not wrap to new lines
# file_path = "./results/adult/tabdiff/results_detailed_1/Custom_tabdiff_Synthesizer_adult.data.gz"
# file_path = "./results/results_detailed_1/Custom:DSF_GAUSSIAN_COPULA_adult.data.gz"
# file_path = "./results/results_detailed_1/Custom:Flow Matching Copula Synthesizer_adult.data.gz"
# with gzip.open(file_path, "rb") as f:
#     df = pickle.load(f)
# print(df)
# print(df['capital-gain'].value_counts())
# print(df['age'].value_counts())
# print(set(df['education']))
# print(set(df['native-country']))


#
# data, metadata = download_demo(
#     modality='single_table',
#     dataset_name='child'
# )
#
# data1, metadata1 = download_demo(
#     modality='single_table',
#     dataset_name='adult'
# )
#
# ds = get_available_datasets()
#
#
# print(data1['capital-gain'].value_counts())
# print(data1.shape)
# print(metadata1)
#print(ds['dataset_name'])


# # Initialize new column
# num_columns = []
#
# # Loop through each dataset and fetch number of columns
# for name in ds['dataset_name']:
#     try:
#         data1, metadata1 = download_demo(modality='single_table', dataset_name=name)
#         num_columns.append(data1.shape[1])
#     except Exception as e:
#         print(f"Failed to load dataset: {name} due to {e}")
#         num_columns.append(None)
#
# # Add new column to the DataFrame
# ds['columns'] = num_columns
#
# # Print result
# print(ds)

#============================================= Evaluation Code =========================================

def compute_shape_and_trend(
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        categorical_columns: list[str],
) -> tuple[float, float]:
    """
    Return the single aggregated Shape and Trend scores
    defined in TabDiff (ICLR 2025).

    Shape  =  mean over *all* columns
              ├─ KST  for numerical columns
              └─ TVD  for categorical columns
    Trend  =  mean over *same-type* column pairs
              ├─ 0.5·|ρ_real−ρ_syn|        for numerical–numerical
              └─ 0.5·Σ|P_real−P_syn|       for categorical–categorical

    Both metrics lie in [0,1] (lower is better).
    """
    # split columns -----------------------------------------------------------
    cat_cols = set(categorical_columns)
    num_cols = [c for c in real_df.columns if c not in cat_cols]

    # -----------------------  SHAPE  ----------------------------------------
    per_col_scores = []

    # numerical → Kolmogorov–Smirnov Test (Eq. 17)  :contentReference[oaicite:0]{index=0}
    for col in num_cols:
        stat, _ = ks_2samp(real_df[col], syn_df[col])
        per_col_scores.append(stat)

    # categorical → Total Variation Distance (Eq. 19)  :contentReference[oaicite:1]{index=1}
    for col in cat_cols:
        r_freq = real_df[col].value_counts(normalize=True)
        s_freq = syn_df[col].value_counts(normalize=True)
        cats = r_freq.index.union(s_freq.index)
        tvd = 0.5 * sum(abs(r_freq.get(c, 0) - s_freq.get(c, 0)) for c in cats)
        per_col_scores.append(tvd)

    shape_score = float(np.mean(per_col_scores))

    # -----------------------  TREND  ----------------------------------------
    per_pair_scores = []

    for col1, col2 in combinations(real_df.columns, 2):
        # numerical–numerical → ½|ρ_r − ρ_s|   (Eq. 21)  :contentReference[oaicite:2]{index=2}
        if col1 in num_cols and col2 in num_cols:
            r_corr = real_df[[col1, col2]].corr().iloc[0, 1]
            s_corr = syn_df[[col1, col2]].corr().iloc[0, 1]
            per_pair_scores.append(0.5 * abs(r_corr - s_corr))

        # categorical–categorical → contingency TVD (Eq. 22)  :contentReference[oaicite:3]{index=3}
        elif col1 in cat_cols and col2 in cat_cols:
            r_tab = pd.crosstab(real_df[col1], real_df[col2], normalize='all')
            s_tab = pd.crosstab(syn_df[col1], syn_df[col2], normalize='all')
            idx = r_tab.index.union(s_tab.index)
            cols = r_tab.columns.union(s_tab.columns)
            r_tab = r_tab.reindex(index=idx, columns=cols, fill_value=0)
            s_tab = s_tab.reindex(index=idx, columns=cols, fill_value=0)
            per_pair_scores.append(0.5 * np.abs(r_tab - s_tab).values.sum())

        # mixed-type pairs are ignored (undefined in TabDiff)

    trend_score = float(np.mean(per_pair_scores)) if per_pair_scores else np.nan

    return shape_score, trend_score

def _align_schema(df, template_cols, cat_cols):
    out = df.copy()
    for col in template_cols:
        if col not in out:
            out[col] = np.nan if col not in cat_cols else "MISSING"
    return out[template_cols]


def compute_columnwise_mle(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    categorical_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
):
    cat_cols = set(categorical_columns)
    num_cols = [c for c in real_df.columns if c not in cat_cols]

    syn_df = _align_schema(syn_df, real_df.columns.tolist(), cat_cols)

    real_train, real_test = train_test_split(
        real_df, test_size=test_size, random_state=random_state
    )

    aucs, rmses = [], []

    for target in real_df.columns:
        cur_num = [c for c in num_cols if c != target]
        cur_cat = [c for c in cat_cols if c != target]
        predictors = cur_num + cur_cat

        # fit encoder on real-train predictor categoricals (if any)
        enc = None
        if cur_cat:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(real_train[cur_cat])

        def make_X(df):
            # numeric block
            if cur_num:
                Xn = df[cur_num].to_numpy()
                if Xn.ndim == 1:                       # → (n_samples, 1)
                    Xn = Xn.reshape(-1, 1)
            else:
                Xn = np.empty((len(df), 0))

            # categorical one-hot block
            if cur_cat:
                Xc = enc.transform(df[cur_cat])
            else:
                Xc = np.empty((len(df), 0))

            # guard against 1-D output (rare but possible)
            if Xc.ndim == 1:
                Xc = Xc.reshape(-1, 1)

            return np.hstack([Xn, Xc])

        X_syn = make_X(syn_df[predictors + [target]])
        y_syn = syn_df[target]

        X_test = make_X(real_test[predictors + [target]])
        y_test = real_test[target]

        if target in cat_cols:  # ---- classification
            # ------------------------------------------------------------------
            # 1. Fit a *single* encoder on the union of train & test labels
            #    so that class IDs line up everywhere.
            # ------------------------------------------------------------------
            le = LabelEncoder().fit(pd.concat([y_syn, y_test], ignore_index=True))

            y_syn_enc = le.transform(y_syn)  # ints 0 … K-1
            y_test_enc = le.transform(y_test)

            # ------------------------------------------------------------------
            # 2. Choose objective automatically
            # ------------------------------------------------------------------
            n_classes = len(le.classes_)
            obj = "binary:logistic" if n_classes == 2 else "multi:softprob"

            clf = XGBClassifier(
                objective=obj,
                num_class=None if n_classes == 2 else n_classes,
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                random_state=random_state, tree_method="hist"
            )

            clf.fit(X_syn, y_syn_enc)

            # ------------------------------------------------------------------
            # 3. Probabilities + AUC the same way you were doing it,
            #    but with the *encoded* y_test_enc
            # ------------------------------------------------------------------
            prob = clf.predict_proba(X_test)

            if n_classes == 2:  # binary
                temp_result = roc_auc_score(y_test_enc, prob[:, 1])
                print("column, ",target, "has RUC: ",temp_result)
                aucs.append(temp_result)
            else:  # multi-class
                temp_result =  roc_auc_score(label_binarize(
                        y_test_enc,
                        classes=np.arange(n_classes)),
                        prob,
                        average="macro",
                        multi_class="ovr",
                    )
                print("column, ",target, "has RUC: ",temp_result)
                aucs.append(temp_result)
        else:                                       # ---- regression
            reg = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, tree_method="hist",
            )
            reg.fit(X_syn, y_syn)
            pred = reg.predict(X_test)
            rmses.append(np.sqrt(mean_squared_error(y_test, pred)))

    avg_auc  = np.nan if not aucs  else float(np.nanmean(aucs))
    avg_rmse = np.nan if not rmses else float(np.mean(rmses))
    return avg_auc, avg_rmse

# file_path = "./results/results_detailed_1/Custom:Flow Matching Copula Synthesizer_adult.data.gz"
# file_path = "./results/adult/tabdiff/results_detailed_1/Custom_tabdiff_Synthesizer_adult.data.gz"
file_path = "./results/results_detailed_1/Custom:Flow Matching Copula Synthesizer_adult.data.gz"


with gzip.open(file_path, "rb") as f:
    df_synthesized = pickle.load(f)

data1, metadata1 = download_demo(
    modality='single_table',
    dataset_name='adult'
)

# print(metadata1.to_dict()['tables']['adult']['columns'])

numeric_cols, categorical_cols = [], []
for x in metadata1.to_dict()['tables']['adult']['columns']:
    column_dict = metadata1.to_dict()['tables']['adult']['columns'][x]
    if column_dict["sdtype"] == "id":
        continue
    if column_dict["sdtype"] == "numerical":
        numeric_cols.append(x)
    elif column_dict["sdtype"] == "categorical":
        categorical_cols.append(x)


def compute_dcr_score(real_df, syn_df, categorical_columns, test_size=0.5, random_state=42):
    df_real = real_df.copy()
    df_syn = syn_df.copy()
    df_real["__label__"] = 1
    df_syn["__label__"] = 0

    combined_df = pd.concat([df_real, df_syn], ignore_index=True)

    # Separate features and label
    X = combined_df.drop(columns="__label__")
    y = combined_df["__label__"]

    # Encode categorical columns
    cat_cols = categorical_columns
    num_cols = [c for c in X.columns if c not in cat_cols]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if cat_cols:
        X_cat = enc.fit_transform(X[cat_cols])
    else:
        X_cat = np.empty((len(X), 0))

    X_num = X[num_cols].to_numpy()
    if X_num.ndim == 1:
        X_num = X_num.reshape(-1, 1)

    X_all = np.hstack([X_num, X_cat])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train classifier
    # clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=random_state)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and compute AUC
    prob = clf.predict_proba(X_test)[:, 1]
    dcr_score = roc_auc_score(y_test, prob)

    return dcr_score


shape, trend = compute_shape_and_trend (data1, df_synthesized, categorical_cols)
avg_auc, avg_rmse = compute_columnwise_mle(data1, df_synthesized, categorical_cols)
dcr = compute_dcr_score (data1, df_synthesized, categorical_cols)

print("Shape score is: ",shape)
print("Trend score is: ",trend)
print("Avg auc is: ", avg_auc)
print("Avg rmse is: ",avg_rmse)
print("DCR score is: ",dcr)