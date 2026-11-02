<USER_REQUEST>
# XGBoost Complete Guide: 10 Lessons + 5 Python Labs

A consolidated, offline-first Markdown tutorial covering gradient-boosted trees, classification, regression, DMatrix, early stopping, native categorical features, model tuning and saving. Install dependencies from `requirements.txt` to run examples from the repository root.

## Chapters

1. [What Is XGBoost? Gradient Boosting Explained](docs/01-what-is-xgboost.md)
2. [XGBClassifier Tutorial: Binary Classification](docs/02-xgbclassifier.md)
3. [XGBRegressor Tutorial: RMSE, MAE and Residuals](docs/03-xgbregressor.md)
4. [DMatrix and xgb.train: Native XGBoost Python API](docs/04-dmatrix.md)
5. [XGBoost Early Stopping Without Data Leakage](docs/05-early-stopping.md)
6. [XGBoost Hist vs Exact: Tree Methods and max_bin](docs/06-hist-vs-exact.md)
7. [XGBoost Categorical Features and Missing Values](docs/07-categorical.md)
8. [XGBoost GridSearchCV: Tune on Training Data Only](docs/08-gridsearch.md)
9. [XGBoost Save Model JSON, Feature Importance and Deployment](docs/09-save-interpret.md)
10. [5 Hands-on XGBoost Projects with Python Source Code](docs/10-projects.md)

---

## Lesson 1


# What Is XGBoost? Gradient Boosting Explained

**XGBoost** (Extreme Gradient Boosting) trains an additive ensemble of decision trees. New boosting rounds make corrections to previous predictions by optimizing an objective function using gradients and, where applicable, second-order information. The `learning_rate` controls how much each new tree contributes.

The model is useful on **tabular data** but is not always better than linear models, random forests or neural networks. The right model depends on your dataset, evaluation metric and deployment constraints.

```mermaid
flowchart TD
  INIT["1. Initial Base Prediction: y_hat_0 = log-odds / mean"] --> ROUNDS["2. Sequential Boosting Rounds t = 1 ... T"]

  subgraph ROUND ["Round t Optimization"]
    direction TB
    GRAD["Compute 1st Gradients (g_i) and 2nd Hessians (h_i)"]
    TREE["Build Decision Tree f_t(x) maximizing Gain"]
    SHRINK["Scale Tree Output by Learning Rate (eta)"]
    UPDATE["Update Accumulator: y_hat_t = y_hat_{t-1} + eta * f_t(x)"]
    GRAD --> TREE --> SHRINK --> UPDATE
  end

  ROUNDS --> ROUND
  ROUND --> ENSEMBLE["3. Final Additive Ensemble: y_hat = y_hat_0 + sum(eta * f_t(x))"]
```

## Runnable example

```python
from labs.common import classification_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
X, y = classification_data(n_samples=400)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=.25, random_state=42)
model = XGBClassifier(n_estimators=70, max_depth=3, tree_method="hist", n_jobs=1, random_state=42)
model.fit(X_tr, y_tr)
print('ROC AUC:', round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3))
```

## Important habit

Do not report accuracy measured on the same rows used for fitting as an unbiased performance estimate.

## Try it yourself

Vary `max_depth` (1, 3, 6) while keeping your test set fixed. Do the results always improve?

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 2


# XGBClassifier Tutorial: Binary Classification

`XGBClassifier` uses the familiar scikit-learn estimator API: `.fit`, `.predict` and `.predict_proba`. Class labels answer *which class?*; probabilities support threshold-based policies and ranking metrics.

| Metric | Meaning | Main caveat |
| --- | --- | --- |
| ROC AUC | Ranking of positive over negative cases | May be less informative under extreme class imbalance |
| Balanced accuracy | Average class-specific recall | Ignores probabilistic calibration |
| Precision | Correct among predicted positives | Depends on the threshold |
| Recall | Positive cases found | Can trade off with false alarms |
| Average precision | Summary of precision-recall ranking | Depends on prevalence |

```mermaid
flowchart LR
  INPUT["Feature Vector x"] --> TREES["Ensemble of T Trees"]
  TREES --> MARGIN["Raw Margin Output: sum(eta * f_t(x))"]
  MARGIN --> SIGMOID["Logistic Sigmoid: 1 / (1 + exp(-z))"]
  SIGMOID --> PROBA["Predicted Probability P(y=1|x)"]
  PROBA --> THRESH{"Decision Threshold (tau = 0.50)"}
  THRESH -->|P >= tau| POS["Positive Class (1)"]
  THRESH -->|P < tau| NEG["Negative Class (0)"]
```

## Runnable example

```python
from labs.common import classification_data
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from xgboost import XGBClassifier
X, y = classification_data()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
clf = XGBClassifier(n_estimators=80, tree_method='hist', eval_metric='logloss', n_jobs=1, random_state=42).fit(X_tr,y_tr)
base = DummyClassifier(strategy='prior').fit(X_tr,y_tr)
print('Balanced accuracy:', balanced_accuracy_score(y_te, clf.predict(X_te)))
print('Model ROC AUC:', roc_auc_score(y_te, clf.predict_proba(X_te)[:,1]))
print('Baseline ROC AUC:', roc_auc_score(y_te, base.predict_proba(X_te)[:,1]))
```

## Important habit

Use a stratified split for appropriate classification datasets; report a baseline rather than only the model score.

## Try it yourself

Change the class balance of the synthetic dataset and compare ROC AUC with average precision.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 3


# XGBRegressor Tutorial: RMSE, MAE and Residuals

`XGBRegressor` predicts a number. Start with `objective="reg:squarederror"` and assess performance on an unseen test set.

- **MAE:** average absolute error, in target units.
- **RMSE:** square root of mean squared error; penalizes larger errors more.
- **R²:** improvement over a constant-mean predictor by squared-error comparison; can be negative.

Residual plots help detect patterns in errors. They do not prove that a model is unbiased or deployable.

```mermaid
flowchart TD
  TARGET["Continuous Target y"] --> BASE_PRED["Initial Prediction: y_hat_0 = mean(y)"]
  BASE_PRED --> RESID_1["Residual r_1 = y - y_hat_0"]
  RESID_1 --> TREE_1["Tree 1 fits r_1"]
  TREE_1 --> RESID_2["Residual r_2 = y - (y_hat_0 + eta * f_1(x))"]
  RESID_2 --> TREE_2["Tree 2 fits r_2"]
  TREE_2 --> METRICS["Evaluation on Unseen Test Set: RMSE, MAE, R-squared"]
```

## Runnable example

```python
from labs.common import regression_data
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
X, y = regression_data()
X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=.25,random_state=42)
model = XGBRegressor(n_estimators=160,max_depth=3,learning_rate=.07,tree_method='hist',n_jobs=1,random_state=42).fit(X_tr,y_tr)
base = DummyRegressor().fit(X_tr,y_tr)
print('MAE:', round(mean_absolute_error(y_te,model.predict(X_te)),3))
print('RMSE:',round(root_mean_squared_error(y_te,model.predict(X_te)),3))
print('Baseline RMSE:',round(root_mean_squared_error(y_te,base.predict(X_te)),3))
```

## Important habit

Do not compare a training-set RMSE with a held-out baseline RMSE; compute both on the same rows.

## Try it yourself

Compare `max_depth=2` and `max_depth=8`, then inspect the residual plot generated by Lab 02.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 4


# DMatrix and xgb.train: Native XGBoost Python API

**`DMatrix`** stores features, labels and optional metadata in a format for native XGBoost training. It is **not** a replacement for a pandas DataFrame when cleaning or joining tables.

| API | Training call | Typical choice |
| --- | --- | --- |
| `XGBClassifier` / `XGBRegressor` | `model.fit(X, y)` | scikit-learn model selection and pipelines |
| `xgb.DMatrix` + `xgb.train` | `xgb.train(params, dtrain)` | Native evaluation, callbacks, custom logic |
| `xgb.QuantileDMatrix` | Native training | Memory-oriented histogram data representation |

Keep feature names and column order consistent between training and inference.

```mermaid
flowchart TD
  subgraph WRAPPER ["Scikit-Learn Wrapper API (XGBClassifier / XGBRegressor)"]
    W1["Accepts pandas DataFrame & NumPy arrays directly"]
    W2["Standard .fit(X, y) and .predict(X) methods"]
    W3["Integrates with scikit-learn Pipeline & GridSearchCV"]
  end

  subgraph NATIVE ["Native XGBoost Python API (xgb.train)"]
    N1["Explicit xgb.DMatrix / xgb.QuantileDMatrix container"]
    N2["Fine-grained training params dict: eta, max_depth, objective"]
    N3["Advanced callbacks, custom objectives, multi-eval sets"]
  end

  INPUT_DATA["Tabular Machine Learning Project"] --> CHOICE{"Development Workflow?"}
  CHOICE -->|"Scikit-learn Pipelines"| WRAPPER
  CHOICE -->|"Native Performance & Custom Callbacks"| NATIVE
```

## Runnable example

```python
import xgboost as xgb
from labs.common import classification_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
X,y=classification_data(n_samples=360)
X_tr,X_te,y_tr,y_te=train_test_split(X,y,stratify=y,test_size=.25,random_state=42)
train=xgb.DMatrix(X_tr,label=y_tr)
test=xgb.DMatrix(X_te,label=y_te)
params={'objective':'binary:logistic','tree_method':'hist','eval_metric':'logloss','eta':.08,'max_depth':3,'nthread':1,'seed':42}
booster=xgb.train(params,train,num_boost_round=80)
print('ROC AUC:',roc_auc_score(y_te,booster.predict(test)))
```

## Important habit

For native `xgb.train`, the parameter is commonly called `eta` rather than `learning_rate` (both are aliases).

## Try it yourself

Build the same classifier once with `XGBClassifier` and once with `xgb.train`, then compare metrics.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 5


# XGBoost Early Stopping Without Data Leakage

Early stopping watches a **validation** metric and chooses the best boosting iteration when subsequent iterations stop improving. It must not use the untouched **final test set**.

```mermaid
flowchart TD
  DATA["All Available Data"] --> SPLIT["Data Partitioning"]
  SPLIT -->|"80% Development Pool"| DEV["Development Pool"]
  SPLIT -->|"20% Test Vault"| TEST["Untouched Test Vault"]

  DEV --> DEV_SPLIT["Inner Train / Validation Split"]
  DEV_SPLIT -->|"75% Fit Set"| FIT["Training Subset (Fit Tree Splits)"]
  DEV_SPLIT -->|"25% Val Set"| VAL["Validation Subset (Track LogLoss)"]

  FIT & VAL --> LOOP["Boosting Loop: Monitor Val Metric each Round"]
  LOOP --> STOP{"No improvement for 12 rounds?"}
  STOP -->|"Best Round Detected"| BEST["Best Iteration Selected (best_iteration)"]
  BEST --> FINAL["Evaluate Tuned Model ONCE on Test Vault"]
  TEST --> FINAL
```

For the scikit-learn wrapper, `early_stopping_rounds` belongs in the estimator constructor. The native `Booster` may retain later iterations, so choose `iteration_range=(0, booster.best_iteration + 1)` for prediction with the best iteration.

## Runnable example

```python
from labs.common import classification_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
X,y=classification_data(n_samples=640)
Xp,Xtest,yp,ytest=train_test_split(X,y,stratify=y,test_size=.2,random_state=42)
Xtr,Xval,ytr,yval=train_test_split(Xp,yp,stratify=yp,test_size=.25,random_state=42)
clf=XGBClassifier(n_estimators=220,tree_method='hist',learning_rate=.06,max_depth=3,
                  early_stopping_rounds=12,eval_metric='logloss',n_jobs=1,random_state=42)
clf.fit(Xtr,ytr,eval_set=[(Xval,yval)],verbose=False)
print('Best iteration:',clf.best_iteration)
print('Final ROC AUC:',roc_auc_score(ytest,clf.predict_proba(Xtest)[:,1]))
```

## Important habit

Using `eval_set=[(X_test, y_test)]` to select a model leaks information from your final test set.

## Try it yourself

Find the selected boosting round, then inspect Lab 03’s validation-loss curves.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 6


# XGBoost Hist vs Exact: Tree Methods and max_bin

Two supported tree-building strategies are **`hist`** (quantizes numeric features into bins) and **`exact`** (evaluates exact split candidates). `hist` is generally a strong production default because of its efficiency.

| Argument | Use |
| --- | --- |
| `tree_method="hist"` | Histogram-optimized approximate splits, native categorical support |
| `tree_method="exact"` | Exact greedy split search on numeric data; native categorical support unavailable |
| `max_bin=256` | Number of numeric bins for `hist`/`approx`, not `exact` |
| `device="cpu"` | CPU mode; included labs require no GPU |
| `device="cuda"` | Optional CUDA acceleration when supported |

Whether hist is faster or more accurate depends on dataset, training configuration and machine. A small code example is **not** a defensible global speed benchmark.

```mermaid
flowchart TD
  FEATURES["Continuous Numerical Features"] --> METHOD{"tree_method Selection"}

  subgraph EXACT ["Exact Greedy Algorithm (tree_method='exact')"]
    E1["Sorts all unique continuous values across dataset"]
    E2["Evaluates N-1 candidate split points per feature"]
    E3["High memory & O(N * d) complexity per node"]
  end

  subgraph HIST ["Histogram Optimization (tree_method='hist')"]
    H1["Quantizes continuous features into max_bin bins (e.g. 256)"]
    H2["Builds gradient & Hessian histograms per node"]
    H3["Constant time O(K * d) per node + Native Categorical Support"]
  end

  METHOD -->|"Small data / exact baseline"| EXACT
  METHOD -->|"Production default / large data"| HIST
```

## Runnable example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
X,y=make_regression(n_samples=250,n_features=6,random_state=31)
Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=31)
for method in ('hist','exact'):
    model=XGBRegressor(tree_method=method,n_estimators=50,max_depth=2,n_jobs=1,random_state=31).fit(Xtr,ytr)
    print(method, round(root_mean_squared_error(yte,model.predict(Xte)),3))
```

## Important habit

`tree_method="exact"` is not the original meaning of the word “XGBoost”; XGBoost supports multiple split-building methods.

## Try it yourself

Try `max_bin=32` versus `max_bin=256` with hist, then compare predictive error.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 7


# XGBoost Categorical Features and Missing Values

XGBoost handles `NaN` missing numeric values using learned default branches. Native categorical feature handling uses pandas columns typed as **`category`** and `enable_categorical=True` with a supported tree method.

Starting with XGBoost **3.1**, supported Python DataFrame inputs can carry category encoding information into the model to support recoding at prediction time. Still validate input data types, feature names and the domain of allowed values.

**Limitations:** plain Python object-string columns are not automatically categorical. Learned NaN routing does not repair missing source data. Categorical models should be serialized with JSON or UBJSON.

```mermaid
flowchart LR
  subgraph CAT ["Native Categorical Features"]
    C1["pandas column .astype('category')"] --> C2["enable_categorical=True"]
    C2 --> C3["Optimal one-hot / partition splits without dimension blowup"]
  end

  subgraph MISSING ["Default Missing Value Routing"]
    M1["Numeric np.nan present in data"] --> M2["Algorithm computes split gain: NaN -> Left vs NaN -> Right"]
    M2 --> M3["Picks direction with higher Gain as permanent default path"]
  end
```

## Runnable example

```python
from labs.common import churn_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X,y=churn_data() # bundled synthetic CSV; plan/region converted to category
Xtr,Xte,ytr,yte=train_test_split(X,y,stratify=y,test_size=.25,random_state=19)
model=XGBClassifier(n_estimators=80,tree_method='hist',enable_categorical=True,
                    eval_metric='logloss',random_state=19,n_jobs=1)
model.fit(Xtr,ytr)
print('Missing values:',int(X.isna().sum().sum()))
print('First predicted probabilities:',model.predict_proba(Xte)[:3,1].round(3))
```

## Important habit

Treat model feature gain as association, not cause-and-effect; verify data types when deploying.

## Try it yourself

Try dropping a numeric feature and compare metrics without changing the train/test split.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 8


# XGBoost GridSearchCV: Tune on Training Data Only

Small parameter searches help learners build intuition before expensive tuning. `max_depth` changes complexity, `learning_rate` changes each update, `subsample` and `colsample_bytree` add sampling, and `reg_lambda` applies L2 regularization.

1. Split off the final test set **before** tuning.
2. Run `GridSearchCV` with stratified folds inside the training set.
3. Select the best cross-validated estimator.
4. Report the test metric **once** after model selection.

Avoid a global external early-stopping validation set shared indiscriminately across CV folds.

```mermaid
flowchart TD
  PARAMS["XGBoost Hyperparameter Architecture"] --> TREE_S["Tree Architecture"]
  PARAMS --> STOCH["Stochastic Sampling"]
  PARAMS --> REG["Regularization & Step"]

  TREE_S --> P_DEPTH["max_depth: Maximum tree depth (e.g. 3-6)"]
  TREE_S --> P_CHILD["min_child_weight: Minimum Hessian sum in leaf"]
  TREE_S --> P_GAMMA["gamma: Minimum loss reduction for split"]

  STOCH --> P_SUB["subsample: Fraction of rows sampled per tree"]
  STOCH --> P_COL["colsample_bytree: Fraction of features per tree"]

  REG --> P_ETA["learning_rate (eta): Shrinkage factor (0.01 - 0.2)"]
  REG --> P_L2["reg_lambda: L2 leaf weight penalty"]
  REG --> P_L1["reg_alpha: L1 sparsity penalty"]
```

## Runnable example

```python
from labs.common import classification_data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
X,y=classification_data(n_samples=450)
Xtr,Xte,ytr,yte=train_test_split(X,y,stratify=y,test_size=.25,random_state=42)
base=XGBClassifier(n_estimators=55,learning_rate=.08,tree_method='hist',n_jobs=1,random_state=42)
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
search=GridSearchCV(base,{'max_depth':[2,4],'subsample':[.8,1.]},scoring='roc_auc',cv=cv,n_jobs=1)
search.fit(Xtr,ytr)
print('CV AUC:',search.best_score_)
print('Held-out AUC:',roc_auc_score(yte,search.predict_proba(Xte)[:,1]))
```

## Important habit

Do not select hyperparameters based on which combination gives the highest test-set score.

## Try it yourself

Try a small `min_child_weight` grid and compare CV variability.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 9


# XGBoost Save Model JSON, Feature Importance and Deployment

**Portable model IO:** `model.save_model("name.json")` or `.ubj` saves XGBoost trees and model information. It does not automatically include every external preprocessing step, data schema, metric threshold, or deployment contract.

**Feature importance:** `weight` counts split usage; `gain` measures average split-related loss improvement; `cover` reflects average coverage. They are not proof of causality.

For production readiness, document input names/types/order, version dependencies, model provenance, expected class balance, latency constraints, metric thresholds, drift detection and rollback plans.

```mermaid
flowchart TD
  BOOSTER["Trained XGBoost Booster"] --> EXPORT["model.save_model('model.json')"]
  EXPORT --> JSON["Safe JSON / UBJSON Artifact (No arbitrary code execution)"]

  BOOSTER --> IMP{"Feature Importance Metrics"}
  IMP -->|"Weight"| I_W["Weight: Count of times feature used for splitting"]
  IMP -->|"Gain"| I_G["Gain: Average reduction in loss contributed by feature"]
  IMP -->|"Cover"| I_C["Cover: Average relative sample count affected by feature"]
```

## Runnable example

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from labs.common import classification_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X,y=classification_data(n_samples=350)
Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=42)
model=XGBClassifier(tree_method='hist',n_estimators=55,n_jobs=1,random_state=42).fit(Xtr,ytr)
with TemporaryDirectory() as d:
    path=Path(d)/'model.json'
    model.save_model(path)
    reloaded=XGBClassifier()
    reloaded.load_model(path)
    print('Reload matches:',np.allclose(model.predict_proba(Xte),reloaded.predict_proba(Xte)))
```

## Important habit

Avoid loading untrusted pickle or joblib files; keep model artifacts and upstream data trustworthy.

## Try it yourself

Write a short model card describing the training dataset and known evaluation limitations.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

## Lesson 10


# 5 Hands-on XGBoost Projects with Python Source Code

The five scripts below are **real executable Python projects**, not pseudocode, and they run fully offline on bundled synthetic data.

| Lab | Problem | Output |
| --- | --- | --- |
| [`labs/lab01_xgb_classification.py`](labs/lab01_xgb_classification.py) | Binary classification versus a dummy baseline | Confusion matrix + ROC |
| [`labs/lab02_xgb_regression.py`](labs/lab02_xgb_regression.py) | Nonlinear regression | Predicted vs actual + residuals |
| [`labs/lab03_dmatrix_early_stopping.py`](labs/lab03_dmatrix_early_stopping.py) | Native `DMatrix` and early stopping | Training/validation log loss |
| [`labs/lab04_categorical_model_io.py`](labs/lab04_categorical_model_io.py) | Missing values, categoricals, JSON reload | Gain importance + model file |
| [`labs/lab05_grid_search.py`](labs/lab05_grid_search.py) | Cross-validated hyperparameter selection | CV comparison + ROC |

```mermaid
flowchart TD
  subgraph DATA ["Synthetic Datasets"]
    D1["classification_data: 400 samples, 8 features"]
    D2["regression_data: 400 samples, 6 features"]
    D3["churn_data: 500 samples, native categoricals"]
  end

  subgraph LABS ["Five Runnable Python Labs (labs/)"]
    L1["Lab 1: lab01_xgb_classification.py<br/>XGBClassifier vs Dummy + Confusion Matrix & ROC"]
    L2["Lab 2: lab02_xgb_regression.py<br/>XGBRegressor vs Dummy + Residual Diagnostics"]
    L3["Lab 3: lab03_dmatrix_early_stopping.py<br/>Native DMatrix + Early Stopping LogLoss Curves"]
    L4["Lab 4: lab04_categorical_model_io.py<br/>Missing Values + Categorical Gain + JSON Reload"]
    L5["Lab 5: lab05_grid_search.py<br/>StratifiedKFold GridSearchCV + Validation Heatmap"]
  end

  subgraph ARTIFACTS ["Generated Visual Outputs (outputs/)"]
    O1["01-xgboost-classification-roc.png / .svg"]
    O2["02-xgboost-regression-residuals.png / .svg"]
    O3["03-xgboost-early-stopping.png / .svg"]
    O4["04-xgboost-feature-importance.png / .svg"]
    O5["05-xgboost-grid-search.png / .svg"]
  end

  D1 --> L1 --> O1
  D2 --> L2 --> O2
  D1 --> L3 --> O3
  D3 --> L4 --> O4
  D1 --> L5 --> O5
```

Use the scripts' `run()` functions to explore returned metrics or modify the hyperparameters. The scripts save PNG and SVG files under `outputs/`. All charts are computed from synthetic datasets and do not represent production performance.

## Runnable example

```python
from labs.lab01_classification import run as classification
from labs.lab03_dmatrix_early_stopping import run as early_stopping
print('Classification metrics:', classification())
print('Early stopping metrics:', early_stopping())
```

## Important habit

Only report model scores from new unseen data with assumptions and dataset limitations clearly stated.

## Try it yourself

Modify any two experiments, then write down exactly which split and metric you used.

**Continue:** [Complete tutorial](XGBOOST_COMPLETE_GUIDE.md) · [Cheat sheet](docs/cheatsheet.md) · [FAQ](docs/faq.md).

---

# XGBoost Python Cheat Sheet

XGBoost is a gradient-boosted decision-tree library for tabular data. Use `XGBClassifier` for categories, `XGBRegressor` for numbers, or native `DMatrix` + `xgb.train` for low-level control.

| Topic | Code / setting | When to use |
| --- | --- | --- |
| Install | `pip install xgboost` | Local training |
| Classify | `XGBClassifier().fit(X_train, y_train)` | Label prediction |
| Regress | `XGBRegressor().fit(X_train, y_train)` | Numeric prediction |
| Native data | `xgb.DMatrix(X, label=y)` | Native train API |
| Native train | `xgb.train(params, dtrain)` | Booster/callback control |
| Hist | `tree_method="hist"` | Usually a good starting point |
| Exact | `tree_method="exact"` | Small-data numeric comparisons |
| Early stopping | `early_stopping_rounds=15` | Validation-driven selection |
| Native categories | `.astype("category")`, `enable_categorical=True` | DataFrame categorical columns |
| Handle NaN | Leave `np.nan` in numeric features | Learned missing-value branches |
| Save model | `model.save_model("model.json")` | Portable XGBoost trees |
| Cross validate | `GridSearchCV(..., cv=3)` | Tune only on training data |

```mermaid
flowchart TD
  START["Systematic XGBoost Tuning Strategy"] --> S1["Stage 1: Baseline & Learning Pace"]
  S1 --> S1_ACT["Set tree_method='hist'<br/>Set learning_rate=0.05 - 0.1<br/>Use early_stopping_rounds=15"]

  S1_ACT --> S2["Stage 2: Tree Complexity"]
  S2 --> S2_ACT["Tune max_depth (3 to 6)<br/>Tune min_child_weight (1 to 10)"]

  S2_ACT --> S3["Stage 3: Randomness & Subsampling"]
  S3 --> S3_ACT["Tune subsample (0.7 to 1.0)<br/>Tune colsample_bytree (0.6 to 1.0)"]

  S3_ACT --> S4["Stage 4: Regularization Penalties"]
  S4 --> S4_ACT["Tune reg_lambda (L2: 1.0 to 10.0)<br/>Tune reg_alpha (L1: 0 to 5.0)"]
```

**Core tuning parameters:** `n_estimators`, `learning_rate` (`eta`), `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, `max_bin`.

**Always:** split before fitting, compare a baseline, choose metrics for your business question, keep validation separate from final test, store input schemas.

---

# XGBoost FAQ — Common Beginner Questions

### What is XGBoost?
XGBoost is an optimized gradient-boosting library that learns tree-based predictive ensembles from tabular features.

### Is XGBoost part of scikit-learn?
No. It is a separate library with scikit-learn-compatible estimators.

### What is the difference between XGBClassifier and XGBRegressor?
Classification estimates labels or class probabilities; regression estimates numbers.

### What is `DMatrix`?
XGBoost's native container for features, labels and metadata used by `xgb.train`.

### What is the difference between `hist` and `exact`?
`hist` searches splits on bins; `exact` considers all eligible split candidates. Neither is guaranteed to always be more accurate.

### Does XGBoost need feature scaling?
Tree-based boosters generally do not require standardizing numeric inputs. Other components in a broader pipeline might.

### How can I use categorical data?
Convert supported pandas columns to `category`, then pass `enable_categorical=True` and a compatible tree method such as `hist`.

### Does XGBoost support missing values?
Yes, numeric `NaN` can be handled by learned missing-value routing; it does not restore missing data.

### How does early stopping work?
It selects the boosting iteration using a validation metric. Do not use the untouched final test set for stopping.

### Why can ROC AUC look good but precision be poor?
ROC AUC evaluates ranking across thresholds, while precision depends on a specific threshold and class prevalence.

### What is the best XGBoost learning rate?
There is no universal optimal learning rate. Lower rates generally need more rounds; tune with train-only cross-validation and appropriate metrics.

### How can I save a model?
Use `model.save_model("model.json")`, and record feature schema and any external preprocessing as separate assets.

### Can I use these example models in production?
Not without validating the dataset, quality, security and operations for the real use case. The lab data are synthetic and educational.

---

# XGBoost Glossary — Concepts in Plain English

| Concept | Meaning |
| --- | --- |
| Boosting | Sequentially combining learners to improve predictions |
| GBDT | Gradient-boosted decision-tree ensemble |
| Gradient | First derivative of the objective with respect to predictions |
| Hessian | Second derivative information for supported losses |
| Learning rate (`eta`) | Scale applied to each boosting update |
| Boosting round | Model update step, potentially involving multiple trees |
| `hist` | Histogram-based approximate numeric split search |
| `exact` | Exact greedy split search |
| `max_bin` | Maximum numeric bins for histogram/approximate splitting |
| `DMatrix` | Native data format for the Booster API |
| `Booster` | Trained native XGBoost model |
| `eval_metric` | Metric reported during training and validation |
| Early stopping | Select boosting iteration based on a validation metric |
| `best_iteration` | Zero-based best boosting iteration |
| `stratify` | Maintain class proportions during sampling/splitting |
| ROC AUC | Binary-class ranking discrimination |
| RMSE | Root mean squared prediction error |
| Gain importance | Split-improvement-based feature importance, not causality |
| Data leakage | Information contaminating training or evaluation unfairly |
| JSON / UBJSON | Supported model serialization formats |

</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-02T07:00:02+05:30.
</ADDITIONAL_METADATA>