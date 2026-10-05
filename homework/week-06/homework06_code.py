"""Homework 06: KNN vs. logistic regression (Questions 1 and 2).

Expects data/breast-cancer.csv and data/breast-cancer-split.csv.
Writes q1_roc.png and q2_roc.png to the current directory.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")

# ---------- Load data and the provided train/test split ----------
df = pd.read_csv("data/breast-cancer.csv")
sp = pd.read_csv("data/breast-cancer-split.csv")
assert (sp["row"].values == np.arange(1, len(df) + 1)).all()
is_train = (sp["split"] == "train").values

cols = ["mean_radius", "mean_texture", "mean_smoothness"]
X, y = df[cols].values, df["y"].values
Xtr, Xte, ytr, yte = X[is_train], X[~is_train], y[is_train], y[~is_train]

# Standardize with training means / SDs only
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0, ddof=1)
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

# ---------- Question 1 ----------
models = {"Logistic": LogisticRegression(C=np.inf, max_iter=5000).fit(Xtr, ytr)}  # no penalty
for k in [5, 15, 25, 35, 45]:
    models[f"KNN k={k}"] = KNeighborsClassifier(
        n_neighbors=k, metric="euclidean", weights="uniform"
    ).fit(Xtr, ytr)

plt.figure(figsize=(6.5, 6))
print("Question 1: test AUC")
for name, m in models.items():
    p = m.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    auc = roc_auc_score(yte, p)
    print(f"  {name:<10} {auc:.4f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Q1: test ROC curves")
plt.legend()
plt.savefig("q1_roc.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- Question 2 ----------
# predict_proba on training data includes each point among its own neighbors.
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
print("Question 2: train / test AUC")
for ax, k in zip(axes, [1, 3, 5, 10, 20]):
    m = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform").fit(Xtr, ytr)
    aucs = []
    for Xs, ys, label in [(Xtr, ytr, "Train"), (Xte, yte, "Test")]:
        p = m.predict_proba(Xs)[:, 1]
        fpr, tpr, _ = roc_curve(ys, p)
        auc = roc_auc_score(ys, p)
        aucs.append(auc)
        ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
    print(f"  k={k:<3} train={aucs[0]:.4f}  test={aucs[1]:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"k = {k}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
plt.savefig("q2_roc.png", dpi=130, bbox_inches="tight")
plt.close()
