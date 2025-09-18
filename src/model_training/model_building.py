# src/main.py

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.model_training.data_transformation import DataTransformer

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def main():
    # 1. Data Transformation
    transformer = DataTransformer()
    x_train, x_test, y_train, y_test = transformer.transform_and_save()

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    # ========================
    # 2. Define Models + Grids
    # ========================
    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    # ========================
    # 3. Train + Tune Models
    # ========================
    best_models = {}
    results = []

    for name, model in models.items():
        print(f"\n Tuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring="f1", n_jobs=-1)
        grid.fit(x_train, y_train)

        print(f" Best params for {name}: {grid.best_params_}")
        best_model = grid.best_estimator_
        best_models[name] = best_model

        # Evaluate on test set
        y_pred = best_model.predict(x_test)
        y_proba = best_model.predict_proba(x_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        print(acc)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

    # ========================
    # 4. Select Best Model
    # ========================
    df_results = pd.DataFrame(results)
    print("\n Model Comparison Results:")
    print(df_results)

    # Pick model with best F1-score
    best_row = df_results.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = best_models[best_model_name]

    print(f"\nBest model: {best_model_name} with F1 = {best_row['F1-Score']:.4f}")

    # ========================
    # 5. Save Best Model
    # ========================
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model saved at {model_path}")


if __name__ == "__main__":
    main()
