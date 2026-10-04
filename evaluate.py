from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).with_name("feature_engineered_churn_data.csv")
MODEL_PATH = Path(__file__).with_name("best_model_initial.pkl")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col].astype(int)
    return X, y


def train_model(X, y):
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churned", "Churned"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Best Model")
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    cost_per_false_negative = 500
    cost_per_false_positive = 50
    total_cost = (fn * cost_per_false_negative) + (fp * cost_per_false_positive)
    print(f"Estimated business cost of errors: ${total_cost}")

    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    print("Threshold analysis:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        custom_predictions = (y_probs >= threshold).astype(int)
        cm_custom = confusion_matrix(y_test, custom_predictions)
        tn, fp, fn, tp = cm_custom.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Threshold: {threshold} | Precision: {precision:.2f} | Recall: {recall:.2f}")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["Feature"][:15], importance_df["Importance"][:15])
        plt.xlabel("Importance")
        plt.title("Top 15 Feature Importances")
        plt.gca().invert_yaxis()
        plt.show()

        print(importance_df.head(15).to_string(index=False))

    models = {"Best Model": model}
    for name, model_for_eval in models.items():
        model_predictions = model_for_eval.predict(X_test)
        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"{'=' * 50}")
        print(classification_report(y_test, model_predictions, target_names=["Not Churned", "Churned"]))


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if MODEL_PATH.exists():
        best_model = joblib.load(MODEL_PATH)
    else:
        from sklearn.linear_model import LogisticRegression

        best_model = LogisticRegression(max_iter=2000, random_state=42)
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, MODEL_PATH)

    evaluate_model(best_model, X_test, y_test)
