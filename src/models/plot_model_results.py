from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from src.models.train_logistic import prepare_train_test_split

MODEL_PATH = Path("models/logistic_pipeline.joblib")
CONFUSION_MATRIX_PATH = Path("reports/figures/models/confusion_matrix_logistic_t040.png")
ROC_CURVE_PATH = Path("reports/figures/models/roc_curve_logistic.png")
PR_CURVE_PATH = Path("reports/figures/models/pr_curve_logistic.png")

def main():
    # Cargar modelo ya entrenado
    clf = joblib.load(MODEL_PATH)
    
    # Recuperar split reproducible
    X_train, X_test, y_train, y_test = prepare_train_test_split()
    
    # Probabilidades y predicción final con threshold operativo 0.40
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.40).astype(int)
    
    # Crear carpetas si no existen
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title("Matriz de confusión (Threshold=0.40)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 2. ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Curva ROC - Regresión Logística")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 3. Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Curva Precision-Recall")
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    
    print("=== Gráficos de Modelado Generados ===")
    print(f"Guardado: {CONFUSION_MATRIX_PATH}")
    print(f"Guardado: {ROC_CURVE_PATH}")
    print(f"Guardado: {PR_CURVE_PATH}")

if __name__ == "__main__":
    main()