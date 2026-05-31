# 03 — Explainability

## Objetivo
Este notebook documenta la explicabilidad del modelo final seleccionado, la **regresión logística**, desde tres niveles complementarios:

1. **Explicabilidad global**: qué variables son las más influyentes en el modelo.
2. **Explicabilidad local**: por qué el modelo asigna alto o bajo riesgo a clientes concretos.
3. **Explicabilidad agrupada**: qué perfil caracteriza al top 10% de clientes con mayor riesgo estimado.

## Preguntas que responde
- ¿Qué variables empujan el churn hacia arriba o hacia abajo?
- ¿Cómo se explica una predicción individual?
- ¿Qué patrón colectivo caracteriza al grupo de mayor riesgo?
- ¿Cómo se conecta la explicabilidad con la decisión de negocio?


```python
from pathlib import Path
import sys

# Buscar la raíz del proyecto (la carpeta que contiene "src")
PROJECT_ROOT = Path.cwd().resolve()

while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
```

    PROJECT_ROOT = C:\repos\tfm-proyecto1-churn
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import sys
import os

# Ajuste de path si es necesario para importar desde src/
sys.path.append(os.path.abspath(os.path.join('..')))

from src.models.train_logistic import (
    build_logistic_pipeline,
    prepare_train_test_split,
)
from src.explainability.global_explainability import (
    load_or_train_logistic_model,
    build_odds_ratio_table,
    build_selected_summary,
)
from src.explainability.grouped_explainability import (
    build_grouped_explainability_tables,
)
```


```python
clf_log = load_or_train_logistic_model()
X_train, X_test, y_train, y_test = prepare_train_test_split()
```

## BLOQUE A — Explicabilidad Global
Analizamos el peso y la dirección de cada variable en el modelo a través de sus coeficientes y *Odds Ratios*.


```python
coef_df = build_odds_ratio_table(clf_log)
coef_df.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
      <th>odds_ratio</th>
      <th>abs_coefficient</th>
      <th>direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__tenure</td>
      <td>-1.255202</td>
      <td>0.285018</td>
      <td>1.255202</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat__Contract_Two year</td>
      <td>-0.759402</td>
      <td>0.467946</td>
      <td>0.759402</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__InternetService_Fiber optic</td>
      <td>0.648027</td>
      <td>1.911766</td>
      <td>0.648027</td>
      <td>Aumenta riesgo de churn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__InternetService_DSL</td>
      <td>-0.643642</td>
      <td>0.525375</td>
      <td>0.643642</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>num__MonthlyCharges</td>
      <td>-0.601168</td>
      <td>0.548171</td>
      <td>0.601168</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__Contract_Month-to-month</td>
      <td>0.585998</td>
      <td>1.796784</td>
      <td>0.585998</td>
      <td>Aumenta riesgo de churn</td>
    </tr>
    <tr>
      <th>6</th>
      <td>num__TotalCharges</td>
      <td>0.532592</td>
      <td>1.703341</td>
      <td>0.532592</td>
      <td>Aumenta riesgo de churn</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__PaperlessBilling_No</td>
      <td>-0.330735</td>
      <td>0.718396</td>
      <td>0.330735</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat__DeviceProtection_No internet service</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cat__OnlineSecurity_No internet service</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cat__InternetService_No</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cat__OnlineBackup_No internet service</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cat__TechSupport_No internet service</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cat__StreamingTV_No internet service</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cat__StreamingMovies_No internet service</td>
      <td>-0.292955</td>
      <td>0.746056</td>
      <td>0.292955</td>
      <td>Reduce riesgo de churn</td>
    </tr>
  </tbody>
</table>
</div>




```python
summary_df = build_selected_summary(coef_df)
summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>feature</th>
      <th>coefficient</th>
      <th>odds_ratio</th>
      <th>direction</th>
      <th>interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tenure (antigüedad del cliente)</td>
      <td>num__tenure</td>
      <td>-1.255</td>
      <td>0.285</td>
      <td>Reduce riesgo de churn</td>
      <td>Una mayor antigüedad del cliente se asocia con...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Contrato: Two year</td>
      <td>cat__Contract_Two year</td>
      <td>-0.759</td>
      <td>0.468</td>
      <td>Reduce riesgo de churn</td>
      <td>Tener un contrato de dos años se asocia con un...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Contrato: Month-to-month</td>
      <td>cat__Contract_Month-to-month</td>
      <td>0.586</td>
      <td>1.797</td>
      <td>Aumenta riesgo de churn</td>
      <td>El contrato mensual se asocia con un aumento i...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>InternetService: Fiber optic</td>
      <td>cat__InternetService_Fiber optic</td>
      <td>0.648</td>
      <td>1.912</td>
      <td>Aumenta riesgo de churn</td>
      <td>La categoría Fiber optic se asocia con mayores...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>InternetService: DSL</td>
      <td>cat__InternetService_DSL</td>
      <td>-0.644</td>
      <td>0.525</td>
      <td>Reduce riesgo de churn</td>
      <td>La categoría DSL se asocia con menor riesgo es...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PaymentMethod: Electronic check</td>
      <td>cat__PaymentMethod_Electronic check</td>
      <td>0.206</td>
      <td>1.228</td>
      <td>Aumenta riesgo de churn</td>
      <td>Electronic check se asocia con un aumento mode...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MonthlyCharges</td>
      <td>num__MonthlyCharges</td>
      <td>-0.601</td>
      <td>0.548</td>
      <td>Reduce riesgo de churn</td>
      <td>El efecto de MonthlyCharges cambia respecto al...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TotalCharges</td>
      <td>num__TotalCharges</td>
      <td>0.533</td>
      <td>1.703</td>
      <td>Aumenta riesgo de churn</td>
      <td>TotalCharges mantiene una asociación positiva,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from IPython.display import Image, display
from pathlib import Path

odds_plot_path = PROJECT_ROOT / "reports" / "figures" / "models" / "odds_ratio_plot.png"

print(odds_plot_path)
display(Image(filename=str(odds_plot_path)))
```

    C:\repos\tfm-proyecto1-churn\reports\figures\models\odds_ratio_plot.png
    


    
![png](03_explainability_files/03_explainability_7_1.png)
    


## Interpretación global

La explicabilidad global confirma que la **antigüedad del cliente (`tenure`)** y la **estabilidad contractual** son los principales factores protectores frente al churn. En sentido contrario, el contrato mensual, el servicio `Fiber optic` y el método de pago `Electronic check` aparecen asociados con mayor riesgo estimado.

Estos resultados son coherentes con el EDA y refuerzan la validez del modelo desde un punto de vista tanto estadístico como de negocio.

## BLOQUE B — Explicabilidad Local
Para entender cómo el modelo calcula la probabilidad exacta de un individuo, extraemos el peso de cada variable para clientes específicos de nuestro conjunto de test.


```python
preprocessor = clf_log.named_steps["preprocessor"]
model = clf_log.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
coefs = model.coef_
intercept = model.intercept_

X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_test_transformed_df = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

contrib_df = X_test_transformed_df.mul(coefs, axis=1)

log_odds = intercept + contrib_df.sum(axis=1)
proba_from_contrib = 1 / (1 + np.exp(-log_odds))

y_proba = clf_log.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.50).astype(int)

local_summary = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_proba": y_proba
}, index=X_test.index)
```


```python
tp_candidates = local_summary[
    (local_summary["y_true"] == 1) & (local_summary["y_pred"] == 1)
]
case_tp = tp_candidates.sort_values("y_proba", ascending=False).head(1).index[0]

fp_candidates = local_summary[
    (local_summary["y_true"] == 0) & (local_summary["y_pred"] == 1)
]
case_fp = fp_candidates.sort_values("y_proba", ascending=False).head(1).index[0]

print("Caso TP seleccionado:", case_tp, type(case_tp))
print("Caso FP seleccionado:", case_fp, type(case_fp))
```

    Caso TP seleccionado: 3380 <class 'numpy.int64'>
    Caso FP seleccionado: 3346 <class 'numpy.int64'>
    


```python
def explain_case(case_id, top_n=8):
    # Si viene como Index o lista de un solo elemento, extraer el valor
    if hasattr(case_id, "__len__") and not isinstance(case_id, (str, bytes)):
        if not isinstance(case_id, (int, float)):
            try:
                case_id = list(case_id)[0]
            except Exception:
                pass

    print("\n==============================")
    print(f"Explicación local del caso {case_id}")
    print("==============================")

    print("\nDatos originales del cliente:")
    display(X_test.loc[[case_id]])

    print("\nResultado del modelo:")
    print(local_summary.loc[case_id])

    case_contrib = contrib_df.loc[case_id].sort_values()

    print("\nTop variables que reducen churn en este caso:")
    display(
        case_contrib.head(top_n)
        .reset_index()
        .rename(columns={"index": "feature", case_id: "contribution"})
    )

    print("\nTop variables que aumentan churn en este caso:")
    display(
        case_contrib.tail(top_n)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", case_id: "contribution"})
    )

    print("\nChequeo de consistencia:")
    print("Log-odds:", round(log_odds.loc[case_id], 4))
    print("Probabilidad calculada desde contribuciones:", round(proba_from_contrib.loc[case_id], 4))
    print("Probabilidad del pipeline:", round(local_summary.loc[case_id, "y_proba"], 4))
```


```python
explain_case(case_tp, top_n=8)
```

    
    ==============================
    Explicación local del caso 3380
    ==============================
    
    Datos originales del cliente:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3380</th>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>95.1</td>
      <td>95.1</td>
    </tr>
  </tbody>
</table>
</div>


    
    Resultado del modelo:
    y_true     1.000000
    y_pred     1.000000
    y_proba    0.855041
    Name: 3380, dtype: float64
    
    Top variables que reducen churn en este caso:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__MonthlyCharges</td>
      <td>-0.601858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>num__TotalCharges</td>
      <td>-0.515119</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__PhoneService_Yes</td>
      <td>-0.158065</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__Partner_Yes</td>
      <td>-0.133534</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__gender_Male</td>
      <td>-0.133340</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__SeniorCitizen_1</td>
      <td>-0.070924</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__Dependents_No</td>
      <td>-0.032387</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__DeviceProtection_No</td>
      <td>-0.026421</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top variables que aumentan churn en este caso:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__tenure</td>
      <td>1.608697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat__InternetService_Fiber optic</td>
      <td>0.648027</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__Contract_Month-to-month</td>
      <td>0.585998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__StreamingMovies_Yes</td>
      <td>0.212270</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__StreamingTV_Yes</td>
      <td>0.211922</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__PaymentMethod_Electronic check</td>
      <td>0.205585</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__OnlineSecurity_No</td>
      <td>0.165550</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__TechSupport_No</td>
      <td>0.141320</td>
    </tr>
  </tbody>
</table>
</div>


    
    Chequeo de consistencia:
    Log-odds: 1.7747
    Probabilidad calculada desde contribuciones: 0.855
    Probabilidad del pipeline: 0.855
    

**Lectura del True Positive (TP):**
Este caso ilustra a la perfección el perfil de fuga clásico. El modelo detecta un riesgo extremo impulsado por una combinación letal: baja antigüedad, contrato mensual, uso de fibra óptica y ausencia de servicios de soporte/seguridad. El modelo acierta de pleno.


```python
explain_case(case_fp, top_n=8)
```

    
    ==============================
    Explicación local del caso 3346
    ==============================
    
    Datos originales del cliente:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3346</th>
      <td>Female</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>84.05</td>
      <td>186.05</td>
    </tr>
  </tbody>
</table>
</div>


    
    Resultado del modelo:
    y_true     0.000000
    y_pred     1.000000
    y_proba    0.817939
    Name: 3346, dtype: float64
    
    Top variables que reducen churn en este caso:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__TotalCharges</td>
      <td>-0.493864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>num__MonthlyCharges</td>
      <td>-0.381423</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__StreamingTV_No</td>
      <td>-0.207537</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__PhoneService_Yes</td>
      <td>-0.158065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__gender_Female</td>
      <td>-0.155230</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__Partner_No</td>
      <td>-0.155036</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__SeniorCitizen_1</td>
      <td>-0.070924</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__Dependents_No</td>
      <td>-0.032387</td>
    </tr>
  </tbody>
</table>
</div>


    
    Top variables que aumentan churn en este caso:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__tenure</td>
      <td>1.557603</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat__InternetService_Fiber optic</td>
      <td>0.648027</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__Contract_Month-to-month</td>
      <td>0.585998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__StreamingMovies_Yes</td>
      <td>0.212270</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__PaymentMethod_Electronic check</td>
      <td>0.205585</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__OnlineSecurity_No</td>
      <td>0.165550</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__TechSupport_No</td>
      <td>0.141320</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__MultipleLines_Yes</td>
      <td>0.113364</td>
    </tr>
  </tbody>
</table>
</div>


    
    Chequeo de consistencia:
    Log-odds: 1.5024
    Probabilidad calculada desde contribuciones: 0.8179
    Probabilidad del pipeline: 0.8179
    

**Lectura del False Positive (FP):**
Este cliente tiene un perfil de altísimo riesgo (matemáticamente casi idéntico a un *churner*), pero en la realidad decidió quedarse. Esto es crucial: el modelo no emite certezas absolutas, sino que estima probabilidades basadas en perfiles de riesgo histórico.

### Interpretación local

La explicabilidad local muestra que el modelo no asigna riesgo por una sola variable, sino por la combinación de múltiples contribuciones. Los casos analizados confirman que la baja antigüedad, el contrato mensual, la fibra y la ausencia de soporte o seguridad empujan fuertemente el riesgo hacia arriba. También ilustran que un cliente puede presentar un perfil claramente frágil y aun así no abandonar, lo que refuerza la idea de que el modelo estima probabilidades, no certezas absolutas.

## BLOQUE C — Explicabilidad Agrupada
Analizamos el comportamiento colectivo del 10% de clientes con mayor probabilidad de fuga de nuestra cartera.


```python
top10_profile_df, top10_protective_df = build_grouped_explainability_tables(k=0.10)

top10_profile_df
top10_protective_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mean_contrib_top10</th>
      <th>mean_contrib_rest</th>
      <th>diff_top10_vs_rest</th>
      <th>group_effect</th>
      <th>topk_fraction</th>
      <th>topk_size</th>
      <th>min_probability_in_topk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__TotalCharges</td>
      <td>-0.399146</td>
      <td>0.018613</td>
      <td>-0.417759</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>num__MonthlyCharges</td>
      <td>-0.346100</td>
      <td>0.056813</td>
      <td>-0.402913</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__Partner_No</td>
      <td>-0.117384</td>
      <td>-0.076968</td>
      <td>-0.040416</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__StreamingTV_No</td>
      <td>-0.120075</td>
      <td>-0.082263</td>
      <td>-0.037813</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__StreamingMovies_No</td>
      <td>-0.100973</td>
      <td>-0.081909</td>
      <td>-0.019064</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__SeniorCitizen_1</td>
      <td>-0.026343</td>
      <td>-0.009501</td>
      <td>-0.016842</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__DeviceProtection_No</td>
      <td>-0.020948</td>
      <td>-0.010660</td>
      <td>-0.010288</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__Dependents_No</td>
      <td>-0.030768</td>
      <td>-0.021566</td>
      <td>-0.009202</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat__PhoneService_Yes</td>
      <td>-0.151291</td>
      <td>-0.143491</td>
      <td>-0.007799</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cat__DeviceProtection_Yes</td>
      <td>0.006381</td>
      <td>0.010803</td>
      <td>-0.004422</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cat__gender_Male</td>
      <td>-0.071432</td>
      <td>-0.067983</td>
      <td>-0.003449</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cat__gender_Female</td>
      <td>-0.072071</td>
      <td>-0.076086</td>
      <td>0.004015</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Variables que caracterizan al Top 10% de riesgo:")
top10_profile_df
```

    Variables que caracterizan al Top 10% de riesgo:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mean_contrib_top10</th>
      <th>mean_contrib_rest</th>
      <th>diff_top10_vs_rest</th>
      <th>group_effect</th>
      <th>topk_fraction</th>
      <th>topk_size</th>
      <th>min_probability_in_topk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__tenure</td>
      <td>1.320016</td>
      <td>-0.113317</td>
      <td>1.433333</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cat__InternetService_Fiber optic</td>
      <td>0.620255</td>
      <td>0.244606</td>
      <td>0.375649</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__Contract_Month-to-month</td>
      <td>0.585998</td>
      <td>0.292306</td>
      <td>0.293692</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__InternetService_DSL</td>
      <td>-0.027585</td>
      <td>-0.242444</td>
      <td>0.214859</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__Contract_Two year</td>
      <td>0.000000</td>
      <td>-0.201071</td>
      <td>0.201071</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__PaperlessBilling_No</td>
      <td>-0.016537</td>
      <td>-0.146472</td>
      <td>0.129935</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__PaymentMethod_Electronic check</td>
      <td>0.170342</td>
      <td>0.057998</td>
      <td>0.112344</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__OnlineSecurity_No</td>
      <td>0.164367</td>
      <td>0.073317</td>
      <td>0.091050</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat__TechSupport_No</td>
      <td>0.137282</td>
      <td>0.063031</td>
      <td>0.074250</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cat__Dependents_Yes</td>
      <td>-0.012809</td>
      <td>-0.085596</td>
      <td>0.072787</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cat__StreamingMovies_No internet service</td>
      <td>0.000000</td>
      <td>-0.072027</td>
      <td>0.072027</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cat__StreamingTV_No internet service</td>
      <td>0.000000</td>
      <td>-0.072027</td>
      <td>0.072027</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Variables que más alejan del Top 10% de riesgo:")
top10_protective_df
```

    Variables que más alejan del Top 10% de riesgo:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mean_contrib_top10</th>
      <th>mean_contrib_rest</th>
      <th>diff_top10_vs_rest</th>
      <th>group_effect</th>
      <th>topk_fraction</th>
      <th>topk_size</th>
      <th>min_probability_in_topk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>num__TotalCharges</td>
      <td>-0.399146</td>
      <td>0.018613</td>
      <td>-0.417759</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>num__MonthlyCharges</td>
      <td>-0.346100</td>
      <td>0.056813</td>
      <td>-0.402913</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cat__Partner_No</td>
      <td>-0.117384</td>
      <td>-0.076968</td>
      <td>-0.040416</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat__StreamingTV_No</td>
      <td>-0.120075</td>
      <td>-0.082263</td>
      <td>-0.037813</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cat__StreamingMovies_No</td>
      <td>-0.100973</td>
      <td>-0.081909</td>
      <td>-0.019064</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cat__SeniorCitizen_1</td>
      <td>-0.026343</td>
      <td>-0.009501</td>
      <td>-0.016842</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat__DeviceProtection_No</td>
      <td>-0.020948</td>
      <td>-0.010660</td>
      <td>-0.010288</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat__Dependents_No</td>
      <td>-0.030768</td>
      <td>-0.021566</td>
      <td>-0.009202</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat__PhoneService_Yes</td>
      <td>-0.151291</td>
      <td>-0.143491</td>
      <td>-0.007799</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cat__DeviceProtection_Yes</td>
      <td>0.006381</td>
      <td>0.010803</td>
      <td>-0.004422</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cat__gender_Male</td>
      <td>-0.071432</td>
      <td>-0.067983</td>
      <td>-0.003449</td>
      <td>Aleja del Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cat__gender_Female</td>
      <td>-0.072071</td>
      <td>-0.076086</td>
      <td>0.004015</td>
      <td>Caracteriza Top10 riesgo</td>
      <td>0.1</td>
      <td>140</td>
      <td>0.6615</td>
    </tr>
  </tbody>
</table>
</div>



### Interpretación agrupada

El top 10% de clientes con mayor riesgo estimado se caracteriza principalmente por:

- baja antigüedad (`tenure`),
- contrato mensual,
- `Fiber optic`,
- `Electronic check`,
- ausencia de `OnlineSecurity`,
- ausencia de `TechSupport`.

Esto conecta directamente la explicabilidad con el uso operativo del modelo, ya que permite describir el perfil del colectivo prioritario para campañas de retención.

## Conclusión

La explicabilidad del modelo seleccionado resulta coherente en sus tres niveles:

- **global**: identifica variables clave con sentido de negocio,
- **local**: explica predicciones concretas mediante contribuciones individuales,
- **agrupada**: describe el perfil colectivo del top 10% de mayor riesgo.

En conjunto, esta sección refuerza la elección de la regresión logística como modelo principal, ya que combina rendimiento sólido con una capacidad de interpretación muy superior a la de alternativas más complejas.

El siguiente paso será traducir el modelo a una lógica de **impacto operativo y de negocio**, analizando thresholds y escenarios económicos.
