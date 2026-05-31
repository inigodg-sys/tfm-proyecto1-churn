# 02 — Modeling and Evaluation

## Objetivo
Este notebook documenta la fase de modelado supervisado del proyecto. El propósito es:

1. preparar el dataset para entrenamiento y evaluación,
2. construir baselines de referencia,
3. entrenar distintos modelos de clasificación,
4. comparar su rendimiento con métricas técnicas y operativas,
5. y seleccionar el modelo principal del proyecto.

## Preguntas que responde
- ¿Qué baseline mínimo debemos superar?
- ¿Qué modelo ofrece el mejor equilibrio entre rendimiento e interpretabilidad?
- ¿Qué métrica es más útil para el caso de negocio?
- ¿Qué modelo debe seleccionarse como solución final?


```python
import pandas as pd
import numpy as np
import sys
import os

# Ajuste de path si es necesario para importar desde src/
sys.path.append(os.path.abspath(os.path.join('..')))

from src.data.load_data import load_telco_data
from src.data.clean_data import clean_telco_data
from src.features.feature_lists import TARGET
from src.models.baseline import majority_class_baseline, random_topk_baseline
from src.models.train_logistic import (
    build_logistic_pipeline,
    prepare_train_test_split,
    evaluate_model,
)
from src.models.compare_models import build_comparison_table
```


```python
X_train, X_test, y_train, y_test = prepare_train_test_split()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nProporción de churn en train:")
print(y_train.value_counts(normalize=True).round(4))

print("\nProporción de churn en test:")
print(y_test.value_counts(normalize=True).round(4))
```

    X_train: (5634, 19)
    X_test: (1409, 19)
    y_train: (5634,)
    y_test: (1409,)
    
    Proporción de churn en train:
    Churn
    0    0.7346
    1    0.2654
    Name: proportion, dtype: float64
    
    Proporción de churn en test:
    Churn
    0    0.7346
    1    0.2654
    Name: proportion, dtype: float64
    

### División de los Datos (Train-Test Split)

Antes de comenzar con el modelado, es fundamental separar nuestros datos para poder evaluar el rendimiento de los algoritmos sobre datos "no vistos". 

Hemos configurado la división con las siguientes características:
* **Proporción 80/20:** Utilizaremos el 80% de los datos para entrenar los modelos (Train) y reservaremos el 20% exclusivamente para la validación y evaluación final (Test).
* **Muestreo Estratificado:** Dado que nuestra variable objetivo presenta un desbalance moderado (73.5% retenidos vs 26.5% abandonos), una división puramente aleatoria podría alterar esta distribución por puro azar. Al aplicar un split *estratificado*, forzamos al algoritmo a mantener exactamente la misma proporción de la clase `Churn` tanto en el conjunto de entrenamiento como en el de prueba.

Esto garantiza que las métricas de evaluación que obtengamos más adelante sean representativas, fiables y fieles a la realidad del negocio.


```python
baseline_majority = majority_class_baseline(y_test)
baseline_random = random_topk_baseline(y_test, k=0.10, random_state=42)

print("Baseline mayoritario:")
print(baseline_majority)

print("\nBaseline aleatorio Top10%:")
print(baseline_random)
```

    Baseline mayoritario:
    {'model': 'baseline_majority', 'accuracy': 0.7345635202271115, 'precision': 0.0, 'recall': 0.0, 'confusion_matrix': array([[1035,    0],
           [ 374,    0]])}
    
    Baseline aleatorio Top10%:
    {'model': 'baseline_random_topk', 'k': 0.1, 'recall_at_top_k': 0.10962566844919786}
    

### Modelos de Referencia (Baselines)

Antes de entrenar algoritmos complejos, es obligatorio establecer modelos de referencia ("baselines") que nos sirvan como punto de comparación. Si nuestro modelo de Machine Learning no es capaz de batir estas métricas básicas, no aportará valor al negocio.

Hemos definido dos baselines fundamentales:

1. **Baseline Mayoritario :** Este modelo predice siempre la clase mayoritaria ("No Churn"). Dado nuestro desbalance, este modelo consigue una **accuracy artificialmente alta (~73.5%)**. Sin embargo, al no identificar jamás a un cliente en fuga, su **Recall es nulo (0.0)**. Esto demuestra que la métrica de accuracy es engañosa y totalmente inútil para el objetivo de nuestra campaña de retención.

2. **Baseline Aleatorio (El suelo de negocio):** Imaginemos un escenario operativo realista donde la empresa solo tiene presupuesto para contactar al 10% de su cartera de clientes. Si seleccionamos a ese 10% completamente al azar, capturaríamos aproximadamente al 10% de los verdaderos churners (`Recall@Top10% ≈ 0.10`). **Este es nuestro "suelo de negocio"**. Cualquier modelo predictivo que construyamos debe multiplicar significativamente esta métrica para justificar su puesta en producción.


```python
clf_log = build_logistic_pipeline()
clf_log.fit(X_train, y_train)

metrics_log = evaluate_model(clf_log, X_test, y_test, threshold=0.50)

# Convertir el diccionario a DataFrame para una visualización profesional
metrics_log_df = pd.DataFrame([metrics_log])

# Seleccionamos solo las métricas clave y redondeamos a 4 decimales
metrics_log_df[[
    "accuracy", "precision", "recall", "roc_auc", "pr_auc", "recall_at_top10"
]].round(4)
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
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>roc_auc</th>
      <th>pr_auc</th>
      <th>recall_at_top10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8055</td>
      <td>0.6572</td>
      <td>0.5588</td>
      <td>0.842</td>
      <td>0.6337</td>
      <td>0.2781</td>
    </tr>
  </tbody>
</table>
</div>



### Interpretación inicial del modelo base

La regresión logística muestra ya un rendimiento sólido como primer modelo real. Frente al baseline mayoritario (que tenía un recall nulo) y al aleatorio, este modelo demuestra tener una capacidad predictiva real. Captura una cantidad significativa de la señal de *churn* y nos sirve como una excelente marca a batir (benchmark) para evaluar si modelos de "caja negra" más complejos realmente logran aportar un valor extra justificable.


```python
comparison_df = build_comparison_table()
comparison_df
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
      <th>Modelo</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>ROC_AUC</th>
      <th>PR_AUC</th>
      <th>Recall@Top10%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Baseline mayoritario</td>
      <td>0.7346</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.1096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regresión logística</td>
      <td>0.8055</td>
      <td>0.6572</td>
      <td>0.5588</td>
      <td>0.842</td>
      <td>0.6337</td>
      <td>0.2781</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>0.7991</td>
      <td>0.6542</td>
      <td>0.5160</td>
      <td>0.836</td>
      <td>0.6393</td>
      <td>0.2888</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HistGradientBoosting</td>
      <td>0.7942</td>
      <td>0.6400</td>
      <td>0.5134</td>
      <td>0.832</td>
      <td>0.6392</td>
      <td>0.2754</td>
    </tr>
  </tbody>
</table>
</div>



## Comparación de modelos

La comparación muestra que la **regresión logística** ofrece el mejor equilibrio global entre rendimiento técnico e interpretabilidad. Aunque **Random Forest** mejora ligeramente `Recall@Top10%`, la mejora es pequeña y viene acompañada de un empeoramiento en `Recall`, `Accuracy` y `ROC-AUC`. Por su parte, **HistGradientBoosting** no aporta mejoras claras frente a las otras alternativas.

En consecuencia, la **regresión logística** se selecciona como **modelo principal** del proyecto, y **Random Forest** se mantiene como modelo comparativo secundario.

## Modelo seleccionado

Se selecciona como modelo principal la **regresión logística** porque:

1. obtiene el mejor rendimiento global,
2. mantiene una capacidad razonable de detección de churn,
3. es claramente interpretable,
4. y permite una mejor traducción a negocio y explicabilidad.

Este resultado sugiere que el problema presenta una estructura suficientemente clara como para que un modelo lineal bien especificado compita muy bien frente a alternativas no lineales más complejas.

## Conclusión

La fase de modelado confirma que el dataset contiene señal predictiva suficiente para construir un modelo útil y defendible. La **regresión logística** se selecciona como solución principal al combinar rendimiento sólido, estabilidad y alta interpretabilidad.

El siguiente paso será analizar con más detalle **cómo toma decisiones el modelo**, abordando la explicabilidad a nivel global, local y agrupado.
