<<<<<<< HEAD
# tfm-proyecto1-churn
TFM Churn telecom
=======
# Proyecto 1 — Predicción de Churn y Métricas de Negocio (Telco)

## 1. Contexto y objetivo de negocio
Las compañías de telecomunicaciones suelen tener limitación de presupuesto/capacidad para campañas de retención (llamadas, descuentos, mejoras de plan). Por tanto, no es viable actuar sobre toda la base de clientes: hay que **priorizar**.

**Objetivo de negocio**: reducir la tasa de abandono (**churn**) maximizando el impacto económico de las acciones de retención.

**Decisión operativa que habilita el modelo**: seleccionar el **top 10%** de clientes con mayor probabilidad de churn para ejecutar una campaña de retención.

## 2. Preguntas que debe responder el modelo
1) ¿Qué clientes tienen mayor probabilidad de churn en el próximo periodo?
2) ¿Qué variables explican ese riesgo (global y por cliente)?
3) Si actuamos sobre el top 10% de riesgo, ¿cuál es el **impacto económico esperado** frente a una selección aleatoria?

## 3. Dataset
Se utiliza el dataset público **“Telco Customer Churn”** (Kaggle), con un objetivo binario `Churn` (Yes/No) y variables de perfil, contrato, servicios y facturación.

> Nota: `customerID` es un identificador y no se utiliza como predictor. `TotalCharges` puede requerir conversión a numérico y tratamiento de valores vacíos (típicamente asociados a `tenure = 0`).

## 4. Definición de métricas: técnicas y de negocio

### 4.1 Métricas técnicas (calidad predictiva)
- **ROC-AUC**: capacidad global de ranking del modelo.
- **PR-AUC**: útil cuando el churn no es perfectamente balanceado.
- **Recall@Top10%**: porcentaje de churners reales capturados si solo actuamos sobre el 10% con mayor score (métrica muy conectada con la decisión operativa).

### 4.2 Métrica de negocio (impacto económico esperado)
Definimos:
- **V** = valor esperado de retener un cliente (p.ej., margen mensual × meses esperados)
- **C** = coste por acción de retención (p.ej., incentivo / llamada / descuento)
- **r** = tasa de éxito de retención *condicionada a que el cliente realmente iba a churnear*
- **N** = número de clientes contactados (top 10%)
- **TP** = churners reales dentro del top 10% (true positives en esa “lista priorizada”)

**Impacto económico esperado de una campaña sobre el top 10%**:

€ esperado = TP · r · V  −  N · C

**Baseline de negocio**: comparar este € esperado contra una estrategia aleatoria (seleccionar N clientes al azar). El objetivo es demostrar *lift* (mejora) en impacto económico gracias al modelo.

## 5. Enfoque metodológico (resumen)
1) EDA + limpieza + preparación (pipeline reproducible).
2) Entrenamiento, validación y comparación de modelos (baseline + modelos no lineales).
3) Selección de umbral/estrategia basada en **métrica de negocio** (no solo F1).
4) Explicabilidad (global + local) con SHAP/PDP y recomendaciones accionables.
5) Diseño conceptual de despliegue (batch scoring, monitorización, retraining, drift).

## 6. Resultados principales (placeholder)
> Se completará al finalizar el entrenamiento y evaluación:
- Tabla comparativa de modelos (ROC-AUC, PR-AUC, Recall@Top10%).
- Curvas y visualizaciones.
- Estimación de impacto económico (top 10% vs aleatorio).
- Insights de explicabilidad y acciones recomendadas.
>>>>>>> e1071e7 (Initial project structure for churn TFM)
