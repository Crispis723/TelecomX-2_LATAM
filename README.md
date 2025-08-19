# TelecomX – Predicción de Cancelación de Clientes

Proyecto de **Data Science** para predecir la cancelación (churn) de clientes de una telco usando modelos de *Machine Learning* y un pipeline reproducible en Python / Colab.

---

## 🧭 Objetivo

1. **Explorar** los datos y entender los principales factores asociados a la cancelación.
2. **Preparar** un pipeline robusto (limpieza → encoding → balanceo → split → escalado).
3. **Entrenar y evaluar** varios modelos (Logística, Random Forest, KNN y SVM).
4. **Explicar** la relevancia de las variables (coeficientes, importancias y permutation importance).

---

## 📦 Dataset

* Fuente: `TelecomX (1).csv` (repositorio original de Crispis723).
* Variable objetivo: `Cancelacion` (0 = activo, 1 = canceló).
* Variables numéricas: `Antiguedad`, `Cargos_Mensuales`, `Cargos_Totales`, `Cargo_Diario`, etc.
* Variables categóricas: **servicios**, **tipo de contrato**, **método de pago**, **género**, etc.
* Identificador único: `customerID` (**se elimina**).

> ⚠️ **Importante:** eliminar `customerID` **antes** de hacer `get_dummies` para no crear miles de columnas inútiles.

---

## 🧰 Requisitos

```bash
python >= 3.9
pip install -r requirements.txt
```

`requirements.txt` sugerido:

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

---

## 🗂️ Estructura recomendada del repo

```
.
├── data/
│   └── TelecomX (1).csv
├── notebooks/
│   └── Copia_de_TelecomX2_LATAM.ipynb
├── src/
│   └── telecomx_pipeline.py   # (opcional) versión script
├── outputs/
│   ├── figs/                  # gráficos exportados
│   └── models/                # modelos serializados (opcional)
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar

### Opción A — Google Colab (recomendada)

1. Abre el notebook `notebooks/Copia_de_TelecomX2_LATAM.ipynb` en Colab.
2. Ejecuta las celdas en orden (ya incluye carga desde GitHub).

### Opción B — Local

```bash
git clone <tu-repo>
cd <tu-repo>
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Copia_de_TelecomX2_LATAM.ipynb
```

---

## 🧪 Pipeline (resumen)

1. **Carga** y vista general (`df.info()`, nulos, duplicados).
2. **Limpieza**

   * Drop: `customerID`.
   * Conversión booleana → numérica: `{True/False/"true"/"false"} → {1/0}`.
3. **Encoding**

   * `get_dummies` sobre columnas `object` con `drop_first=True`.
4. **Split**

   * `train_test_split` con `stratify=y`, test=0.30.
   * Alinear columnas: `X_test = X_test.reindex(columns=X_train.columns, fill_value=0)`.
5. **Balanceo (solo en train)**

   * SMOTE por defecto; opción de `RandomUnderSampler`.
6. **Escalado (solo modelos sensibles a escala)**

   * `StandardScaler` **fit** en `X_train_res` y **transform** en `X_train_res` y `X_test`.
7. **Modelado**

   * Logística (escalada).
   * Random Forest (sin escalar).
   * KNN (escalado).
   * SVM lineal (escalado).
8. **Optimización**

   * `GridSearchCV` para Random Forest (y opcional para Logística/SVM).
9. **Evaluación**

   * Accuracy, Precision, Recall, F1 y matriz de confusión.
10. **Explicabilidad**

    * Logística/SVM: coeficientes.
    * Random Forest: `feature_importances_`.
    * KNN: `permutation_importance`.
    * **Tabla comparativa normalizada** (promedio entre modelos).

---

## 📈 Gráficos clave

* **EDA**

  * Heatmap de correlación (columna `Cancelacion`).
  * Boxplots: `Antiguedad × Cancelacion`, `Cargos_Totales × Cancelacion`.
  * Scatter: `Antiguedad` vs `Cargos_Totales` (hue = `Cancelacion`).

* **Modelos**

  * Matrices de confusión (Logística / RF / RF Optimizado).
  * Importancia de variables:

    * Logística: barras de coeficientes.
    * SVM lineal: barras de coeficientes.
    * Random Forest: barras `feature_importances_`.
    * KNN: *Permutation Importance*.
  * **Resumen final**: barras con el **promedio normalizado de importancia** entre modelos.

> Tip visual: usa 12–15 variables top por gráfico para evitar ruido.

---

## ✅ Buenas prácticas y “gotchas”

* **No SMOTE en test** (solo en entrenamiento).
* **Escala** únicamente para Logística/KNN/SVM. Los árboles no lo requieren.
* **Columnas consistentes** entre train y test (`reindex`).
* **Booleans a int** tras `get_dummies`:

  ```python
  df_encoded = df_encoded.replace({True: 1, False: 0, "true": 1, "false": 0})
  ```
* **Métricas balanceadas**: si el desbalance es alto, prioriza **Recall** y **F1** además de Accuracy.
* **Umbral de decisión**: considera ajustar el umbral (0.5 → 0.4/0.6) según el costo de falsos negativos/positivos.

---

## 🧠 Hallazgos (resumen interpretativo)

* **Costos** (`Cargos_Mensuales` y `Cargo_Diario`) son los predictores más fuertes de cancelación.
* **Antigüedad** reduce la probabilidad de cancelar (fuerte en modelos de árboles).
* **Contratos de 1 y 2 años** están asociados a menor churn vs. contratos mensuales.
* **Servicio de Internet**: *Fiber optic* suele asociarse a mayor probabilidad de churn (mayores expectativas); *No internet* tiende a menor churn.
* **Servicios adicionales** (Seguridad/Soporte/Respaldo) y **métodos de pago** aportan información secundaria útil.

> Estos patrones son consistentes entre modelos (Logística, RF, KNN y SVM), lo que refuerza su validez.

---

## 📊 Cómo reproducir las tablas/gráficos de importancia

En el notebook ya están los bloques para:

* Coeficientes (Logística/SVM).
* `feature_importances_` (RF).
* `permutation_importance` (KNN).
* **DataFrame comparativo normalizado** y **ranking global** por promedio.

---

## 🔮 Próximos pasos

* Calibración de probabilidades (Platt / Isotónica) para mejores umbrales.
* Curvas **ROC/PR** y **AUC** para comparar modelos.
* **XGBoost / LightGBM** y *RandomizedSearchCV* para mejorar tiempo de búsqueda.
* *SHAP values* para explicabilidad local y auditoría de sesgos.
* Segmentación de clientes (clustering) para estrategias de retención diferenciadas.

---


Si quieres, puedo convertir este README en **versión inglesa** y generar un **badge set** (build, license, python version) y un **Makefile** con comandos (`make setup`, `make train`, `make report`).
