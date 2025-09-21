# proyecto1-ia

> Segmentación de lesiones en imágenes dermatoscópicas — comparación de métodos supervisados (Bayesiano, PCA) y no supervisados (K-Means).

---

## Tabla de contenidos

- [Descripción](#descripción)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Uso rápido](#uso-rápido)
- [Ejecución reproducible (headless)](#ejecución-reproducible-headless)
- [Resultados de ejemplo](#resultados-de-ejemplo)
- [Archivos generados](#archivos-generados)
- [Notas y recomendaciones](#notas-y-recomendaciones)
- [Contacto / Autor](#contacto--autor)

---

## Descripción

Este repositorio implementa y compara técnicas para la clasificación de píxeles y segmentación de lesiones en imágenes dermatoscópicas. Incluye:

- Un clasificador Bayesiano entrenado en canales RGB.
- Un clasificador Bayesiano aplicado sobre un espacio reducido por PCA.
- Un pipeline no supervisado con K-Means evaluado en diferentes espacios de color (RGB, HSV, LAB, YCrCb).

El objetivo es comparar desempeño (precisión, sensibilidad, especificidad, AUC, Jaccard) y producir visualizaciones para análisis y trazabilidad.

## Estructura del repositorio

- `dataset/` — Imágenes `.jpg` y máscaras `_expert.png` (formato ISIC en este proyecto).
- `src/` — Código fuente principal:
  - `main.py` — Orquesta el pipeline completo.
  - `data_loader.py` — Carga y prepara imágenes y máscaras (particionado y muestreo de píxeles).
  - `bayesian_classifiers.py` — Implementación de Bayesiano (RGB y PCA) y utilidades ROC.
  - `kmeans_clustering.py` — Pipeline K-Means y evaluación (Jaccard) en distintos espacios de color.
  - `evaluation_metrics.py` — Cálculos y visualizaciones de métricas y comparaciones finales.
  - `rgb_analysis.py` — Histogramas y análisis estadístico por canal.

## Requisitos

Crear un entorno virtual e instalar dependencias (si existe `requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate   # Git Bash / WSL en Windows
pip install -r requirements.txt
```

Dependencias principales: `numpy`, `opencv-python`, `scikit-learn`, `scipy`, `matplotlib`.

> Nota: si `requirements.txt` fue modificado o eliminado, instala manualmente las dependencias listadas arriba.

## Uso rápido

1. Coloca el dataset en `dataset/` (cada imagen `.jpg` debe tener su máscara `_expert.png`).
2. Desde la raíz del repositorio, ejecuta el análisis completo:

```bash
python src/main.py
```

Esto realizará: carga y preparación de datos, análisis RGB, entrenamiento de clasificadores bayesianos, evaluación con PCA, ejecución de K-Means y la comparación final.

> Si `src/data_loader.py` usa una ruta absoluta para el dataset, cámbiala a `"dataset"` o llama a la función con la ruta deseada.

## Ejecución reproducible (headless)

Para ejecutar sin abrir ventanas de Matplotlib (útil en servidores o integración continua):

```bash
export MPLBACKEND=Agg
C:/Python312/python.exe src/main.py
```

En PowerShell:

```powershell
$env:MPLBACKEND = 'Agg'
C:/Python312/python.exe src/main.py
```

## Resultados de ejemplo

Los resultados mostrados abajo provienen de una ejecución de referencia en este repositorio. Los números pueden variar según el particionado aleatorio y la versión de paquetes.

### Particionado (por imagen)

- Entrenamiento: 90
- Validación: 30
- Test: 30

### Clasificadores bayesianos (validación)

| Modelo | Exactitud | Precisión | Sensibilidad | Especificidad | AUC |
|---|---:|---:|---:|---:|---:|
| Bayesiano RGB | 0.8798 | 0.9123 | 0.8404 | 0.9192 | 0.9432 |
| Bayesiano + PCA | 0.8742 | 0.8896 | 0.8544 | 0.8940 | 0.9365 |

### K-Means (validación, Jaccard por espacio de color)

| Espacio | Jaccard (media ± std) |
|---|---:|
| RGB | 0.4182 ± 0.2164 |
| HSV | 0.3408 ± 0.2001 |
| LAB | 0.4142 ± 0.2130 |
| YCrCb | 0.4159 ± 0.2139 |

### Comparación final en test (métricas resumidas)

| Métrica | Bayesiano-RGB | Bayesiano-PCA | K-Means |
|---|---:|---:|---:|
| Exactitud | 0.8427 | 0.8346 | 0.7251 |
| Precisión | 0.6246 | 0.6070 | 0.4507 |
| Sensibilidad | 0.8613 | 0.8785 | 0.6730 |
| Especificidad | 0.8369 | 0.8207 | 0.7415 |
| F1 | 0.7241 | 0.7179 | 0.5398 |
| Jaccard | 0.5676 | 0.5600 | 0.3697 |

> Observación: en este experimento los métodos supervisados (Bayesiano RGB / PCA) superan al método no supervisado (K-Means) en Jaccard y varias métricas; el mejor según Jaccard fue Bayesiano-RGB.

## Archivos generados (ejemplos)

- `roc_curves.png` — Curvas ROC comparativas.
- `kmeans_results.png` — Visualizaciones de resultados K-Means.
- `kmeans_grafic.png`, `comparison_final.png`, `comparative_final.png` — figuras con comparaciones y resúmenes.

Revisa el directorio raíz y `src/` para ubicar las imágenes generadas tras la ejecución.

## Notas y recomendaciones

- Si `src/data_loader.py` usa una ruta absoluta para el dataset, cámbiala a `"dataset"` para portabilidad.
- K-Means en este proyecto incluye heurísticas para asignar clusters a la clase lesión/no-lesión (intensidad + ruido). Revisa `src/kmeans_clustering.py` si necesitas ajustar criterios o eliminar la inyección de ruido para análisis más determinista.
- Los resultados pueden variar por la semilla aleatoria y la selección de píxeles al muestrear; para reproducibilidad fija la semilla donde corresponda.



