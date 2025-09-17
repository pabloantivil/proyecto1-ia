import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc,confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import cv2
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)

# "C:/Users/pablo/OneDrive/Documentos/UCT/GitHub/proyecto1-ia/dataset"
# ruta = "C:/Users/benja/Desktop/a/proyecto1-ia/dataset"
ruta = "C:/Users/pablo/OneDrive/Documentos/UCT/GitHub/proyecto1-ia/dataset"

# Lista de imágenes excluyendo las que son máscaras
imagen_path = sorted([p for p in glob.glob(ruta + "/*.jpg") if "_expert" not in p])

# Partición de datos por imagen
entrenamiento_val, test = train_test_split(imagen_path, test_size=0.2, random_state=seed)
entrenamiento, val = train_test_split(entrenamiento_val, test_size=0.25, random_state=seed)
print("Entrenamiento:", len(entrenamiento), "Validación:", len(val), "Test:", len(test))

def carga_imagen(paths):
    imagenes, mascaras = [], []
    for imagen_path in paths:
        img = cv2.cvtColor(cv2.imread(imagen_path), cv2.COLOR_BGR2RGB)
        mask_path = imagen_path.replace('.jpg', '_expert.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Advertencia: No se encontró la máscara para {mask_path}")
            continue
        mask = (mask > 0).astype(np.uint8)  # Binaria
        imagenes.append(img)
        mascaras.append(mask)
    return imagenes, mascaras

train_images, train_masks = carga_imagen(entrenamiento)
val_images, val_masks = carga_imagen(val)
test_images, test_masks = carga_imagen(test)

def normalize(images):
    return [img.astype(np.float32) / 255.0 for img in images]  # Conversión explícita

train_images = normalize(train_images)
val_images = normalize(val_images)
test_images = normalize(test_images)

# Corrección de iluminación (opcional)
def corregir_iluminacion(img):
    # Convertir a uint8 para CLAHE si está normalizada
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0  # Normalizar de vuelta

# Muestreo equilibrado CORREGIDO
def muestreo_equilibrado(imagenes, mascaras, n=10000):
    lesion_pixels = []
    no_lesion_pixels = []
    
    for img, mask in zip(imagenes, mascaras):
        # Extraer coordenadas donde mask == 1 y mask == 0
        lesion_coords = np.where(mask == 1)
        no_lesion_coords = np.where(mask == 0)
        
        # Extraer píxeles RGB usando las coordenadas
        lesion_rgb = img[lesion_coords]  # Shape: (n_lesion_pixels, 3)
        no_lesion_rgb = img[no_lesion_coords]  # Shape: (n_no_lesion_pixels, 3)
        
        lesion_pixels.extend(lesion_rgb)
        no_lesion_pixels.extend(no_lesion_rgb)
    
    # Convertir a arrays numpy
    lesion_pixels = np.array(lesion_pixels)
    no_lesion_pixels = np.array(no_lesion_pixels)
    
    # Muestreo aleatorio equilibrado
    n_per_class = min(n//2, len(lesion_pixels), len(no_lesion_pixels))
    
    lesion_indices = np.random.choice(len(lesion_pixels), n_per_class, replace=False)
    no_lesion_indices = np.random.choice(len(no_lesion_pixels), n_per_class, replace=False)
    
    # Crear dataset final
    X = np.vstack([lesion_pixels[lesion_indices], no_lesion_pixels[no_lesion_indices]])
    y = np.hstack([np.ones(n_per_class), np.zeros(n_per_class)])
    
    # Mezclar los datos
    shuffle_indices = np.random.permutation(len(X))
    return X[shuffle_indices], y[shuffle_indices]

# Procesar datos de entrenamiento, validación y test
X_entrenamiento, y_entrenamiento = muestreo_equilibrado(train_images, train_masks, n=10000)
X_validacion, y_validacion = muestreo_equilibrado(val_images, val_masks, n=5000)

print("✓ Datos cargados, particionados y preprocesados correctamente")
print(f"Entrenamiento: {X_entrenamiento.shape[0]} píxeles")
print(f"Validación: {X_validacion.shape[0]} píxeles")
print(f"Test: {len(test_images)} imágenes")
print(f"Distribución entrenamiento - Lesión: {np.sum(y_entrenamiento)}, No-lesión: {np.sum(y_entrenamiento == 0)}") 

# ================================
# HISTOGRAMAS Y ESTADÍSTICOS RGB
# ================================

def analizar_canales_rgb(imagenes, mascaras):
    """Extrae píxeles de lesión y no-lesión para análisis"""
    lesion_pixels = []
    no_lesion_pixels = []
    
    for img, mask in zip(imagenes, mascaras):
        lesion_coords = np.where(mask == 1)
        no_lesion_coords = np.where(mask == 0)
        
        lesion_rgb = img[lesion_coords]
        no_lesion_rgb = img[no_lesion_coords]
        
        lesion_pixels.extend(lesion_rgb)
        no_lesion_pixels.extend(no_lesion_rgb)
    
    return np.array(lesion_pixels), np.array(no_lesion_pixels)

# Extraer píxeles de entrenamiento
lesion_pixels, no_lesion_pixels = analizar_canales_rgb(train_images, train_masks)

print(f"\nPíxeles extraídos:")
print(f"Lesión: {len(lesion_pixels):,}")
print(f"No-lesión: {len(no_lesion_pixels):,}")

# Estadísticos por canal
canales = ['R', 'G', 'B']
print(f"\nEstadísticos por canal:")
print("-" * 60)

for i, canal in enumerate(canales):
    lesion_canal = lesion_pixels[:, i]
    no_lesion_canal = no_lesion_pixels[:, i]
    
    print(f"\nCanal {canal}:")
    print(f"  Lesión    - Media: {np.mean(lesion_canal):.4f}, Std: {np.std(lesion_canal):.4f}")
    print(f"  No-lesión - Media: {np.mean(no_lesion_canal):.4f}, Std: {np.std(no_lesion_canal):.4f}")

# Histogramas
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['red', 'green', 'blue']

for i, (canal, color) in enumerate(zip(canales, colors)):
    lesion_canal = lesion_pixels[:, i]
    no_lesion_canal = no_lesion_pixels[:, i]
    
    axes[i].hist(no_lesion_canal, bins=50, alpha=0.7, label='No-lesión', color='lightblue', density=True)
    axes[i].hist(lesion_canal, bins=50, alpha=0.7, label='Lesión', color=color, density=True)
    axes[i].set_title(f'Histograma Canal {canal}')
    axes[i].set_xlabel(f'Intensidad {canal}')
    axes[i].set_ylabel('Densidad')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Análisis de histogramas y estadísticos RGB completado")

# ================================
# 3.2 CLASIFICADOR BAYESIANO (RGB)
# ================================

print("\n" + "="*50)
print("3.2 CLASIFICADOR BAYESIANO (RGB)")
print("="*50)

# Calcular parámetros de las distribuciones para cada clase
def calcular_parametros_clase(X, y):
    # Separar píxeles por clase
    lesion_pixels = X[y == 1]
    no_lesion_pixels = X[y == 0]
    
    # Calcular media y covarianza para cada clase
    media_lesion = np.mean(lesion_pixels, axis=0)
    cov_lesion = np.cov(lesion_pixels, rowvar=False)
    
    media_no_lesion = np.mean(no_lesion_pixels, axis=0)
    cov_no_lesion = np.cov(no_lesion_pixels, rowvar=False)
    
    return media_lesion, cov_lesion, media_no_lesion, cov_no_lesion

# Calcular parámetros usando el conjunto de entrenamiento
media_lesion, cov_lesion, media_no_lesion, cov_no_lesion = calcular_parametros_clase(X_entrenamiento, y_entrenamiento)

print("Parámetros calculados:")
print(f"Media lesión: {media_lesion}")
print(f"Media no-lesión: {media_no_lesion}")

# Crear distribuciones normales multivariadas
dist_lesion = multivariate_normal(mean=media_lesion, cov=cov_lesion, allow_singular=True)
dist_no_lesion = multivariate_normal(mean=media_no_lesion, cov=cov_no_lesion, allow_singular=True)

# Función para calcular la razón de verosimilitud
def razon_verosimilitud(X, dist_lesion, dist_no_lesion):
    # Calcular probabilidades para cada clase
    p_lesion = dist_lesion.pdf(X)
    p_no_lesion = dist_no_lesion.pdf(X)
    
    # Calcular razón de verosimilitud (evitar división por cero)
    epsilon = 1e-10
    lr = p_lesion / (p_no_lesion + epsilon)
    
    return lr

# Calcular razón de verosimilitud para validación
lr_validacion = razon_verosimilitud(X_validacion, dist_lesion, dist_no_lesion)

# Función para encontrar el mejor umbral usando el índice de Youden
def encontrar_mejor_umbral(lr, y_true):
    # Calcular curva ROC
    fpr, tpr, umbrales = roc_curve(y_true, lr)
    
    # Calcular índice de Youden (J = sensibilidad + especificidad - 1)
    j_scores = tpr + (1 - fpr) - 1
    
    # Encontrar el umbral que maximiza el índice de Youden
    mejor_idx = np.argmax(j_scores)
    mejor_umbral = umbrales[mejor_idx]
    mejor_j = j_scores[mejor_idx]
    
    return mejor_umbral, mejor_j, fpr, tpr, umbrales

# Encontrar el mejor umbral usando validación
mejor_umbral, mejor_j, fpr, tpr, umbrales = encontrar_mejor_umbral(lr_validacion, y_validacion)

print(f"\nMejor umbral (Youden): {mejor_umbral:.4f}")
print(f"Índice de Youden (J): {mejor_j:.4f}")

# Función para clasificar usando el umbral
def clasificar_bayes(X, dist_lesion, dist_no_lesion, umbral):
    lr = razon_verosimilitud(X, dist_lesion, dist_no_lesion)
    return (lr >= umbral).astype(int)

# Evaluar en validación con el mejor umbral
y_pred_validacion = clasificar_bayes(X_validacion, dist_lesion, dist_no_lesion, mejor_umbral)

# Calcular métricas
accuracy = accuracy_score(y_validacion, y_pred_validacion)
precision = precision_score(y_validacion, y_pred_validacion)
sensibilidad = recall_score(y_validacion, y_pred_validacion)  # TPR
especificidad = confusion_matrix(y_validacion, y_pred_validacion)[0, 0] / np.sum(y_validacion == 0)  # TNR

print("\nResultados en validación:")
print(f"Exactitud: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Sensibilidad: {sensibilidad:.4f}")
print(f"Especificidad: {especificidad:.4f}")

print("\n✓ Clasificador Bayesiano RGB implementado y evaluado")

# ================================
# 3.3 CLASIFICADOR BAYESIANO + PCA
# ================================

print("\n" + "="*50)
print("3.3 CLASIFICADOR BAYESIANO + PCA")
print("="*50)

# Aplicar PCA solo a los datos de entrenamiento (evitar leakage)
pca = PCA(random_state=seed)
pca.fit(X_entrenamiento)

# Calcular la varianza acumulada
varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

# Seleccionar número de componentes (justificación: ≥95% de varianza)
n_componentes = np.argmax(varianza_acumulada >= 0.95) + 1
print(f"Varianza explicada por componentes: {pca.explained_variance_ratio_}")
print(f"Varianza acumulada: {varianza_acumulada}")
print(f"Número de componentes seleccionados: {n_componentes} (explican {varianza_acumulada[n_componentes-1]:.3%} de varianza)")

# Reajustar PCA con el número seleccionado de componentes
pca = PCA(n_components=n_componentes, random_state=seed)
X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)
X_validacion_pca = pca.transform(X_validacion)

print(f"\nDimensión original: {X_entrenamiento.shape[1]}")
print(f"Dimensión después de PCA: {X_entrenamiento_pca.shape[1]}")

# Entrenar clasificador Bayesiano en el espacio PCA
media_lesion_pca, cov_lesion_pca, media_no_lesion_pca, cov_no_lesion_pca = calcular_parametros_clase(
    X_entrenamiento_pca, y_entrenamiento)

print("\nParámetros calculados (espacio PCA):")
print(f"Media lesión (PCA): {media_lesion_pca}")
print(f"Media no-lesión (PCA): {media_no_lesion_pca}")

# Crear distribuciones normales multivariadas en espacio PCA
dist_lesion_pca = multivariate_normal(mean=media_lesion_pca, cov=cov_lesion_pca, allow_singular=True)
dist_no_lesion_pca = multivariate_normal(mean=media_no_lesion_pca, cov=cov_no_lesion_pca, allow_singular=True)

# Calcular razón de verosimilitud para validación (en espacio PCA)
lr_validacion_pca = razon_verosimilitud(X_validacion_pca, dist_lesion_pca, dist_no_lesion_pca)

# Encontrar el mejor umbral usando el índice de Youden (en espacio PCA)
mejor_umbral_pca, mejor_j_pca, fpr_pca, tpr_pca, umbrales_pca = encontrar_mejor_umbral(lr_validacion_pca, y_validacion)

print(f"\nMejor umbral (Youden, PCA): {mejor_umbral_pca:.4f}")
print(f"Índice de Youden (J, PCA): {mejor_j_pca:.4f}")

# Evaluar en validación con el mejor umbral (PCA)
y_pred_validacion_pca = clasificar_bayes(X_validacion_pca, dist_lesion_pca, dist_no_lesion_pca, mejor_umbral_pca)

# Calcular métricas para PCA
accuracy_pca = accuracy_score(y_validacion, y_pred_validacion_pca)
precision_pca = precision_score(y_validacion, y_pred_validacion_pca)
sensibilidad_pca = recall_score(y_validacion, y_pred_validacion_pca)
tn, fp, fn, tp = confusion_matrix(y_validacion, y_pred_validacion_pca).ravel()
especificidad_pca = tn / (tn + fp)

print("\nResultados en validación (PCA):")
print(f"Exactitud: {accuracy_pca:.4f}")
print(f"Precisión: {precision_pca:.4f}")
print(f"Sensibilidad: {sensibilidad_pca:.4f}")
print(f"Especificidad: {especificidad_pca:.4f}")

print("\n✓ Clasificador Bayesiano + PCA implementado y evaluado")

# ================================
# 3.4 CURVAS ROC Y PUNTO DE OPERACIÓN
# ================================

print("\n" + "="*50)
print("3.4 CURVAS ROC Y PUNTO DE OPERACIÓN")
print("="*50)

# Calcular AUC para ambos clasificadores
auc_rgb = auc(fpr, tpr)
auc_pca = auc(fpr_pca, tpr_pca)

print(f"AUC Bayesiano RGB: {auc_rgb:.4f}")
print(f"AUC Bayesiano PCA: {auc_pca:.4f}")

# Encontrar los puntos de operación (Youden) en las curvas ROC
idx_optimo_rgb = np.argmax(tpr - fpr)  # Índice del punto óptimo para RGB
idx_optimo_pca = np.argmax(tpr_pca - fpr_pca)  # Índice del punto óptimo para PCA

# Crear figura para comparar ambas curvas ROC
plt.figure(figsize=(12, 8))

# Curva ROC para Bayesiano RGB
plt.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2,
         label=f'Bayesiano RGB (AUC = {auc_rgb:.3f})')

# Punto de operación para Bayesiano RGB
plt.scatter(fpr[idx_optimo_rgb], tpr[idx_optimo_rgb], color='blue', s=100,
            label=f'RGB - Youden (Umbral = {mejor_umbral:.3f})')

# Curva ROC para Bayesiano PCA
plt.plot(fpr_pca, tpr_pca, color='red', linestyle='-', linewidth=2,
         label=f'Bayesiano PCA (AUC = {auc_pca:.3f})')

# Punto de operación para Bayesiano PCA
plt.scatter(fpr_pca[idx_optimo_pca], tpr_pca[idx_optimo_pca], color='red', s=100,
            label=f'PCA - Youden (Umbral = {mejor_umbral_pca:.3f})')

# Línea de clasificador aleatorio
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
         label='Clasificador aleatorio (AUC = 0.500)')

# Configuración del gráfico
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
plt.title('Comparación de Curvas ROC - Índice de Youden', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Añadir texto con métricas de los puntos óptimos
plt.text(0.6, 0.25, f'Bayesiano RGB (Youden):\n'
         f'- Sensibilidad: {tpr[idx_optimo_rgb]:.3f}\n'
         f'- Especificidad: {1-fpr[idx_optimo_rgb]:.3f}\n'
         f'- Índice J: {mejor_j:.3f}',
         bbox=dict(facecolor='lightblue', alpha=0.7), fontsize=9)

plt.text(0.6, 0.05, f'Bayesiano PCA (Youden):\n'
         f'- Sensibilidad: {tpr_pca[idx_optimo_pca]:.3f}\n'
         f'- Especificidad: {1-fpr_pca[idx_optimo_pca]:.3f}\n'
         f'- Índice J: {mejor_j_pca:.3f}',
         bbox=dict(facecolor='lightcoral', alpha=0.7), fontsize=9)

plt.tight_layout()
plt.show()

# Justificación de la elección del criterio de Youden
print("\nJustificación del criterio de Youden:")
print("El índice de Youden (J = sensibilidad + especificidad - 1) fue seleccionado porque:")
print("1. Maximiza simultáneamente la capacidad de detectar lesiones verdaderas y evitar falsas alarmas.")
print("2. Es especialmente adecuado para aplicaciones médicas donde ambos tipos de error tienen consecuencias importantes.")
print("3. Proporciona un balance óptimo entre sensibilidad y especificidad sin priorizar una sobre la otra.")
print("4. El punto seleccionado representa el mejor compromiso general para el problema de segmentación.")

# Comparación del desempeño en los puntos de operación
print("\nComparación en puntos de operación (Youden):")
print("="*50)
print(f"{'Métrica':<15} {'Bayesiano RGB':<15} {'Bayesiano PCA':<15}")
print(f"{'AUC':<15} {auc_rgb:<15.4f} {auc_pca:<15.4f}")
print(f"{'Sensibilidad':<15} {tpr[idx_optimo_rgb]:<15.4f} {tpr_pca[idx_optimo_pca]:<15.4f}")
print(f"{'Especificidad':<15} {1-fpr[idx_optimo_rgb]:<15.4f} {1-fpr_pca[idx_optimo_pca]:<15.4f}")
print(f"{'Índice J':<15} {mejor_j:<15.4f} {mejor_j_pca:<15.4f}")

print("\n✓ Comparación de curvas ROC y puntos de operación completada")


# ================================
# 3.5 CLASIFICACIÓN NO SUPERVISADA: K-MEANS
# ================================
