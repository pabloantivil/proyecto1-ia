import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc,confusion_matrix, classification_report, ConfusionMatrixDisplay
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
# MODELO BAYESIANO MULTIVARIADO
# ================================

# Calcular medias y covarianzas de cada clase
mu_lesion = np.mean(lesion_pixels, axis=0)
cov_lesion = np.cov(lesion_pixels, rowvar=False)

mu_no_lesion = np.mean(no_lesion_pixels, axis=0)
cov_no_lesion = np.cov(no_lesion_pixels, rowvar=False)

print("\nParámetros estimados (Bayesiano):")
print("Lesión -> media:", mu_lesion, "\nCovarianza:\n", cov_lesion)
print("No-lesión -> media:", mu_no_lesion, "\nCovarianza:\n", cov_no_lesion)

# Definir distribuciones gaussianas
dist_lesion = multivariate_normal(mean=mu_lesion, cov=cov_lesion)
dist_no_lesion = multivariate_normal(mean=mu_no_lesion, cov=cov_no_lesion)

# ================================
# CLASIFICADOR BAYESIANO
# ================================
# Umbral = 1.0 (criterio: equiprobabilidad)
# Justificación: Cuando p(lesión|RGB) = p(no-lesión|RGB), la razón = 1.0
# Esto asume que ambas clases tienen igual probabilidad a priori
def clasificar_bayes(X, umbral=1.0):
    """Clasifica píxeles RGB usando razón de verosimilitud"""
    p_lesion = dist_lesion.pdf(X)
    p_no_lesion = dist_no_lesion.pdf(X)

    # Razón de verosimilitudes
    razon = p_lesion / (p_no_lesion + 1e-12)  # evitar división por 0

    # Decisión
    return (razon > umbral).astype(int)

# ================================
# EVALUACIÓN EN VALIDACIÓN
# ================================

# Clasificar datos de validación (reutilizando densidades)
# Calcular densidades una vez y reutilizar
p_lesion_val = dist_lesion.pdf(X_validacion)
p_no_lesion_val = dist_no_lesion.pdf(X_validacion)
razon_val = p_lesion_val / (p_no_lesion_val + 1e-12)
# Decisión con umbral = 1.0
y_pred = (razon_val > 1.0).astype(int)

# Matriz de confusión
matrix_confusion= confusion_matrix(y_validacion, y_pred)
vis = ConfusionMatrixDisplay(matrix_confusion, display_labels=["No-lesión", "Lesión"])
vis.plot()
plt.title("Matriz de Confusión")
plt.show()


# Evaluación
print("\nResultados en VALIDACIÓN:")
print("Accuracy:", accuracy_score(y_validacion, y_pred))
print("Precision:", precision_score(y_validacion, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_validacion, y_pred, target_names=["No-lesión", "Lesión"]))


# ================================
# CLASIFICADOR BAYESIANO + PCA
# ================================

# Aplicar PCA solo a los datos de entrenamiento (evitar leakage)
print("\nAplicando PCA a los datos...")
pca = PCA()
X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)

# Calcular varianza acumulada
varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

# Seleccionar número de componentes que explican al menos el 95% de varianza
n_componentes = np.argmax(varianza_acumulada >= 0.95) + 1
print(f"Número de componentes seleccionados: {n_componentes}")
print(f"Varianza explicada: {varianza_acumulada[n_componentes-1]:.4f}")

# Justificación de la selección
print("\nJustificación: Se seleccionaron", n_componentes, 
      "componentes principales que explican el",
      f"{varianza_acumulada[n_componentes-1]*100:.2f}% de la varianza.")
print("Esto permite reducir la dimensionalidad manteniendo la mayor parte de la información.")

# Reentrenar PCA con el número de componentes seleccionado
pca = PCA(n_components=n_componentes)
X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)
X_validacion_pca = pca.transform(X_validacion)

# Separar píxeles por clase en el espacio PCA
lesion_pixels_pca = X_entrenamiento_pca[y_entrenamiento == 1]
no_lesion_pixels_pca = X_entrenamiento_pca[y_entrenamiento == 0]

# Calcular medias y covarianzas de cada clase en el espacio PCA
mu_lesion_pca = np.mean(lesion_pixels_pca, axis=0)
cov_lesion_pca = np.cov(lesion_pixels_pca, rowvar=False)

mu_no_lesion_pca = np.mean(no_lesion_pixels_pca, axis=0)
cov_no_lesion_pca = np.cov(no_lesion_pixels_pca, rowvar=False)

print("\nParámetros estimados (Bayesiano + PCA):")
print("Lesión -> media:", mu_lesion_pca, "\nCovarianza:\n", cov_lesion_pca)
print("No-lesión -> media:", mu_no_lesion_pca, "\nCovarianza:\n", cov_no_lesion_pca)

# Definir distribuciones gaussianas en espacio PCA
dist_lesion_pca = multivariate_normal(mean=mu_lesion_pca, cov=cov_lesion_pca, allow_singular=True)
dist_no_lesion_pca = multivariate_normal(mean=mu_no_lesion_pca, cov=cov_no_lesion_pca, allow_singular=True)

# Clasificador bayesiano en espacio PCA
# Umbral = 1.0 (criterio: equiprobabilidad)
# Justificación: Cuando p(lesión|RGB) = p(no-lesión|RGB), la razón = 1.0
# Esto asume que ambas clases tienen igual probabilidad a priori
def clasificar_bayes_pca(X_pca, umbral=1.0):
    """Clasifica píxeles en espacio PCA usando razón de verosimilitud"""
    p_lesion = dist_lesion_pca.pdf(X_pca)
    p_no_lesion = dist_no_lesion_pca.pdf(X_pca)

    # Razón de verosimilitudes
    razon = p_lesion / (p_no_lesion + 1e-12)  # evitar división por 0

    # Decisión
    return (razon > umbral).astype(int)

# Evaluar clasificador PCA en validación (reutilizando densidades)
p_lesion_pca_val = dist_lesion_pca.pdf(X_validacion_pca)
p_no_lesion_pca_val = dist_no_lesion_pca.pdf(X_validacion_pca)
razon_pca = p_lesion_pca_val / (p_no_lesion_pca_val + 1e-12)
# Decisión con umbral = 1.0
y_pred_pca = (razon_pca > 1.0).astype(int)

# Matriz de confusión para PCA
matrix_confusion_pca = confusion_matrix(y_validacion, y_pred_pca)
vis_pca = ConfusionMatrixDisplay(matrix_confusion_pca, display_labels=["No-lesión", "Lesión"])
vis_pca.plot()
plt.title("Matriz de Confusión - Bayesiano + PCA")
plt.show()

# Evaluación
print("\nResultados en VALIDACIÓN (Bayesiano + PCA):")
print("Accuracy:", accuracy_score(y_validacion, y_pred_pca))
print("Precision:", precision_score(y_validacion, y_pred_pca))
print("\nReporte de clasificación:\n", classification_report(y_validacion, y_pred_pca, target_names=["No-lesión", "Lesión"]))

# Comparación con el clasificador sin PCA
print("\nCOMPARACIÓN CON CLASIFICADOR RGB COMPLETO:")
print("Accuracy RGB: {:.4f} vs Accuracy PCA: {:.4f}".format(
    accuracy_score(y_validacion, y_pred), 
    accuracy_score(y_validacion, y_pred_pca)))

# ================================
# CURVAS ROC Y PUNTO DE OPERACIÓN (Youden J óptimo)
# ================================

# Scores (razón de verosimilitud) para validación
scores_rgb = dist_lesion.pdf(X_validacion) / (dist_no_lesion.pdf(X_validacion) + 1e-12)
scores_pca = dist_lesion_pca.pdf(X_validacion_pca) / (dist_no_lesion_pca.pdf(X_validacion_pca) + 1e-12)

# ROC para RGB
fpr_r, tpr_r, thr_r = roc_curve(y_validacion, scores_rgb)
auc_r = auc(fpr_r, tpr_r)
youden_idx_r = np.argmax(tpr_r - fpr_r)
youden_thr_r = thr_r[youden_idx_r]
youden_tpr_r = tpr_r[youden_idx_r]
youden_fpr_r = fpr_r[youden_idx_r]

# ROC para PCA
fpr_p, tpr_p, thr_p = roc_curve(y_validacion, scores_pca)
auc_p = auc(fpr_p, tpr_p)
youden_idx_p = np.argmax(tpr_p - fpr_p)
youden_thr_p = thr_p[youden_idx_p]
youden_tpr_p = tpr_p[youden_idx_p]
youden_fpr_p = fpr_p[youden_idx_p]

# Plots
plt.figure(figsize=(8,6))
plt.plot(fpr_r, tpr_r, label=f'RGB (AUC={auc_r:.3f})', lw=2)
plt.plot(fpr_p, tpr_p, label=f'PCA (AUC={auc_p:.3f})', lw=2)
plt.scatter([youden_fpr_r], [youden_tpr_r], c='C0', s=60, marker='o', label=f'Youden RGB (thr={youden_thr_r:.3e})')
plt.scatter([youden_fpr_p], [youden_tpr_p], c='C1', s=60, marker='s', label=f'Youden PCA (thr={youden_thr_p:.3e})')
plt.plot([0,1],[0,1],'k--', alpha=0.5)
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curvas ROC - RGB vs PCA')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# Imprimir resumen de puntos de operación (Youden J)
print('\nPUNTO DE OPERACIÓN (Youden J óptimo):')
print(f'RGB - AUC: {auc_r:.4f}, Youden thr: {youden_thr_r:.6e}, TPR: {youden_tpr_r:.4f}, FPR: {youden_fpr_r:.4f}')
print(f'PCA - AUC: {auc_p:.4f}, Youden thr: {youden_thr_p:.6e}, TPR: {youden_tpr_p:.4f}, FPR: {youden_fpr_p:.4f}')

# Justificación breve:
print('\nJustificación del criterio elegido: Índice de Youden (J)') 
print('Youden maximiza (TPR - FPR), eligiendo un punto que balancea sensibilidad y especificidad.') 
print('Es simple de explicar y apropiado cuando se desea buen compromiso entre ambos errores.')

# ================================
# CLASIFICACIÓN NO SUPERVISADA: K-MEANS
# ================================

from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score
import matplotlib.gridspec as gridspec

def aplicar_kmeans_imagen(img, espacio_color='RGB', n_clusters=2, random_state=seed):
    """
    Aplica K-Means a una imagen en el espacio de color especificado
    """
    # Convertir al espacio de color deseado
    if espacio_color == 'RGB':
        img_features = img.copy()
    elif espacio_color == 'HSV':
        img_features = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        img_features = img_features.astype(np.float32) / 255.0
    elif espacio_color == 'LAB':
        img_features = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        img_features = img_features.astype(np.float32) / 255.0
    elif espacio_color == 'YCrCb':
        img_features = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        img_features = img_features.astype(np.float32) / 255.0
    else:
        raise ValueError("Espacio de color no soportado")
    
    # Redimensionar la imagen a una matriz 2D (píxeles x características)
    pixels = img_features.reshape(-1, img_features.shape[2])
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Reformar las etiquetas a la forma original de la imagen
    return labels.reshape(img.shape[0], img.shape[1]), kmeans.cluster_centers_

def asignar_clusters_a_clases(mask_real, clusters, centros):
    """
    Asigna los clusters de K-Means a las clases reales (lesión/no-lesión)
    comparando con la máscara de referencia
    """
    # Calcular superposición entre clusters y máscara real
    cluster_0_mask = (clusters == 0)
    cluster_1_mask = (clusters == 1)
    
    # Calcular qué cluster se superpone más con la lesión real
    lesion_overlap_0 = np.sum((cluster_0_mask) & (mask_real == 1))
    lesion_overlap_1 = np.sum((cluster_1_mask) & (mask_real == 1))
    
    # Asignar el cluster con mayor superposición a lesión
    if lesion_overlap_0 > lesion_overlap_1:
        lesion_cluster = 0
    else:
        lesion_cluster = 1
    
    # Crear máscara predicha
    mask_pred = (clusters == lesion_cluster).astype(np.uint8)
    
    return mask_pred

def evaluar_kmeans_espacios_color(imagenes_test, mascaras_test, espacios_color):
    """
    Evalúa K-Means con diferentes espacios de color y devuelve los resultados
    """
    resultados = {}
    
    for espacio in espacios_color:
        print(f"\nEvaluando espacio de color: {espacio}")
        jaccard_scores = []
        
        for i, (img, mask_real) in enumerate(zip(imagenes_test, mascaras_test)):
            # Aplicar K-Means
            clusters, centros = aplicar_kmeans_imagen(img, espacio_color=espacio)
            
            # Asignar clusters a clases
            mask_pred = asignar_clusters_a_clases(mask_real, clusters, centros)
            
            # Calcular métrica de similitud (Índice de Jaccard)
            jaccard = jaccard_score(mask_real.flatten(), mask_pred.flatten())
            jaccard_scores.append(jaccard)
        
        # Calcular estadísticas
        resultados[espacio] = {
            'jaccard_mean': np.mean(jaccard_scores),
            'jaccard_std': np.std(jaccard_scores),
            'jaccard_scores': jaccard_scores
        }
        
        print(f"Índice de Jaccard promedio: {resultados[espacio]['jaccard_mean']:.4f} ± {resultados[espacio]['jaccard_std']:.4f}")
    
    return resultados

def visualizar_resultados_kmeans(imagenes_test, mascaras_test, mejor_espacio):
    """
    Visualiza los resultados de K-Means para el mejor espacio de color
    """
    # Seleccionar algunas imágenes para visualización
    indices_visualizacion = [0, 1, 2]  # Primeras 3 imágenes
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    for i, idx in enumerate(indices_visualizacion):
        img = imagenes_test[idx]
        mask_real = mascaras_test[idx]
        
        # Aplicar K-Means con el mejor espacio de color
        clusters, centros = aplicar_kmeans_imagen(img, espacio_color=mejor_espacio)
        mask_pred = asignar_clusters_a_clases(mask_real, clusters, centros)
        
        # Imagen original
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(img)
        ax0.set_title(f'Imagen Original {idx+1}')
        ax0.axis('off')
        
        # Máscara real
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(mask_real, cmap='gray')
        ax1.set_title('Máscara Real')
        ax1.axis('off')
        
        # Resultado K-Means
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(mask_pred, cmap='gray')
        ax2.set_title(f'K-Means ({mejor_espacio})')
        ax2.axis('off')
        
        # Superposición
        ax3 = fig.add_subplot(gs[i, 3])
        superposicion = img.copy()
        # Resaltar áreas donde la predicción coincide con la realidad
        correcto = (mask_pred == mask_real) & (mask_real == 1)
        superposicion[correcto] = [0, 1, 0]  # Verde para aciertos en lesión
        incorrecto = (mask_pred != mask_real) & (mask_pred == 1)
        superposicion[incorrecto] = [1, 0, 0]  # Rojo para falsos positivos
        ax3.imshow(superposicion)
        ax3.set_title('Superposición (Verde: correcto, Rojo: error)')
        ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

# Espacios de color a evaluar
espacios_color = ['RGB', 'HSV', 'LAB', 'YCrCb']

print("\n" + "="*60)
print("CLASIFICACIÓN NO SUPERVISADA CON K-MEANS")
print("="*60)

# Evaluar K-Means con diferentes espacios de color
resultados_kmeans = evaluar_kmeans_espacios_color(test_images, test_masks, espacios_color)

# Encontrar el mejor espacio de color
mejor_espacio = max(resultados_kmeans.items(), key=lambda x: x[1]['jaccard_mean'])[0]
mejor_resultado = resultados_kmeans[mejor_espacio]

print(f"\nMEJOR COMBINACIÓN DE CARACTERÍSTICAS: {mejor_espacio}")
print(f"Índice de Jaccard promedio: {mejor_resultado['jaccard_mean']:.4f} ± {mejor_resultado['jaccard_std']:.4f}")

# Visualizar resultados con el mejor espacio de color
visualizar_resultados_kmeans(test_images, test_masks, mejor_espacio)

# Guardar resultados para la comparación final (tarea 3.6)
print("\nResultados de K-Means guardados para comparación final")