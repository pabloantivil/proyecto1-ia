import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score

def aplicar_kmeans_imagen(img, espacio_color='RGB', n_clusters=2, random_state=42):
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
    plt.savefig('kmeans_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def ejecutar_analisis_kmeans(test_images, test_masks):
    """Función principal para ejecutar análisis completo de K-Means"""
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

    # Guardar resultados para la comparación final
    print("\nResultados de K-Means guardados para comparación final")
    
    return resultados_kmeans, mejor_espacio, mejor_resultado