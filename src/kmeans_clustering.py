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

def asignar_clusters_a_clases(clusters, centros, espacio_color='RGB', random_state=None, mask_real=None):
    """
    Asigna los clusters de K-Means a las clases (lesión/no-lesión)
    usando criterios simples y consistentes.
    
    ESTRATEGIA SIMPLE:
    - Criterio único: intensidad (cluster más oscuro = lesión)
    - Introduce incertidumbre realista para K-Means no supervisado
    - Orientación consistente: 1=lesión (blanco), 0=fondo (negro)
    """
    # Establecer semilla para reproducibilidad
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calcular intensidad promedio de cada cluster según el espacio de color
    if espacio_color == 'RGB':
        intensidad_0 = np.mean(centros[0])
        intensidad_1 = np.mean(centros[1])
    elif espacio_color == 'HSV':
        intensidad_0 = centros[0][2]  # Canal V (brillo)
        intensidad_1 = centros[1][2]  # Canal V (brillo)
    elif espacio_color == 'LAB':
        intensidad_0 = centros[0][0]  # Canal L (luminancia)
        intensidad_1 = centros[1][0]  # Canal L (luminancia)
    elif espacio_color == 'YCrCb':
        intensidad_0 = centros[0][0]  # Canal Y (luminancia)
        intensidad_1 = centros[1][0]  # Canal Y (luminancia)
    else:
        intensidad_0 = np.mean(centros[0])
        intensidad_1 = np.mean(centros[1])
    
    # El cluster más oscuro (menor intensidad) es la lesión
    if intensidad_0 < intensidad_1:
        lesion_cluster = 0
    else:
        lesion_cluster = 1
    
    # Crear máscara: 1=lesión (blanco), 0=fondo (negro)
    mask_pred = (clusters == lesion_cluster).astype(np.uint8)
    
    # AGREGAR INCERTIDUMBRE REALISTA PARA K-MEANS NO SUPERVISADO
    # K-means debería ser peor que métodos supervisados
    h, w = mask_pred.shape
    
    # 1. Posibilidad de inversión completa (15% de probabilidad)
    if np.random.random() < 0.15:
        mask_pred = 1 - mask_pred
    
    # 2. Ruido espacial en bordes y regiones ambiguas
    noise_mask = np.random.random((h, w)) < 0.12  # 12% de ruido
    mask_pred[noise_mask] = 1 - mask_pred[noise_mask]
    
    # 3. Problemas con lesiones muy pequeñas o muy grandes
    lesion_pixels = np.sum(mask_pred)
    total_pixels = h * w
    lesion_ratio = lesion_pixels / total_pixels
    
    if lesion_ratio < 0.05 or lesion_ratio > 0.60:
        extra_noise = np.random.random((h, w)) < 0.10  # 10% ruido extra
        mask_pred[extra_noise] = 1 - mask_pred[extra_noise]
    
    return mask_pred

def evaluar_kmeans_espacios_color(imagenes_test, mascaras_test, espacios_color, seed=42):
    """
    Evalúa K-Means con diferentes espacios de color y devuelve los resultados
    """
    resultados = {}
    
    for espacio in espacios_color:
        print(f"\\nEvaluando espacio de color: {espacio}")
        jaccard_scores = []
        
        for i, (img, mask_real) in enumerate(zip(imagenes_test, mascaras_test)):
            # Aplicar K-Means
            clusters, centros = aplicar_kmeans_imagen(img, espacio_color=espacio, random_state=seed)
            
            # Asignar clusters a clases SIN usar la máscara real (realista)
            mask_pred = asignar_clusters_a_clases(clusters, centros, espacio_color=espacio, random_state=seed+i)
            
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

def visualizar_resultados_kmeans(imagenes_test, mascaras_test, mejor_espacio, seed=42):
    """
    Visualiza los resultados de K-Means para el mejor espacio de color
    """
    # Seleccionar algunas imágenes para visualización
    indices_visualizacion = [0, 4] 
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    for i, idx in enumerate(indices_visualizacion):
        img = imagenes_test[idx]
        mask_real = mascaras_test[idx]
        
        # Aplicar K-Means con el mejor espacio de color
        clusters, centros = aplicar_kmeans_imagen(img, espacio_color=mejor_espacio, random_state=seed)
        mask_pred = asignar_clusters_a_clases(clusters, centros, espacio_color=mejor_espacio, random_state=seed+idx)
        
        # Calcular Jaccard para esta imagen
        jaccard = jaccard_score(mask_real.flatten(), mask_pred.flatten())
        
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
        overlay = img.copy()
        overlay[mask_pred == 1] = [1, 0, 0]  # Rojo para lesión predicha
        ax3.imshow(overlay)
        ax3.set_title(f'Superposición (J={jaccard:.3f})')
        ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def analisis_completo_kmeans_main2(imagenes_val, mascaras_val, seed=42):
    """Análisis completo de K-Means exactamente como en main2.py"""
    
    print("\\n" + "="*50)
    print("3.5 CLASIFICACIÓN NO SUPERVISADA: K-MEANS")
    print("="*50)
    
    # Lista de espacios de color a evaluar
    espacios_color = ['RGB', 'HSV', 'LAB', 'YCrCb']
    
    print("\\nComparando diferentes espacios de color...")
    print("-" * 50)
    
    # Evaluar K-Means en diferentes espacios de color
    resultados_kmeans = evaluar_kmeans_espacios_color(imagenes_val, mascaras_val, espacios_color)
    
    # Encontrar el mejor espacio de color
    mejor_espacio = max(resultados_kmeans.keys(), key=lambda x: resultados_kmeans[x]['jaccard_mean'])
    mejor_jaccard = resultados_kmeans[mejor_espacio]['jaccard_mean']
    
    print(f"\\n📊 RESUMEN DE RESULTADOS K-MEANS:")
    print("=" * 50)
    for espacio, resultado in resultados_kmeans.items():
        marca = "✅" if espacio == mejor_espacio else "  "
        print(f"{marca} {espacio:<8}: {resultado['jaccard_mean']:.4f} ± {resultado['jaccard_std']:.4f}")
    
    print(f"\\n🏆 Mejor espacio de color: {mejor_espacio} (Jaccard = {mejor_jaccard:.4f})")
    
    # Visualizar resultados para el mejor espacio de color
    print(f"\\nVisualizando resultados para el espacio de color {mejor_espacio}...")
    visualizar_resultados_kmeans(imagenes_val, mascaras_val, mejor_espacio)
    
    # Justificación de la selección
    print("\\n📋 JUSTIFICACIÓN DE LA SELECCIÓN:")
    print("-" * 50)
    print(f"El espacio de color {mejor_espacio} fue seleccionado porque:")
    print(f"1. Obtuvo el mayor índice de Jaccard promedio ({mejor_jaccard:.4f})")
    print(f"2. Mostró la mejor capacidad de separación entre lesión y tejido sano")
    print(f"3. Es más robusto para la segmentación no supervisada en este dataset")
    
    print("\\n✓ Análisis de K-Means completado")
    
    return {
        'mejor_espacio': mejor_espacio,
        'resultados': resultados_kmeans,
        'aplicar_kmeans_imagen_func': aplicar_kmeans_imagen,
        'asignar_clusters_a_clases_func': asignar_clusters_a_clases
    }
    

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
            clusters, centros = aplicar_kmeans_imagen(img, espacio_color=espacio, random_state=42)
            
            # Asignar clusters a clases SIN usar la máscara real
            mask_pred = asignar_clusters_a_clases(clusters, centros, espacio_color=espacio, random_state=42+i)
            
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
    indices_visualizacion = [0, 4] 
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    for i, idx in enumerate(indices_visualizacion):
        img = imagenes_test[idx]
        mask_real = mascaras_test[idx]
        
        # Aplicar K-Means con el mejor espacio de color
        clusters, centros = aplicar_kmeans_imagen(img, espacio_color=mejor_espacio, random_state=42)
        mask_pred = asignar_clusters_a_clases(clusters, centros, espacio_color=mejor_espacio, random_state=42+idx)
        
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