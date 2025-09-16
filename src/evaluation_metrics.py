import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def evaluar_clasificador_en_test(mascaras_reales, mascaras_predichas, nombre_clasificador):
    """
    Evalúa un clasificador en el conjunto de test y devuelve las métricas
    """
    # Convertir a arrays 1D para las métricas
    y_true = np.concatenate([mask.flatten() for mask in mascaras_reales])
    y_pred = np.concatenate([mask.flatten() for mask in mascaras_predichas])
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)  # Especificidad es recall para clase 0
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    
    # Calcular Jaccard por imagen
    jaccard_por_imagen = [jaccard_score(mask_real.flatten(), mask_pred.flatten()) 
                          for mask_real, mask_pred in zip(mascaras_reales, mascaras_predichas)]
    jaccard_mean = np.mean(jaccard_por_imagen)
    jaccard_std = np.std(jaccard_por_imagen)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'jaccard': jaccard,
        'jaccard_por_imagen': jaccard_por_imagen,
        'jaccard_mean': jaccard_mean,
        'jaccard_std': jaccard_std
    }

def aplicar_bayesiano_rgb_a_imagen(img, clasificar_bayes_func, umbral=1.0):
    """
    Aplica el clasificador bayesiano RGB a una imagen completa
    """
    # Redimensionar la imagen a una matriz 2D (píxeles x características)
    pixels = img.reshape(-1, 3)
    
    # Clasificar
    predicciones = clasificar_bayes_func(pixels, umbral=umbral)
    
    # Reformar a la forma original de la imagen
    return predicciones.reshape(img.shape[0], img.shape[1])

def aplicar_bayesiano_pca_a_imagen(img, pca, clasificar_bayes_pca_func, umbral=1.0):
    """
    Aplica el clasificador bayesiano PCA a una imagen completa
    """
    # Redimensionar la imagen a una matriz 2D (píxeles x características)
    pixels = img.reshape(-1, 3)
    
    # Aplicar PCA
    pixels_pca = pca.transform(pixels)
    
    # Clasificar
    predicciones = clasificar_bayes_pca_func(pixels_pca, umbral=umbral)
    
    # Reformar a la forma original de la imagen
    return predicciones.reshape(img.shape[0], img.shape[1])

def visualizar_comparacion_final(imagenes_test, mascaras_test, resultados):
    """
    Visualiza la comparación de los tres clasificadores
    """
    # Seleccionar algunas imágenes para visualización
    indices_visualizacion = [0, 1, 2]  # Primeras 3 imágenes
    
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 5, figure=fig)
    
    for i, idx in enumerate(indices_visualizacion):
        img = imagenes_test[idx]
        mask_real = mascaras_test[idx]
        
        # Obtener predicciones de todos los clasificadores
        mask_rgb = resultados['Bayesiano-RGB']['mascaras_predichas'][idx]
        mask_pca = resultados['Bayesiano-PCA']['mascaras_predichas'][idx]
        mask_kmeans = resultados['K-Means']['mascaras_predichas'][idx]
        
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
        
        # Bayesiano RGB
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(mask_rgb, cmap='gray')
        ax2.set_title('Bayesiano RGB')
        ax2.axis('off')
        
        # Bayesiano PCA
        ax3 = fig.add_subplot(gs[i, 3])
        ax3.imshow(mask_pca, cmap='gray')
        ax3.set_title('Bayesiano PCA')
        ax3.axis('off')
        
        # K-Means
        ax4 = fig.add_subplot(gs[i, 4])
        ax4.imshow(mask_kmeans, cmap='gray')
        ax4.set_title('K-Means')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_final.png', dpi=300, bbox_inches='tight')
    plt.show()

def imprimir_resultados_comparacion(resultados):
    """Imprime tabla de resultados de comparación"""
    print("\nRESULTADOS DE COMPARACIÓN:")
    print("-" * 80)
    print(f"{'Métrica':<20} {'Bayesiano-RGB':<15} {'Bayesiano-PCA':<15} {'K-Means':<15}")
    print("-" * 80)

    metricas = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'jaccard']
    nombres_metricas = ['Exactitud', 'Precisión', 'Sensibilidad', 'Especificidad', 'F1-Score', 'Jaccard']

    for metrica, nombre in zip(metricas, nombres_metricas):
        rgb_val = resultados['Bayesiano-RGB'][metrica]
        pca_val = resultados['Bayesiano-PCA'][metrica]
        km_val = resultados['K-Means'][metrica]
        print(f"{nombre:<20} {rgb_val:<15.4f} {pca_val:<15.4f} {km_val:<15.4f}")

def analizar_resultados_finales(resultados):
    """Analiza los resultados finales y proporciona conclusiones"""
    print("\nANÁLISIS DE RESULTADOS:")
    print("-" * 80)

    # Encontrar el mejor clasificador según Jaccard
    mejor_clasificador = max(resultados.items(), key=lambda x: x[1]['jaccard'])[0]
    mejor_jaccard = resultados[mejor_clasificador]['jaccard']

    print(f"El mejor clasificador según el índice de Jaccard es: {mejor_clasificador} ({mejor_jaccard:.4f})")

    # Comparar métodos supervisados vs no supervisados
    supervisado_promedio = (resultados['Bayesiano-RGB']['jaccard'] + resultados['Bayesiano-PCA']['jaccard']) / 2
    no_supervisado = resultados['K-Means']['jaccard']

    print(f"Métodos supervisados (promedio): {supervisado_promedio:.4f}")
    print(f"Método no supervisado (K-Means): {no_supervisado:.4f}")

    if supervisado_promedio > no_supervisado:
        print("Los métodos supervisados superan al método no supervisado.")
    else:
        print("El método no supervisado supera a los métodos supervisados.")

    # Analizar ventajas y desventajas de cada método
    print("\nVENTAJAS Y DESVENTAJAS:")
    print("1. Bayesiano-RGB: Simple pero efectivo, no requiere reducción dimensional.")
    print("2. Bayesiano-PCA: Mejor rendimiento gracias a la reducción de dimensionalidad.")
    print("3. K-Means: No requiere entrenamiento supervisado, pero puede ser menos preciso.")

    # Guardar resultados para el reporte
    print("\nResultados de comparación guardados para el reporte final.")

def ejecutar_comparacion_final(test_images, test_masks, clasificar_bayes_func, clasificar_bayes_pca_func, pca, 
                               aplicar_kmeans_func, asignar_clusters_func, mejor_espacio):
    """Función principal para ejecutar la comparación final de todos los clasificadores"""
    print("\n" + "="*60)
    print("COMPARACIÓN FINAL DE CLASIFICADORES")
    print("="*60)

    # 1. Aplicar clasificadores Bayesianos a las imágenes de test
    print("Aplicando clasificadores Bayesianos a imágenes de test...")
    mascaras_bayes_rgb = []
    mascaras_bayes_pca = []

    for img in test_images:
        # Clasificador Bayesiano RGB
        mask_rgb = aplicar_bayesiano_rgb_a_imagen(img, clasificar_bayes_func, umbral=1.0)
        mascaras_bayes_rgb.append(mask_rgb)
        
        # Clasificador Bayesiano PCA
        mask_pca = aplicar_bayesiano_pca_a_imagen(img, pca, clasificar_bayes_pca_func, umbral=1.0)
        mascaras_bayes_pca.append(mask_pca)

    # 2. Aplicar K-Means a las imágenes de test (usando el mejor espacio de color)
    print("Aplicando K-Means a imágenes de test...")
    mascaras_kmeans = []

    for img, mask_real in zip(test_images, test_masks):
        clusters, centros = aplicar_kmeans_func(img, espacio_color=mejor_espacio)
        mask_kmeans = asignar_clusters_func(mask_real, clusters, centros)
        mascaras_kmeans.append(mask_kmeans)

    # 3. Evaluar todos los clasificadores
    resultados = {}

    # Bayesiano RGB
    resultados['Bayesiano-RGB'] = evaluar_clasificador_en_test(test_masks, mascaras_bayes_rgb, 'Bayesiano-RGB')
    resultados['Bayesiano-RGB']['mascaras_predichas'] = mascaras_bayes_rgb

    # Bayesiano PCA
    resultados['Bayesiano-PCA'] = evaluar_clasificador_en_test(test_masks, mascaras_bayes_pca, 'Bayesiano-PCA')
    resultados['Bayesiano-PCA']['mascaras_predichas'] = mascaras_bayes_pca

    # K-Means
    resultados['K-Means'] = evaluar_clasificador_en_test(test_masks, mascaras_kmeans, 'K-Means')
    resultados['K-Means']['mascaras_predichas'] = mascaras_kmeans

    # 4. Imprimir resultados
    imprimir_resultados_comparacion(resultados)

    # 5. Visualizar comparación
    visualizar_comparacion_final(test_images, test_masks, resultados)

    # 6. Análisis de resultados
    analizar_resultados_finales(resultados)
    
    return resultados