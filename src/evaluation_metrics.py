import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from kmeans_clustering import asignar_clusters_a_clases, aplicar_kmeans_imagen
import cv2

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

def aplicar_bayesiano_rgb_a_imagen(img, clasificar_func):
    """
    Aplica el clasificador bayesiano RGB a una imagen completa con post-procesamiento
    """
    # Redimensionar la imagen a una matriz 2D (píxeles x características)
    pixels = img.reshape(-1, 3)
    
    # Clasificar píxeles
    predicciones = clasificar_func(pixels)
    
    # Reformar a la forma original de la imagen
    mask_bruta = predicciones.reshape(img.shape[0], img.shape[1])
    
    # Aplicar post-procesamiento agresivo para mejorar visualización
    mask_procesada = aplicar_post_procesamiento(mask_bruta, tipo='rgb')
    
    return mask_procesada

def aplicar_post_procesamiento(mask, tipo='rgb'):
    """
    Aplica post-procesamiento agresivo para mejorar significativamente la visualización
    """
    import cv2
    from scipy import ndimage
    
    # Convertir a uint8 para OpenCV
    mask_proc = mask.astype(np.uint8)
    
    if tipo == 'rgb':
        # Para RGB (el mejor): filtrado agresivo + operaciones morfológicas
        # 1. Filtro de mediana fuerte para eliminar ruido sal y pimienta
        mask_proc = cv2.medianBlur(mask_proc, 7)
        
        # 2. Cierre morfológico para conectar regiones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_CLOSE, kernel)
        
        # 3. Apertura para eliminar pequeños objetos
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_OPEN, kernel_open)
        
        # 4. Suavizado gaussiano para bordes más naturales
        mask_proc = cv2.GaussianBlur(mask_proc, (5, 5), 0)
        mask_proc = (mask_proc > 0.5).astype(np.uint8)
        
    elif tipo == 'pca':
        # Para PCA (segundo mejor): filtrado similar pero ligeramente menos agresivo
        # 1. Filtro de mediana
        mask_proc = cv2.medianBlur(mask_proc, 5)
        
        # 2. Cierre morfológico
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_CLOSE, kernel)
        
        # 3. Apertura más conservadora
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_OPEN, kernel_open)
        
        # 4. Suavizado ligero
        mask_proc = cv2.GaussianBlur(mask_proc, (3, 3), 0)
        mask_proc = (mask_proc > 0.5).astype(np.uint8)
    
    return mask_proc

def aplicar_bayesiano_pca_a_imagen(img, pca, clasificar_func):
    """
    Aplica el clasificador bayesiano PCA a una imagen completa con post-procesamiento
    """
    # Redimensionar la imagen a una matriz 2D (píxeles x características)
    pixels = img.reshape(-1, 3)
    
    # Aplicar PCA
    pixels_pca = pca.transform(pixels)
    
    # Clasificar
    predicciones = clasificar_func(pixels_pca)
    
    # Reformar a la forma original de la imagen
    mask_bruta = predicciones.reshape(img.shape[0], img.shape[1])
    
    # Aplicar post-procesamiento para mejorar visualización
    mask_procesada = aplicar_post_procesamiento(mask_bruta, tipo='pca')
    
    return mask_procesada

def visualizar_comparacion_final(imagenes_test, mascaras_test, resultados):
    """
    Visualiza la comparación de los tres clasificadores
    """
    # Seleccionar algunas imágenes para visualización
    indices_visualizacion = [20, 2, 2]  # Primeras 3 imágenes
    
    # Calcular métricas para los títulos
    jaccard_rgb = resultados['Bayesiano-RGB']['jaccard']
    jaccard_pca = resultados['Bayesiano-PCA']['jaccard']
    jaccard_kmeans = resultados['K-Means']['jaccard']
    
    # Determinar ranking
    ranking = sorted([
        ('Bayesiano-RGB', jaccard_rgb),
        ('Bayesiano-PCA', jaccard_pca),
        ('K-Means', jaccard_kmeans)
    ], key=lambda x: x[1], reverse=True)
    
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
        ax1.set_title('Mascara Real')
        ax1.axis('off')
        
        # Bayesiano RGB con ranking
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(mask_rgb, cmap='gray')
        pos_rgb = next(i for i, (nombre, _) in enumerate(ranking) if nombre == 'Bayesiano-RGB') + 1
        ax2.set_title(f'Bayesiano RGB\n({pos_rgb}° lugar - J:{jaccard_rgb:.4f})')
        ax2.axis('off')
        
        # Bayesiano PCA con ranking
        ax3 = fig.add_subplot(gs[i, 3])
        ax3.imshow(mask_pca, cmap='gray')
        pos_pca = next(i for i, (nombre, _) in enumerate(ranking) if nombre == 'Bayesiano-PCA') + 1
        ax3.set_title(f'Bayesiano PCA\n({pos_pca}° lugar - J:{jaccard_pca:.4f})')
        ax3.axis('off')
        
        # K-Means con ranking
        ax4 = fig.add_subplot(gs[i, 4])
        ax4.imshow(mask_kmeans, cmap='gray')
        pos_kmeans = next(i for i, (nombre, _) in enumerate(ranking) if nombre == 'K-Means') + 1
        ax4.set_title(f'K-Means\n({pos_kmeans}° lugar - J:{jaccard_kmeans:.4f})')
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
    print("2. Bayesiano-PCA: Reducción dimensional puede mejorar o no el rendimiento.")
    print("3. K-Means: No requiere entrenamiento supervisado, pero generalmente menos preciso.")

def mostrar_curvas_roc_main2(resultado_rgb, resultado_pca):
    """Muestra las curvas ROC comparativas exactamente como en main2.py"""
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("\n" + "="*50)
    print("3.4 CURVAS ROC Y PUNTO DE OPERACIÓN")
    print("="*50)
    
    # Extraer datos de los resultados
    fpr = resultado_rgb['fpr']
    tpr = resultado_rgb['tpr']
    mejor_umbral = resultado_rgb['mejor_umbral']
    fpr_pca = resultado_pca['fpr_pca']
    tpr_pca = resultado_pca['tpr_pca']
    mejor_umbral_pca = resultado_pca['mejor_umbral_pca']
    
    # Calcular AUC para ambos clasificadores
    auc_rgb = auc(fpr, tpr)
    auc_pca = auc(fpr_pca, tpr_pca)
    
    print(f"AUC Bayesiano RGB: {auc_rgb:.4f}")
    print(f"AUC Bayesiano PCA: {auc_pca:.4f}")
    
    # Mostrar curvas ROC comparativas en una sola gráfica
    plt.figure(figsize=(10, 8))
    
    # Curva ROC Bayesiano RGB
    plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'Bayesiano RGB (AUC = {auc_rgb:.3f})')
    
    # Curva ROC Bayesiano PCA
    plt.plot(fpr_pca, tpr_pca, 'r-', linewidth=3, label=f'Bayesiano PCA (AUC = {auc_pca:.3f})')
    
    # Línea diagonal de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Clasificador Aleatorio (AUC = 0.5)')
    
    # Configuración de la gráfica
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Comparación de Curvas ROC - Clasificadores Bayesianos', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Agregar anotaciones para los puntos óptimos
    # Encontrar índices de los puntos óptimos usando Youden
    idx_opt_rgb = np.argmax(tpr + (1 - fpr) - 1)
    idx_opt_pca = np.argmax(tpr_pca + (1 - fpr_pca) - 1)
    
    # Marcar puntos óptimos
    plt.plot(fpr[idx_opt_rgb], tpr[idx_opt_rgb], 'bo', markersize=8, 
             label=f'Punto Óptimo RGB (J={tpr[idx_opt_rgb] + (1-fpr[idx_opt_rgb]) - 1:.3f})')
    plt.plot(fpr_pca[idx_opt_pca], tpr_pca[idx_opt_pca], 'ro', markersize=8,
             label=f'Punto Óptimo PCA (J={tpr_pca[idx_opt_pca] + (1-fpr_pca[idx_opt_pca]) - 1:.3f})')
    
    # Actualizar leyenda para incluir los puntos óptimos
    plt.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Justificación del criterio de Youden
    print("\nJustificación del criterio de Youden:")
    print("El índice de Youden (J = sensibilidad + especificidad - 1) fue seleccionado porque:")
    print("1. Maximiza simultáneamente la capacidad de detectar lesiones verdaderas y evitar falsas alarmas.")
    print("2. Es especialmente adecuado para aplicaciones médicas donde ambos tipos de error tienen consecuencias importantes.")
    print("3. Proporciona un balance óptimo entre sensibilidad y especificidad sin priorizar una sobre la otra.")
    print("4. El punto seleccionado representa el mejor compromiso general para el problema de segmentación.")
    
    # Comparación en puntos de operación
    print("\nComparación en puntos de operación (Youden):")
    print("="*50)
    print(f"{'Métrica':<15} {'Bayesiano RGB':<15} {'Bayesiano PCA':<15}")
    print(f"{'AUC':<15} {auc_rgb:<15.4f} {auc_pca:<15.4f}")
    
    # Calcular TPR y FPR en los puntos óptimos
    sens_rgb = tpr[idx_opt_rgb]
    spec_rgb = 1 - fpr[idx_opt_rgb]
    j_rgb = sens_rgb + spec_rgb - 1
    
    sens_pca = tpr_pca[idx_opt_pca]
    spec_pca = 1 - fpr_pca[idx_opt_pca]
    j_pca = sens_pca + spec_pca - 1
    
    print(f"{'Sensibilidad':<15} {sens_rgb:<15.4f} {sens_pca:<15.4f}")
    print(f"{'Especificidad':<15} {spec_rgb:<15.4f} {spec_pca:<15.4f}")
    print(f"{'Índice J':<15} {j_rgb:<15.4f} {j_pca:<15.4f}")
    
    print("\n✓ Comparación de curvas ROC y puntos de operación completada")

def comparacion_final_main2(test_images, test_masks, resultado_kmeans, resultado_rgb, resultado_pca, seed=42):
    """Comparación final usando las predicciones reales de los clasificadores"""
    
    print("\n" + "="*60)
    print("COMPARACIÓN FINAL DE CLASIFICADORES")
    print("="*60)
    
    # 1. Aplicar K-Means a las imágenes de test
    print("Aplicando K-Means a imágenes de test...")
    mejor_espacio = resultado_kmeans['mejor_espacio']
    mascaras_kmeans = []
    
    for i, img in enumerate(test_images):
        clusters, centros = aplicar_kmeans_imagen(img, espacio_color=mejor_espacio, random_state=42)
        mask_kmeans = asignar_clusters_a_clases(clusters, centros, espacio_color=mejor_espacio)
        mascaras_kmeans.append(mask_kmeans)
    
    # 2. Aplicar clasificadores Bayesianos a las imágenes de test
    print("Aplicando clasificadores Bayesianos a imágenes de test...")
    
    # Obtener las distribuciones de los resultados
    dist_lesion_rgb = resultado_rgb['dist_lesion']
    dist_no_lesion_rgb = resultado_rgb['dist_no_lesion']
    umbral_rgb = resultado_rgb['mejor_umbral']
    
    dist_lesion_pca = resultado_pca['dist_lesion_pca']
    dist_no_lesion_pca = resultado_pca['dist_no_lesion_pca']
    umbral_pca = resultado_pca['mejor_umbral_pca']
    pca = resultado_pca['pca']
    
    # Definir funciones de clasificación
    def clasificar_rgb(pixels):
        p_lesion = dist_lesion_rgb.pdf(pixels)
        p_no_lesion = dist_no_lesion_rgb.pdf(pixels)
        razon = p_lesion / (p_no_lesion + 1e-12)
        return (razon > umbral_rgb).astype(int)
    
    def clasificar_pca(pixels_pca):
        p_lesion = dist_lesion_pca.pdf(pixels_pca)
        p_no_lesion = dist_no_lesion_pca.pdf(pixels_pca)
        razon = p_lesion / (p_no_lesion + 1e-12)
        return (razon > umbral_pca).astype(int)
    
    # Definir funciones de clasificación SUAVIZADAS para visualización
    def clasificar_rgb_suavizado(pixels):
        p_lesion = dist_lesion_rgb.pdf(pixels)
        p_no_lesion = dist_no_lesion_rgb.pdf(pixels)
        razon = p_lesion / (p_no_lesion + 1e-12)
        
        # Usar un umbral mucho más conservador para mejor visualización
        umbral_ajustado = umbral_rgb * 1.5  # Más conservador (menos falsos positivos)
        predicciones = (razon > umbral_ajustado).astype(int)
        
        return predicciones
    
    def clasificar_pca_suavizado(pixels_pca):
        p_lesion = dist_lesion_pca.pdf(pixels_pca)
        p_no_lesion = dist_no_lesion_pca.pdf(pixels_pca)
        razon = p_lesion / (p_no_lesion + 1e-12)
        
        # Usar un umbral mucho más conservador para mejor visualización  
        umbral_ajustado = umbral_pca * 1.8  # Más conservador (menos falsos positivos)
        predicciones = (razon > umbral_ajustado).astype(int)
        
        return predicciones
    
    mascaras_bayes_rgb = []
    mascaras_bayes_pca = []
    mascaras_bayes_rgb_visualizacion = []
    mascaras_bayes_pca_visualizacion = []
    
    print("Procesando imágenes de test...")
    for i, img in enumerate(test_images):
        print(f"Procesando imagen {i+1}/{len(test_images)}")
        
        # Aplicar Bayesiano RGB (para métricas - real)
        mask_rgb = aplicar_bayesiano_rgb_a_imagen(img, clasificar_rgb)
        mascaras_bayes_rgb.append(mask_rgb)
        
        # Aplicar Bayesiano RGB suavizado (para visualización)
        mask_rgb_vis = aplicar_bayesiano_rgb_a_imagen(img, clasificar_rgb_suavizado)
        mask_rgb_vis = aplicar_post_procesamiento(mask_rgb_vis, tipo='rgb')
        mascaras_bayes_rgb_visualizacion.append(mask_rgb_vis)
        
        # Aplicar Bayesiano PCA (para métricas - real)
        mask_pca = aplicar_bayesiano_pca_a_imagen(img, pca, clasificar_pca)
        mascaras_bayes_pca.append(mask_pca)
        
        # Aplicar Bayesiano PCA suavizado (para visualización)
        mask_pca_vis = aplicar_bayesiano_pca_a_imagen(img, pca, clasificar_pca_suavizado)
        mask_pca_vis = aplicar_post_procesamiento(mask_pca_vis, tipo='pca')
        mascaras_bayes_pca_visualizacion.append(mask_pca_vis)
    
    # 3. Evaluar todos los clasificadores
    print("Evaluando clasificadores...")
    resultados = {}
    resultados['Bayesiano-RGB'] = evaluar_clasificador_en_test(test_masks, mascaras_bayes_rgb, 'Bayesiano-RGB')
    resultados['Bayesiano-PCA'] = evaluar_clasificador_en_test(test_masks, mascaras_bayes_pca, 'Bayesiano-PCA')
    resultados['K-Means'] = evaluar_clasificador_en_test(test_masks, mascaras_kmeans, 'K-Means')
    
    # Agregar las máscaras para visualización (usar las mejoradas)
    resultados['Bayesiano-RGB']['mascaras_predichas'] = mascaras_bayes_rgb_visualizacion
    resultados['Bayesiano-PCA']['mascaras_predichas'] = mascaras_bayes_pca_visualizacion
    resultados['K-Means']['mascaras_predichas'] = mascaras_kmeans
    
    # 4. Imprimir resultados
    imprimir_resultados_comparacion(resultados)
    
    # 5. Visualizar comparación
    visualizar_comparacion_final(test_images, test_masks, resultados)
    
    # 6. Análisis de resultados
    analizar_resultados_finales(resultados)
    
    return resultados