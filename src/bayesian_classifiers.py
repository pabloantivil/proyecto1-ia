import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from scipy.stats import multivariate_normal

def calcular_parametros_clase(X, y):
    """Calcula parámetros de las distribuciones para cada clase"""
    # Separar píxeles por clase
    lesion_pixels = X[y == 1]
    no_lesion_pixels = X[y == 0]
    
    # Calcular media y covarianza para cada clase
    media_lesion = np.mean(lesion_pixels, axis=0)
    cov_lesion = np.cov(lesion_pixels, rowvar=False)
    
    media_no_lesion = np.mean(no_lesion_pixels, axis=0)
    cov_no_lesion = np.cov(no_lesion_pixels, rowvar=False)
    
    return media_lesion, cov_lesion, media_no_lesion, cov_no_lesion

def razon_verosimilitud(X, dist_lesion, dist_no_lesion):
    """Calcula la razón de verosimilitud"""
    # Calcular probabilidades para cada clase
    p_lesion = dist_lesion.pdf(X)
    p_no_lesion = dist_no_lesion.pdf(X)
    
    # Calcular razón de verosimilitud (evitar división por cero)
    epsilon = 1e-10
    lr = p_lesion / (p_no_lesion + epsilon)
    
    return lr

def encontrar_mejor_umbral(lr, y_true):
    """Encuentra el mejor umbral usando el índice de Youden"""
    # Calcular curva ROC
    fpr, tpr, umbrales = roc_curve(y_true, lr)
    
    # Calcular índice de Youden (J = sensibilidad + especificidad - 1)
    j_scores = tpr + (1 - fpr) - 1
    
    # Encontrar el umbral que maximiza el índice de Youden
    mejor_idx = np.argmax(j_scores)
    mejor_umbral = umbrales[mejor_idx]
    mejor_j = j_scores[mejor_idx]
    
    return mejor_umbral, mejor_j, fpr, tpr, umbrales

def clasificar_bayes(X, dist_lesion, dist_no_lesion, umbral):
    """Clasifica usando el umbral de la razón de verosimilitud"""
    lr = razon_verosimilitud(X, dist_lesion, dist_no_lesion)
    return (lr >= umbral).astype(int)

def entrenar_bayesiano_rgb_main2(X_entrenamiento, y_entrenamiento, X_validacion, y_validacion):
    """Entrena clasificador Bayesiano RGB exactamente como en main2.py"""
    
    print("\\n" + "="*50)
    print("3.2 CLASIFICADOR BAYESIANO (RGB)")
    print("="*50)
    
    # Calcular parámetros usando el conjunto de entrenamiento
    media_lesion, cov_lesion, media_no_lesion, cov_no_lesion = calcular_parametros_clase(X_entrenamiento, y_entrenamiento)

    print("Parámetros calculados:")
    print(f"Media lesión: {media_lesion}")
    print(f"Media no-lesión: {media_no_lesion}")

    # Crear distribuciones normales multivariadas
    dist_lesion = multivariate_normal(mean=media_lesion, cov=cov_lesion, allow_singular=True)
    dist_no_lesion = multivariate_normal(mean=media_no_lesion, cov=cov_no_lesion, allow_singular=True)

    # Calcular razón de verosimilitud para validación
    lr_validacion = razon_verosimilitud(X_validacion, dist_lesion, dist_no_lesion)

    # Encontrar el mejor umbral usando validación
    mejor_umbral, mejor_j, fpr, tpr, umbrales = encontrar_mejor_umbral(lr_validacion, y_validacion)

    print(f"\\nMejor umbral (Youden): {mejor_umbral:.4f}")
    print(f"Índice de Youden (J): {mejor_j:.4f}")

    # Evaluar en validación con el mejor umbral
    y_pred_validacion = clasificar_bayes(X_validacion, dist_lesion, dist_no_lesion, mejor_umbral)

    # Calcular métricas
    accuracy = accuracy_score(y_validacion, y_pred_validacion)
    precision = precision_score(y_validacion, y_pred_validacion)
    sensibilidad = recall_score(y_validacion, y_pred_validacion)  # TPR
    especificidad = confusion_matrix(y_validacion, y_pred_validacion)[0, 0] / np.sum(y_validacion == 0)  # TNR

    print("\\nResultados en validación:")
    print(f"Exactitud: {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad: {sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")

    # Matriz de confusión
    matrix_confusion= confusion_matrix(y_validacion, y_pred_validacion)
    vis = ConfusionMatrixDisplay(matrix_confusion, display_labels=["No-lesión", "Lesión"])
    vis.plot()
    plt.title("Matriz de Confusión - Clasificador Bayesiano RGB")
    plt.show()

    print("\\n✓ Clasificador Bayesiano RGB implementado y evaluado")
    
    return {
        'dist_lesion': dist_lesion,
        'dist_no_lesion': dist_no_lesion,
        'mejor_umbral': mejor_umbral,
        'fpr': fpr,
        'tpr': tpr,
        'umbrales': umbrales,
        'roc_auc': auc(fpr, tpr)
    }

def entrenar_bayesiano_pca_main2(X_entrenamiento, y_entrenamiento, X_validacion, y_validacion, seed=42):
    """Entrena clasificador Bayesiano + PCA exactamente como en main2.py"""
    
    print("\\n" + "="*50)
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

    print(f"\\nDimensión original: {X_entrenamiento.shape[1]}")
    print(f"Dimensión después de PCA: {X_entrenamiento_pca.shape[1]}")

    # Entrenar clasificador Bayesiano en el espacio PCA
    media_lesion_pca, cov_lesion_pca, media_no_lesion_pca, cov_no_lesion_pca = calcular_parametros_clase(
        X_entrenamiento_pca, y_entrenamiento)

    print("\\nParámetros en espacio PCA:")
    print(f"Media lesión PCA: {media_lesion_pca}")
    print(f"Media no-lesión PCA: {media_no_lesion_pca}")

    # Crear distribuciones en espacio PCA
    dist_lesion_pca = multivariate_normal(mean=media_lesion_pca, cov=cov_lesion_pca, allow_singular=True)
    dist_no_lesion_pca = multivariate_normal(mean=media_no_lesion_pca, cov=cov_no_lesion_pca, allow_singular=True)

    # Encontrar mejor umbral en espacio PCA
    lr_validacion_pca = razon_verosimilitud(X_validacion_pca, dist_lesion_pca, dist_no_lesion_pca)
    mejor_umbral_pca, mejor_j_pca, fpr_pca, tpr_pca, umbrales_pca = encontrar_mejor_umbral(lr_validacion_pca, y_validacion)

    print(f"\\nMejor umbral PCA (Youden): {mejor_umbral_pca:.4f}")
    print(f"Índice de Youden PCA (J): {mejor_j_pca:.4f}")

    # Evaluar en validación
    y_pred_validacion_pca = clasificar_bayes(X_validacion_pca, dist_lesion_pca, dist_no_lesion_pca, mejor_umbral_pca)

    # Métricas
    accuracy_pca = accuracy_score(y_validacion, y_pred_validacion_pca)
    precision_pca = precision_score(y_validacion, y_pred_validacion_pca)
    sensibilidad_pca = recall_score(y_validacion, y_pred_validacion_pca)
    especificidad_pca = confusion_matrix(y_validacion, y_pred_validacion_pca)[0, 0] / np.sum(y_validacion == 0)

    print("\\nResultados PCA en validación:")
    print(f"Exactitud: {accuracy_pca:.4f}")
    print(f"Precisión: {precision_pca:.4f}")
    print(f"Sensibilidad: {sensibilidad_pca:.4f}")
    print(f"Especificidad: {especificidad_pca:.4f}")

    # Matriz de confusión PCA
    matrix_confusion_pca = confusion_matrix(y_validacion, y_pred_validacion_pca)
    vis_pca = ConfusionMatrixDisplay(matrix_confusion_pca, display_labels=["No-lesión", "Lesión"])
    vis_pca.plot()
    plt.title("Matriz de Confusión - Clasificador Bayesiano + PCA")
    plt.show()

    print("\\n✓ Clasificador Bayesiano + PCA implementado y evaluado")
    
    return {
        'pca': pca,
        'dist_lesion_pca': dist_lesion_pca,
        'dist_no_lesion_pca': dist_no_lesion_pca,
        'mejor_umbral_pca': mejor_umbral_pca,
        'fpr_pca': fpr_pca,
        'tpr_pca': tpr_pca,
        'umbrales_pca': umbrales_pca,
        'roc_auc_pca': auc(fpr_pca, tpr_pca)
    }
    
    # Clasificar datos de validación
    p_lesion_val = dist_lesion.pdf(X_validacion)
    p_no_lesion_val = dist_no_lesion.pdf(X_validacion)
    razon_val = p_lesion_val / (p_no_lesion_val + 1e-12)
    
    # Decisión con umbral = 1.0
    y_pred = (razon_val > 1.0).astype(int)

    if mostrar_matriz:
        # Matriz de confusión
        matrix_confusion = confusion_matrix(y_validacion, y_pred)
        vis = ConfusionMatrixDisplay(matrix_confusion, display_labels=["No-lesión", "Lesión"])
        vis.plot()
        plt.title("Matriz de Confusión - Bayesiano RGB")
        plt.savefig('confusion_bayesiano_rgb.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Evaluación
    print("\nResultados en VALIDACIÓN:")
    print("Accuracy:", accuracy_score(y_validacion, y_pred))
    print("Precision:", precision_score(y_validacion, y_pred))
    print("\nReporte de clasificación:\n", classification_report(y_validacion, y_pred, target_names=["No-lesión", "Lesión"]))
    
    return y_pred, razon_val

def configurar_bayesiano_pca(X_entrenamiento, n_componentes=None, varianza_objetivo=0.95):
    """Configura PCA y clasificador Bayesiano en espacio PCA"""
    global pca, dist_lesion_pca, dist_no_lesion_pca
    
    print("\nAplicando PCA a los datos...")
    pca = PCA()
    X_entrenamiento_pca = pca.fit_transform(X_entrenamiento)

    # Calcular varianza acumulada
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

    # Seleccionar número de componentes que explican al menos el objetivo de varianza
    if n_componentes is None:
        n_componentes = np.argmax(varianza_acumulada >= varianza_objetivo) + 1
    
    print(f"Número de componentes seleccionados: {n_componentes}")
    
    # Usar solo las primeras n_componentes
    X_entrenamiento_pca = X_entrenamiento_pca[:, :n_componentes]
    pca_reducido = PCA(n_components=n_componentes)
    X_entrenamiento_pca = pca_reducido.fit_transform(X_entrenamiento)
    
    # Actualizar PCA global
    pca = pca_reducido
    
    return X_entrenamiento_pca, n_componentes

def entrenar_bayesiano_pca(X_entrenamiento_pca, y_entrenamiento):
    """Entrena el clasificador Bayesiano en espacio PCA"""
    global dist_lesion_pca, dist_no_lesion_pca
    
    # Separar datos por clase en espacio PCA
    lesion_indices = y_entrenamiento == 1
    no_lesion_indices = y_entrenamiento == 0
    
    lesion_pca = X_entrenamiento_pca[lesion_indices]
    no_lesion_pca = X_entrenamiento_pca[no_lesion_indices]
    
    # Calcular medias y covarianzas en espacio PCA
    mu_lesion_pca = np.mean(lesion_pca, axis=0)
    cov_lesion_pca = np.cov(lesion_pca, rowvar=False)
    
    mu_no_lesion_pca = np.mean(no_lesion_pca, axis=0)
    cov_no_lesion_pca = np.cov(no_lesion_pca, rowvar=False)
    
    # Crear distribuciones
    dist_lesion_pca = multivariate_normal(mean=mu_lesion_pca, cov=cov_lesion_pca, allow_singular=True)
    dist_no_lesion_pca = multivariate_normal(mean=mu_no_lesion_pca, cov=cov_no_lesion_pca, allow_singular=True)
    
    return mu_lesion_pca, cov_lesion_pca, mu_no_lesion_pca, cov_no_lesion_pca

def clasificar_bayes_pca(X_pca, umbral=1.0):
    """Clasifica píxeles en espacio PCA usando razón de verosimilitud"""
    global dist_lesion_pca, dist_no_lesion_pca
    
    p_lesion = dist_lesion_pca.pdf(X_pca)
    p_no_lesion = dist_no_lesion_pca.pdf(X_pca)

    # Razón de verosimilitudes
    razon = p_lesion / (p_no_lesion + 1e-12)  # evitar división por 0

    # Decisión
    return (razon > umbral).astype(int)

def evaluar_bayesiano_pca(X_validacion_pca, y_validacion, mostrar_matriz=True):
    """Evalúa el clasificador Bayesiano PCA en datos de validación"""
    global dist_lesion_pca, dist_no_lesion_pca
    
    # Evaluar clasificador PCA en validación
    p_lesion_pca_val = dist_lesion_pca.pdf(X_validacion_pca)
    p_no_lesion_pca_val = dist_no_lesion_pca.pdf(X_validacion_pca)
    razon_pca = p_lesion_pca_val / (p_no_lesion_pca_val + 1e-12)
    
    # Decisión con umbral = 1.0
    y_pred_pca = (razon_pca > 1.0).astype(int)

    if mostrar_matriz:
        # Matriz de confusión para PCA
        matrix_confusion_pca = confusion_matrix(y_validacion, y_pred_pca)
        vis_pca = ConfusionMatrixDisplay(matrix_confusion_pca, display_labels=["No-lesión", "Lesión"])
        vis_pca.plot()
        plt.title("Matriz de Confusión - Bayesiano + PCA")
        plt.savefig('confusion_bayesiano_pca.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Evaluación
    print("\nResultados en VALIDACIÓN (Bayesiano + PCA):")
    print("Accuracy:", accuracy_score(y_validacion, y_pred_pca))
    print("Precision:", precision_score(y_validacion, y_pred_pca))
    print("\nReporte de clasificación:\n", classification_report(y_validacion, y_pred_pca, target_names=["No-lesión", "Lesión"]))
    
    return y_pred_pca, razon_pca

def comparar_clasificadores(y_validacion, y_pred_rgb, y_pred_pca):
    """Compara los resultados de los clasificadores RGB y PCA"""
    print("\nCOMPARACIÓN CON CLASIFICADOR RGB COMPLETO:")
    print("Accuracy RGB: {:.4f} vs Accuracy PCA: {:.4f}".format(
        accuracy_score(y_validacion, y_pred_rgb), 
        accuracy_score(y_validacion, y_pred_pca)))

def generar_curvas_roc(X_validacion, X_validacion_pca, y_validacion):
    """Genera curvas ROC para ambos clasificadores"""
    global dist_lesion, dist_no_lesion, dist_lesion_pca, dist_no_lesion_pca
    
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
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Imprimir resumen de puntos de operación (Youden J)
    print('\nPUNTO DE OPERACIÓN (Youden J óptimo):')
    print(f'RGB - AUC: {auc_r:.4f}, Youden thr: {youden_thr_r:.6e}, TPR: {youden_tpr_r:.4f}, FPR: {youden_fpr_r:.4f}')
    print(f'PCA - AUC: {auc_p:.4f}, Youden thr: {youden_thr_p:.6e}, TPR: {youden_tpr_p:.4f}, FPR: {youden_fpr_p:.4f}')

    # Justificación breve:
    print('\nJustificación del criterio elegido: Índice de Youden (J)') 
    print('Youden maximiza (TPR - FPR), eligiendo un punto que balancea sensibilidad y especificidad.') 
    print('Es simple de explicar y apropiado cuando se desea buen compromiso entre ambos errores.')
    
    return {
        'rgb': {'fpr': fpr_r, 'tpr': tpr_r, 'auc': auc_r, 'youden_thr': youden_thr_r},
        'pca': {'fpr': fpr_p, 'tpr': tpr_p, 'auc': auc_p, 'youden_thr': youden_thr_p}
    }