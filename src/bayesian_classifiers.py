import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from scipy.stats import multivariate_normal

# Variables globales para almacenar distribuciones
dist_lesion = None
dist_no_lesion = None
dist_lesion_pca = None
dist_no_lesion_pca = None
pca = None

def configurar_bayesiano_rgb(lesion_pixels, no_lesion_pixels):
    """Configura el clasificador Bayesiano RGB calculando medias y covarianzas"""
    global dist_lesion, dist_no_lesion
    
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
    
    return mu_lesion, cov_lesion, mu_no_lesion, cov_no_lesion

def clasificar_bayes(X, umbral=1.0):
    """Clasifica píxeles RGB usando razón de verosimilitud"""
    global dist_lesion, dist_no_lesion
    
    p_lesion = dist_lesion.pdf(X)
    p_no_lesion = dist_no_lesion.pdf(X)

    # Razón de verosimilitudes
    razon = p_lesion / (p_no_lesion + 1e-12)  # evitar división por 0

    # Decisión
    return (razon > umbral).astype(int)

def evaluar_bayesiano_rgb(X_validacion, y_validacion, mostrar_matriz=True):
    """Evalúa el clasificador Bayesiano RGB en datos de validación"""
    global dist_lesion, dist_no_lesion
    
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