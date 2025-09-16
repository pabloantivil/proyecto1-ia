import numpy as np
from data_loader import cargar_y_preparar_datos, muestreo_equilibrado
from rgb_analysis import analisis_completo_rgb
from bayesian_classifiers import (configurar_bayesiano_rgb, evaluar_bayesiano_rgb, 
                                 configurar_bayesiano_pca, entrenar_bayesiano_pca, 
                                 evaluar_bayesiano_pca, comparar_clasificadores, 
                                 generar_curvas_roc, clasificar_bayes, clasificar_bayes_pca)
from kmeans_clustering import ejecutar_analisis_kmeans, aplicar_kmeans_imagen, asignar_clusters_a_clases
from evaluation_metrics import ejecutar_comparacion_final

def main():
    # Configuración inicial
    seed = 42
    np.random.seed(seed)

    # Ruta del dataset
    ruta = "C:/Users/benja/Desktop/ia/proyecto1-ia/dataset"

    print("="*60)
    print("ANÁLISIS DE LESIONES DERMATOSCÓPICAS")
    print("="*60)

    # ================================
    # 1. CARGA Y PREPARACIÓN DE DATOS
    # ================================
    print("\n1. CARGANDO Y PREPARANDO DATOS...")
    datos = cargar_y_preparar_datos(ruta, seed)
    
    train_images = datos['train_images']
    train_masks = datos['train_masks']
    val_images = datos['val_images']
    val_masks = datos['val_masks']
    test_images = datos['test_images']
    test_masks = datos['test_masks']

    # Procesar datos de entrenamiento, validación y test
    X_entrenamiento, y_entrenamiento = muestreo_equilibrado(train_images, train_masks, n=10000)
    X_validacion, y_validacion = muestreo_equilibrado(val_images, val_masks, n=5000)
    X_test, y_test = muestreo_equilibrado(test_images, test_masks, n=5000)

    print(f"Datos de entrenamiento: {len(X_entrenamiento)} muestras")
    print(f"Datos de validación: {len(X_validacion)} muestras") 
    print(f"Datos de test: {len(X_test)} muestras")

    # ================================
    # 2. ANÁLISIS DE CANALES RGB
    # ================================
    print("\n2. ANÁLISIS DE CANALES RGB...")
    lesion_pixels, no_lesion_pixels = analisis_completo_rgb(train_images, train_masks)

    # ================================
    # 3. CLASIFICADOR BAYESIANO RGB
    # ================================
    print("\n3. CLASIFICADOR BAYESIANO RGB...")
    configurar_bayesiano_rgb(lesion_pixels, no_lesion_pixels)
    y_pred_rgb, scores_rgb = evaluar_bayesiano_rgb(X_validacion, y_validacion)

    # ================================
    # 4. CLASIFICADOR BAYESIANO + PCA
    # ================================
    print("\n4. CLASIFICADOR BAYESIANO + PCA...")
    X_entrenamiento_pca, n_componentes = configurar_bayesiano_pca(X_entrenamiento)
    entrenar_bayesiano_pca(X_entrenamiento_pca, y_entrenamiento)
    
    # Transformar datos de validación con PCA
    from bayesian_classifiers import pca
    X_validacion_pca = pca.transform(X_validacion)
    
    y_pred_pca, scores_pca = evaluar_bayesiano_pca(X_validacion_pca, y_validacion)

    # ================================
    # 5. COMPARACIÓN RGB vs PCA
    # ================================
    print("\n5. COMPARACIÓN RGB vs PCA...")
    comparar_clasificadores(y_validacion, y_pred_rgb, y_pred_pca)
    generar_curvas_roc(X_validacion, X_validacion_pca, y_validacion)

    # ================================
    # 6. CLASIFICACIÓN NO SUPERVISADA: K-MEANS
    # ================================
    print("\n6. CLASIFICACIÓN NO SUPERVISADA: K-MEANS...")
    resultados_kmeans, mejor_espacio, mejor_resultado = ejecutar_analisis_kmeans(test_images, test_masks)

    # ================================
    # 7. COMPARACIÓN FINAL DE TODOS LOS CLASIFICADORES
    # ================================
    print("\n7. COMPARACIÓN FINAL DE TODOS LOS CLASIFICADORES...")
    resultados_finales = ejecutar_comparacion_final(
        test_images, test_masks, 
        clasificar_bayes, clasificar_bayes_pca, pca,
        aplicar_kmeans_imagen, asignar_clusters_a_clases, mejor_espacio
    )

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("Se han generado las siguientes figuras:")
    print("1. histogramas_rgb.png - Histogramas de canales RGB")
    print("2. confusion_bayesiano_rgb.png - Matriz de confusión Bayesiano RGB")
    print("3. confusion_bayesiano_pca.png - Matriz de confusión Bayesiano PCA")
    print("4. roc_curves.png - Curvas ROC comparativas")
    print("5. kmeans_results.png - Resultados de K-Means")
    print("6. comparison_final.png - Comparación final de todos los métodos")

if __name__ == "__main__":
    main()