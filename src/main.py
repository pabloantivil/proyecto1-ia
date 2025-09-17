#!/usr/bin/env python3
"""
Ejecuta el análisis completo del main2.py usando los módulos organizados
"""
from data_loader import preparar_datos_main2
from rgb_analysis import mostrar_histogramas_estadisticos_main2
from bayesian_classifiers import (
    entrenar_bayesiano_rgb_main2, 
    entrenar_bayesiano_pca_main2,
    razon_verosimilitud,
    clasificar_bayes
)
from evaluation_metrics import mostrar_curvas_roc_main2, comparacion_final_main2
from kmeans_clustering import analisis_completo_kmeans_main2

def main():
    """Función principal que ejecuta todo el análisis del main2.py"""
    
    print("Iniciando análisis de imágenes médicas...")
    print("="*60)
    
    # 1. Cargar y preparar datos
    print("\n1. CARGA Y PREPARACIÓN DE DATOS")
    print("-"*40)
    datos = preparar_datos_main2()
    
    # Extraer datos
    train_images = datos['train_images']
    train_masks = datos['train_masks']
    val_images = datos['val_images']
    val_masks = datos['val_masks']
    test_images = datos['test_images']
    test_masks = datos['test_masks']
    X_train = datos['X_train']
    y_train = datos['y_train']
    X_val = datos['X_val']
    y_val = datos['y_val']
    
    # 2. Análisis RGB
    print("\n2. ANÁLISIS RGB")
    print("-"*40)
    mostrar_histogramas_estadisticos_main2(train_images, train_masks)
    
    # 3. Clasificadores Bayesianos
    print("\n3. CLASIFICADORES BAYESIANOS")
    print("-"*40)
    
    # 3.1 Bayesiano RGB
    resultado_rgb = entrenar_bayesiano_rgb_main2(X_train, y_train, X_val, y_val)
    
    # 3.2 Bayesiano PCA
    resultado_pca = entrenar_bayesiano_pca_main2(X_train, y_train, X_val, y_val)
    
    # 3.3 Curvas ROC
    mostrar_curvas_roc_main2(resultado_rgb, resultado_pca)
    
    # 4. K-Means
    print("\n4. K-MEANS CLUSTERING")
    print("-"*40)
    resultado_kmeans = analisis_completo_kmeans_main2(val_images, val_masks)
    
    # 5. Comparación final
    print("\n5. COMPARACIÓN FINAL")
    print("-"*40)
    comparacion_final_main2(test_images, test_masks, resultado_kmeans, resultado_rgb, resultado_pca)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETO FINALIZADO")
    print("="*60)

if __name__ == "__main__":
    main()