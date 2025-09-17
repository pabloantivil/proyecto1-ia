#!/usr/bin/env python3
"""
Ejecuta el análisis completo del main2.py usando los módulos organizados
"""

import numpy as np
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
    print("\\n1. CARGA Y PREPARACIÓN DE DATOS")
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
    X_test = datos['X_test']
    y_test = datos['y_test']
    
    # 2. Análisis de histogramas RGB
    print("\\n2. ANÁLISIS DE HISTOGRAMAS RGB")
    print("-"*40)
    lesion_pixels, no_lesion_pixels = mostrar_histogramas_estadisticos_main2(train_images, train_masks)
    
    # 3. Entrenamiento de clasificador Bayesiano RGB
    resultado_rgb = entrenar_bayesiano_rgb_main2(X_train, y_train, X_val, y_val)
    
    # 4. Entrenamiento de clasificador Bayesiano con PCA
    resultado_pca = entrenar_bayesiano_pca_main2(X_train, y_train, X_val, y_val)
    
    # 5. Mostrar curvas ROC comparativas
    print("\\n5. CURVAS ROC COMPARATIVAS")
    print("-"*40)
    mostrar_curvas_roc_main2(resultado_rgb, resultado_pca)
    
    # 6. Análisis de K-Means
    print("\\n6. ANÁLISIS DE K-MEANS")
    print("-"*40)
    resultado_kmeans = analisis_completo_kmeans_main2(val_images, val_masks)
    
    # 7. Comparación final de todos los clasificadores
    print("\\n7. COMPARACIÓN FINAL")
    print("-"*40)
    resultados_finales = comparacion_final_main2(
        test_images, test_masks,
        resultado_kmeans, resultado_rgb, resultado_pca
    )
    
    print("\\n" + "="*60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("Todos los resultados han sido calculados y visualizados.")
    print("Las gráficas se han mostrado en ventanas separadas.")
    
    return resultados_finales

if __name__ == "__main__":
    # Ejecutar el análisis completo
    resultados = main()