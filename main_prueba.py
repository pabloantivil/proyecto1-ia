"""
Proyecto de Clasificación de Lesiones Dermatológicas
Clasificadores a nivel de píxel: Bayesiano, Bayesiano+PCA, K-Means
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Fijar semilla para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class DermatologyClassifier:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.random_seed = RANDOM_SEED
        self.scaler = StandardScaler()
        self.pca = None
        self.bayesian_rgb_params = {}
        self.bayesian_pca_params = {}
        self.kmeans_model = None
        self.train_images = []
        self.val_images = []
        self.test_images = []
        
    def load_and_split_data(self):
        """Cargar imágenes y dividir en entrenamiento, validación y test"""
        print("Cargando y dividiendo datos...")
        
        # Obtener lista de imágenes (archivos .jpg sin _expert)
        image_files = glob.glob(os.path.join(self.dataset_path, "*.jpg"))
        image_files = [f for f in image_files if '_expert' not in f]
        
        print(f"Total de imágenes encontradas: {len(image_files)}")
        
        # Dividir por imagen (no por píxeles)
        train_files, temp_files = train_test_split(image_files, test_size=0.4, random_state=self.random_seed)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=self.random_seed)
        
        print(f"Entrenamiento: {len(train_files)} imágenes")
        print(f"Validación: {len(val_files)} imágenes") 
        print(f"Test: {len(test_files)} imágenes")
        
        self.train_images = train_files
        self.val_images = val_files
        self.test_images = test_files
        
        return train_files, val_files, test_files
    
    def load_image_and_mask(self, image_path):
        """Cargar imagen y su máscara correspondiente"""
        # Cargar imagen RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Obtener path de la máscara
        base_name = os.path.basename(image_path).replace('.jpg', '')
        mask_path = os.path.join(self.dataset_path, f"{base_name}_expert.png")
        
        # Cargar máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)  # Binarizar
        
        return image, mask
    
    def preprocess_image(self, image):
        """Preprocesamiento de imagen: normalización y corrección de iluminación"""
        # Normalización por canal
        image_norm = image.astype(np.float32) / 255.0
        
        # Corrección de iluminación (opcional - bonus)
        # Aplicar filtro gaussiano para estimar iluminación
        illumination = cv2.GaussianBlur(image_norm, (51, 51), 0)
        corrected = image_norm / (illumination + 0.01)  # Evitar división por cero
        corrected = np.clip(corrected, 0, 1)
        
        return corrected
    
    def extract_pixels_features(self, images_list, balanced_sampling=True, max_pixels_per_class=10000):
        """Extraer características de píxeles de las imágenes"""
        print(f"Extrayendo características de {len(images_list)} imágenes...")
        
        lesion_pixels = []
        normal_pixels = []
        
        for img_path in images_list:
            image, mask = self.load_image_and_mask(img_path)
            image_processed = self.preprocess_image(image)
            
            # Reshape para obtener píxeles
            h, w, c = image_processed.shape
            pixels = image_processed.reshape(-1, c)
            mask_flat = mask.reshape(-1)
            
            # Separar píxeles de lesión y normales
            lesion_idx = mask_flat == 1
            normal_idx = mask_flat == 0
            
            lesion_pixels.append(pixels[lesion_idx])
            normal_pixels.append(pixels[normal_idx])
        
        # Concatenar todos los píxeles
        all_lesion = np.vstack(lesion_pixels)
        all_normal = np.vstack(normal_pixels)
        
        print(f"Píxeles de lesión: {all_lesion.shape[0]}")
        print(f"Píxeles normales: {all_normal.shape[0]}")
        
        # Muestreo equilibrado para entrenamiento
        if balanced_sampling:
            n_samples = min(max_pixels_per_class, all_lesion.shape[0], all_normal.shape[0])
            
            # Muestreo aleatorio
            lesion_idx = np.random.choice(all_lesion.shape[0], n_samples, replace=False)
            normal_idx = np.random.choice(all_normal.shape[0], n_samples, replace=False)
            
            X_lesion = all_lesion[lesion_idx]
            X_normal = all_normal[normal_idx]
            
            # Combinar datos
            X = np.vstack([X_lesion, X_normal])
            y = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
            
            print(f"Datos balanceados: {n_samples} píxeles por clase")
        else:
            X = np.vstack([all_lesion, all_normal])
            y = np.hstack([np.ones(all_lesion.shape[0]), np.zeros(all_normal.shape[0])])
        
        return X, y
    
    def visualize_features(self, X_train, y_train):
        """Visualización de características RGB"""
        print("Generando visualizaciones...")
        
        # Separar clases
        lesion_pixels = X_train[y_train == 1]
        normal_pixels = X_train[y_train == 0]
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        # Histogramas por canal
        for i, (channel, color) in enumerate(zip(channels, colors)):
            # Histogramas
            axes[0, i].hist(lesion_pixels[:, i], bins=50, alpha=0.7, label='Lesión', 
                          color=color, density=True)
            axes[0, i].hist(normal_pixels[:, i], bins=50, alpha=0.7, label='Normal', 
                          color='gray', density=True)
            axes[0, i].set_title(f'Histograma Canal {channel}')
            axes[0, i].set_xlabel('Intensidad')
            axes[0, i].set_ylabel('Densidad')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # Estadísticas
        stats_data = []
        for i, channel in enumerate(channels):
            lesion_mean = np.mean(lesion_pixels[:, i])
            lesion_std = np.std(lesion_pixels[:, i])
            normal_mean = np.mean(normal_pixels[:, i])
            normal_std = np.std(normal_pixels[:, i])
            
            stats_data.append({
                'Canal': channel,
                'Lesión Media': f'{lesion_mean:.3f}',
                'Lesión Std': f'{lesion_std:.3f}',
                'Normal Media': f'{normal_mean:.3f}',
                'Normal Std': f'{normal_std:.3f}'
            })
        
        # Tabla de estadísticas
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table_data = [[row['Canal'], row['Lesión Media'], row['Lesión Std'], 
                      row['Normal Media'], row['Normal Std']] for row in stats_data]
        table = axes[1, 0].table(cellText=table_data,
                                colLabels=['Canal', 'Lesión Media', 'Lesión Std', 'Normal Media', 'Normal Std'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 0].set_title('Estadísticas por Canal')
        
        # Scatter plot 2D
        sample_size = min(1000, len(lesion_pixels), len(normal_pixels))
        lesion_sample = lesion_pixels[np.random.choice(len(lesion_pixels), sample_size, replace=False)]
        normal_sample = normal_pixels[np.random.choice(len(normal_pixels), sample_size, replace=False)]
        
        axes[1, 1].scatter(lesion_sample[:, 0], lesion_sample[:, 1], alpha=0.6, 
                          c='red', s=1, label='Lesión')
        axes[1, 1].scatter(normal_sample[:, 0], normal_sample[:, 1], alpha=0.6, 
                          c='blue', s=1, label='Normal')
        axes[1, 1].set_xlabel('Canal Rojo')
        axes[1, 1].set_ylabel('Canal Verde')
        axes[1, 1].set_title('Distribución R-G')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Boxplot
        data_to_plot = [lesion_pixels[:, 0], normal_pixels[:, 0],
                       lesion_pixels[:, 1], normal_pixels[:, 1],
                       lesion_pixels[:, 2], normal_pixels[:, 2]]
        labels = ['Lesión R', 'Normal R', 'Lesión G', 'Normal G', 'Lesión B', 'Normal B']
        
        axes[1, 2].boxplot(data_to_plot, labels=labels)
        axes[1, 2].set_title('Distribuciones por Canal y Clase')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_data
    
    def train_bayesian_classifier_rgb(self, X_train, y_train):
        """Entrenar clasificador Bayesiano con RGB completo"""
        print("Entrenando clasificador Bayesiano RGB...")
        
        # Separar clases
        X_lesion = X_train[y_train == 1]
        X_normal = X_train[y_train == 0]
        
        # Calcular parámetros gaussianos para cada clase
        self.bayesian_rgb_params = {
            'lesion': {
                'mean': np.mean(X_lesion, axis=0),
                'cov': np.cov(X_lesion.T),
                'prior': len(X_lesion) / len(X_train)
            },
            'normal': {
                'mean': np.mean(X_normal, axis=0),
                'cov': np.cov(X_normal.T),
                'prior': len(X_normal) / len(X_train)
            }
        }
        
        print(f"Prior lesión: {self.bayesian_rgb_params['lesion']['prior']:.3f}")
        print(f"Prior normal: {self.bayesian_rgb_params['normal']['prior']:.3f}")
    
    def apply_pca(self, X_train, n_components=None, variance_threshold=0.95):
        """Aplicar PCA y seleccionar componentes"""
        print("Aplicando PCA...")
        
        # Normalizar datos
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Si no se especifica número de componentes, usar criterio de varianza
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            
            print(f"Componentes seleccionados: {n_components} (varianza acumulada: {cumsum_variance[n_components-1]:.3f})")
        
        # Aplicar PCA con componentes seleccionados
        self.pca = PCA(n_components=n_components, random_state=self.random_seed)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Visualizar varianza explicada
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_)
        plt.xlabel('Componente Principal')
        plt.ylabel('Varianza Explicada')
        plt.title('Varianza Explicada por Componente')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                np.cumsum(self.pca.explained_variance_ratio_), 'o-')
        plt.axhline(y=variance_threshold, color='r', linestyle='--', 
                   label=f'Umbral {variance_threshold}')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Acumulada')
        plt.title('Varianza Acumulada')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return X_pca, n_components
    
    def train_bayesian_classifier_pca(self, X_pca, y_train):
        """Entrenar clasificador Bayesiano con componentes PCA"""
        print("Entrenando clasificador Bayesiano PCA...")
        
        # Separar clases
        X_lesion = X_pca[y_train == 1]
        X_normal = X_pca[y_train == 0]
        
        # Calcular parámetros gaussianos para cada clase
        self.bayesian_pca_params = {
            'lesion': {
                'mean': np.mean(X_lesion, axis=0),
                'cov': np.cov(X_lesion.T),
                'prior': len(X_lesion) / len(X_pca)
            },
            'normal': {
                'mean': np.mean(X_normal, axis=0),
                'cov': np.cov(X_normal.T),
                'prior': len(X_normal) / len(X_pca)
            }
        }
    
    def predict_bayesian(self, X, params, threshold=0.5):
        """Predicción con clasificador Bayesiano"""
        # Calcular verosimilitudes
        likelihood_lesion = multivariate_normal.pdf(X, 
                                                   params['lesion']['mean'], 
                                                   params['lesion']['cov'])
        likelihood_normal = multivariate_normal.pdf(X, 
                                                   params['normal']['mean'], 
                                                   params['normal']['cov'])
        
        # Aplicar priors
        posterior_lesion = likelihood_lesion * params['lesion']['prior']
        posterior_normal = likelihood_normal * params['normal']['prior']
        
        # Razón de verosimilitudes
        likelihood_ratio = likelihood_lesion / (likelihood_normal + 1e-10)
        
        # Decisión por umbral
        predictions = (likelihood_ratio > threshold).astype(int)
        
        # Probabilidades posteriores
        total_posterior = posterior_lesion + posterior_normal
        prob_lesion = posterior_lesion / (total_posterior + 1e-10)
        
        return predictions, prob_lesion, likelihood_ratio
    
    def find_optimal_threshold(self, y_true, prob_scores, criterion='youden'):
        """Encontrar umbral óptimo según criterio especificado"""
        fpr, tpr, thresholds = roc_curve(y_true, prob_scores)
        
        if criterion == 'youden':
            # Índice de Youden (J = TPR - FPR)
            youden_scores = tpr - fpr
            optimal_idx = np.argmax(youden_scores)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Umbral óptimo (Youden): {optimal_threshold:.4f}")
            
        elif criterion == 'eer':
            # Equal Error Rate (EER)
            fnr = 1 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            optimal_threshold = thresholds[eer_idx]
            print(f"Umbral óptimo (EER): {optimal_threshold:.4f}")
            
        elif criterion == 'tpr_constraint':
            # Restricción operativa (TPR >= 0.9)
            valid_idx = tpr >= 0.9
            if np.any(valid_idx):
                valid_fpr = fpr[valid_idx]
                min_fpr_idx = np.argmin(valid_fpr)
                optimal_idx = np.where(valid_idx)[0][min_fpr_idx]
                optimal_threshold = thresholds[optimal_idx]
                print(f"Umbral óptimo (TPR>=0.9): {optimal_threshold:.4f}")
            else:
                optimal_threshold = thresholds[np.argmax(tpr)]
                print("No se puede satisfacer TPR>=0.9, usando mejor TPR disponible")
        
        return optimal_threshold, fpr[optimal_idx], tpr[optimal_idx]
    
    def train_kmeans(self, X_train, n_clusters=2):
        """Entrenar clasificador K-Means"""
        print("Entrenando K-Means...")
        
        # Normalizar datos
        X_scaled = self.scaler.transform(X_train)
        
        # Entrenar K-Means
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        return cluster_labels
    
    def assign_kmeans_labels(self, cluster_labels, y_true):
        """Asignar etiquetas de clusters a clases reales"""
        # Determinar qué cluster corresponde a qué clase
        cluster_0_lesion_ratio = np.mean(y_true[cluster_labels == 0])
        cluster_1_lesion_ratio = np.mean(y_true[cluster_labels == 1])
        
        # Asignar cluster con mayor proporción de lesiones a clase 1
        if cluster_0_lesion_ratio > cluster_1_lesion_ratio:
            label_mapping = {0: 1, 1: 0}
        else:
            label_mapping = {0: 0, 1: 1}
        
        mapped_labels = np.array([label_mapping[label] for label in cluster_labels])
        
        print(f"Cluster 0 - % lesión: {cluster_0_lesion_ratio:.3f}")
        print(f"Cluster 1 - % lesión: {cluster_1_lesion_ratio:.3f}")
        
        return mapped_labels
    
    def evaluate_classifier(self, y_true, y_pred, y_proba=None, classifier_name=""):
        """Evaluar clasificador con métricas completas"""
        print(f"\n=== Evaluación {classifier_name} ===")
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Especificidad
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Mostrar resultados
        print(f"Exactitud: {accuracy:.4f}")
        print(f"Precisión (Normal/Lesión): {precision[0]:.4f} / {precision[1]:.4f}")
        print(f"Sensibilidad (Recall): {recall[1]:.4f}")
        print(f"Especificidad: {specificity:.4f}")
        
        # AUC si hay probabilidades
        auc_score = None
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = auc(fpr, tpr)
            print(f"AUC: {auc_score:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'precision_normal': precision[0],
            'precision_lesion': precision[1],
            'recall_lesion': recall[1],
            'specificity': specificity,
            'auc': auc_score,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_roc_curves(self, results):
        """Graficar curvas ROC comparativas"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green']
        
        for i, (name, result) in enumerate(results.items()):
            if 'fpr' in result and 'tpr' in result:
                plt.plot(result['fpr'], result['tpr'], 
                        color=colors[i], linewidth=2,
                        label=f"{name} (AUC = {result['auc']:.3f})")
                
                # Marcar punto de operación si existe
                if 'optimal_fpr' in result and 'optimal_tpr' in result:
                    plt.plot(result['optimal_fpr'], result['optimal_tpr'], 
                            'o', color=colors[i], markersize=8,
                            label=f"{name} - Punto Óptimo")
        
        # Línea diagonal
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador Aleatorio')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title('Curvas ROC Comparativas')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_on_test_images(self, test_images, show_samples=True):
        """Evaluar clasificadores en imágenes de test completas"""
        print("\n=== Evaluación en conjunto de test ===")
        
        all_results = {
            'bayesian_rgb': {'y_true': [], 'y_pred': [], 'y_proba': [], 'jaccard_scores': []},
            'bayesian_pca': {'y_true': [], 'y_pred': [], 'y_proba': [], 'jaccard_scores': []},
            'kmeans': {'y_true': [], 'y_pred': [], 'jaccard_scores': []}
        }
        
        sample_images = []
        
        for i, img_path in enumerate(test_images):
            if i % 10 == 0:
                print(f"Procesando imagen {i+1}/{len(test_images)}")
            
            # Cargar imagen y máscara
            image, mask = self.load_image_and_mask(img_path)
            image_processed = self.preprocess_image(image)
            
            # Preparar datos
            h, w, c = image_processed.shape
            pixels = image_processed.reshape(-1, c)
            mask_flat = mask.reshape(-1)
            
            # Predicciones Bayesiano RGB
            pred_rgb, prob_rgb, _ = self.predict_bayesian(pixels, self.bayesian_rgb_params, 
                                                        threshold=self.optimal_threshold_rgb)
            
            # Predicciones Bayesiano PCA
            pixels_scaled = self.scaler.transform(pixels)
            pixels_pca = self.pca.transform(pixels_scaled)
            pred_pca, prob_pca, _ = self.predict_bayesian(pixels_pca, self.bayesian_pca_params, 
                                                        threshold=self.optimal_threshold_pca)
            
            # Predicciones K-Means
            cluster_labels = self.kmeans_model.predict(pixels_scaled)
            pred_kmeans = self.assign_kmeans_labels(cluster_labels, mask_flat)
            
            # Guardar resultados a nivel de píxel
            all_results['bayesian_rgb']['y_true'].extend(mask_flat)
            all_results['bayesian_rgb']['y_pred'].extend(pred_rgb)
            all_results['bayesian_rgb']['y_proba'].extend(prob_rgb)
            
            all_results['bayesian_pca']['y_true'].extend(mask_flat)
            all_results['bayesian_pca']['y_pred'].extend(pred_pca)
            all_results['bayesian_pca']['y_proba'].extend(prob_pca)
            
            all_results['kmeans']['y_true'].extend(mask_flat)
            all_results['kmeans']['y_pred'].extend(pred_kmeans)
            
            # Calcular índice de Jaccard por imagen
            for method_name, preds in [('bayesian_rgb', pred_rgb), 
                                     ('bayesian_pca', pred_pca), 
                                     ('kmeans', pred_kmeans)]:
                intersection = np.sum((mask_flat == 1) & (preds == 1))
                union = np.sum((mask_flat == 1) | (preds == 1))
                jaccard = intersection / union if union > 0 else 0
                all_results[method_name]['jaccard_scores'].append(jaccard)
            
            # Guardar muestra para visualización
            if len(sample_images) < 3:
                pred_rgb_img = pred_rgb.reshape(h, w)
                pred_pca_img = pred_pca.reshape(h, w)
                pred_kmeans_img = pred_kmeans.reshape(h, w)
                
                sample_images.append({
                    'original': image,
                    'mask': mask,
                    'bayesian_rgb': pred_rgb_img,
                    'bayesian_pca': pred_pca_img,
                    'kmeans': pred_kmeans_img,
                    'filename': os.path.basename(img_path)
                })
        
        # Evaluar métricas finales
        final_results = {}
        for method_name, results in all_results.items():
            y_true = np.array(results['y_true'])
            y_pred = np.array(results['y_pred'])
            y_proba = np.array(results['y_proba']) if 'y_proba' in results else None
            jaccard_scores = results['jaccard_scores']
            
            metrics = self.evaluate_classifier(y_true, y_pred, y_proba, method_name)
            metrics['mean_jaccard'] = np.mean(jaccard_scores)
            metrics['std_jaccard'] = np.std(jaccard_scores)
            
            print(f"Índice de Jaccard promedio: {metrics['mean_jaccard']:.4f} ± {metrics['std_jaccard']:.4f}")
            
            final_results[method_name] = metrics
        
        # Mostrar muestras de segmentación
        if show_samples:
            self.show_segmentation_samples(sample_images)
        
        return final_results
    
    def show_segmentation_samples(self, sample_images):
        """Mostrar muestras de segmentación"""
        fig, axes = plt.subplots(len(sample_images), 5, figsize=(20, 4*len(sample_images)))
        
        if len(sample_images) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(sample_images):
            # Imagen original
            axes[i, 0].imshow(sample['original'])
            axes[i, 0].set_title(f"Original\n{sample['filename']}")
            axes[i, 0].axis('off')
            
            # Máscara ground truth
            axes[i, 1].imshow(sample['mask'], cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Bayesiano RGB
            axes[i, 2].imshow(sample['bayesian_rgb'], cmap='gray')
            axes[i, 2].set_title("Bayesiano RGB")
            axes[i, 2].axis('off')
            
            # Bayesiano PCA
            axes[i, 3].imshow(sample['bayesian_pca'], cmap='gray')
            axes[i, 3].set_title("Bayesiano PCA")
            axes[i, 3].axis('off')
            
            # K-Means
            axes[i, 4].imshow(sample['kmeans'], cmap='gray')
            axes[i, 4].set_title("K-Means")
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig('segmentation_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self, results):
        """Crear tabla de comparación final"""
        print("\n=== TABLA DE COMPARACIÓN FINAL ===")
        print("-" * 80)
        print(f"{'Métrica':<20} {'Bayesiano RGB':<15} {'Bayesiano PCA':<15} {'K-Means':<15}")
        print("-" * 80)
        
        metrics_to_show = [
            ('Exactitud', 'accuracy'),
            ('Precisión Lesión', 'precision_lesion'),
            ('Sensibilidad', 'recall_lesion'),
            ('Especificidad', 'specificity'),
            ('AUC', 'auc'),
            ('Jaccard Promedio', 'mean_jaccard')
        ]
        
        for metric_name, metric_key in metrics_to_show:
            row = f"{metric_name:<20}"
            for method in ['bayesian_rgb', 'bayesian_pca', 'kmeans']:
                value = results[method].get(metric_key, 'N/A')
                if isinstance(value, float):
                    row += f"{value:<15.4f}"
                else:
                    row += f"{str(value):<15}"
            print(row)
        
        print("-" * 80)
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("=== INICIO DEL ANÁLISIS COMPLETO ===\n")
        
        # 1. Cargar y dividir datos
        train_files, val_files, test_files = self.load_and_split_data()
        
        # 2. Extraer características de entrenamiento
        X_train, y_train = self.extract_pixels_features(train_files, balanced_sampling=True)
        
        # 3. Visualización y análisis exploratorio
        stats = self.visualize_features(X_train, y_train)
        
        # 4. Entrenar clasificador Bayesiano RGB
        self.train_bayesian_classifier_rgb(X_train, y_train)
        
        # 5. Aplicar PCA y entrenar clasificador Bayesiano PCA
        X_train_pca, n_components = self.apply_pca(X_train, variance_threshold=0.95)
        self.train_bayesian_classifier_pca(X_train_pca, y_train)
        
        # 6. Entrenar K-Means
        kmeans_labels = self.train_kmeans(X_train)
        kmeans_mapped = self.assign_kmeans_labels(kmeans_labels, y_train)
        
        # 7. Evaluar en validación para encontrar umbrales óptimos
        print("\n=== Evaluación en conjunto de validación ===")
        X_val, y_val = self.extract_pixels_features(val_files, balanced_sampling=False)
        
        # Predicciones en validación
        _, prob_rgb_val, lr_rgb_val = self.predict_bayesian(X_val, self.bayesian_rgb_params)
        
        X_val_scaled = self.scaler.transform(X_val)
        X_val_pca = self.pca.transform(X_val_scaled)
        _, prob_pca_val, lr_pca_val = self.predict_bayesian(X_val_pca, self.bayesian_pca_params)
        
        # Encontrar umbrales óptimos (usando Youden por defecto)
        self.optimal_threshold_rgb, opt_fpr_rgb, opt_tpr_rgb = self.find_optimal_threshold(
            y_val, lr_rgb_val, criterion='youden')
        self.optimal_threshold_pca, opt_fpr_pca, opt_tpr_pca = self.find_optimal_threshold(
            y_val, lr_pca_val, criterion='youden')
        
        # 8. Generar curvas ROC
        fpr_rgb, tpr_rgb, _ = roc_curve(y_val, lr_rgb_val)
        auc_rgb = auc(fpr_rgb, tpr_rgb)
        
        fpr_pca, tpr_pca, _ = roc_curve(y_val, lr_pca_val)
        auc_pca = auc(fpr_pca, tpr_pca)
        
        roc_results = {
            'Bayesiano RGB': {
                'fpr': fpr_rgb, 'tpr': tpr_rgb, 'auc': auc_rgb,
                'optimal_fpr': opt_fpr_rgb, 'optimal_tpr': opt_tpr_rgb
            },
            'Bayesiano PCA': {
                'fpr': fpr_pca, 'tpr': tpr_pca, 'auc': auc_pca,
                'optimal_fpr': opt_fpr_pca, 'optimal_tpr': opt_tpr_pca
            }
        }
        
        self.plot_roc_curves(roc_results)
        
        # 9. Evaluación final en test
        final_results = self.evaluate_on_test_images(test_files)
        
        # 10. Tabla de comparación final
        self.create_comparison_table(final_results)
        
        print("\n=== ANÁLISIS COMPLETADO ===")
        return final_results

# Ejecutar análisis
if __name__ == "__main__":
    # Crear instancia del clasificador
    classifier = DermatologyClassifier(dataset_path='dataset')
    
    # Ejecutar análisis completo
    results = classifier.run_complete_analysis()
    
    print("\nProyecto completado exitosamente!")
    print("Archivos generados:")
    print("- feature_analysis.png: Análisis exploratorio de características")
    print("- pca_analysis.png: Análisis de componentes principales")
    print("- roc_curves_comparison.png: Comparación de curvas ROC")
    print("- segmentation_samples.png: Muestras de segmentación")