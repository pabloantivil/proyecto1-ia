import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split

def carga_imagen(paths):
    """Carga imágenes y sus máscaras correspondientes"""
    imagenes, mascaras = [], []
    for imagen_path in paths:
        img = cv2.cvtColor(cv2.imread(imagen_path), cv2.COLOR_BGR2RGB)
        mask_path = imagen_path.replace('.jpg', '_expert.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Advertencia: No se encontró la máscara para {mask_path}")
            continue
        mask = (mask > 0).astype(np.uint8)  # Binaria
        imagenes.append(img)
        mascaras.append(mask)
    return imagenes, mascaras

def normalize(images):
    """Normaliza las imágenes a valores entre 0 y 1"""
    return [img.astype(np.float32) / 255.0 for img in images]  # Conversión explícita

def corregir_iluminacion(img):
    """Aplica corrección de iluminación usando CLAHE"""
    # Convertir a uint8 para CLAHE si está normalizada
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0  # Normalizar de vuelta

def muestreo_equilibrado(imagenes, mascaras, n=10000):
    """Muestreo equilibrado de píxeles de lesión y no lesión"""
    lesion_pixels = []
    no_lesion_pixels = []
    
    for img, mask in zip(imagenes, mascaras):
        # Extraer coordenadas donde mask == 1 y mask == 0
        lesion_coords = np.where(mask == 1)
        no_lesion_coords = np.where(mask == 0)
        
        # Extraer píxeles RGB usando las coordenadas
        lesion_rgb = img[lesion_coords]  # Shape: (n_lesion_pixels, 3)
        no_lesion_rgb = img[no_lesion_coords]  # Shape: (n_no_lesion_pixels, 3)
        
        lesion_pixels.extend(lesion_rgb)
        no_lesion_pixels.extend(no_lesion_rgb)
    
    # Convertir a arrays numpy
    lesion_pixels = np.array(lesion_pixels)
    no_lesion_pixels = np.array(no_lesion_pixels)
    
    # Muestreo aleatorio equilibrado
    n_per_class = min(n//2, len(lesion_pixels), len(no_lesion_pixels))
    
    lesion_indices = np.random.choice(len(lesion_pixels), n_per_class, replace=False)
    no_lesion_indices = np.random.choice(len(no_lesion_pixels), n_per_class, replace=False)
    
    # Crear dataset final
    X = np.vstack([lesion_pixels[lesion_indices], no_lesion_pixels[no_lesion_indices]])
    y = np.hstack([np.ones(n_per_class), np.zeros(n_per_class)])
    
    # Mezclar los datos
    shuffle_indices = np.random.permutation(len(X))
    return X[shuffle_indices], y[shuffle_indices]

def cargar_y_preparar_datos(ruta, seed=42):
    """Función principal para cargar y preparar todos los datos"""
    np.random.seed(seed)
    
    # Lista de imágenes excluyendo las que son máscaras
    imagen_path = sorted([p for p in glob.glob(ruta + "/*.jpg") if "_expert" not in p])
    
    # Partición de datos por imagen
    entrenamiento_val, test = train_test_split(imagen_path, test_size=0.2, random_state=seed)
    entrenamiento, val = train_test_split(entrenamiento_val, test_size=0.25, random_state=seed)
    
    print("Entrenamiento:", len(entrenamiento), "Validación:", len(val), "Test:", len(test))
    
    # Cargar imágenes
    train_images, train_masks = carga_imagen(entrenamiento)
    val_images, val_masks = carga_imagen(val)
    test_images, test_masks = carga_imagen(test)
    
    # Normalizar
    train_images = normalize(train_images)
    val_images = normalize(val_images)
    test_images = normalize(test_images)
    
    return {
        'train_images': train_images,
        'train_masks': train_masks,
        'val_images': val_images,
        'val_masks': val_masks,
        'test_images': test_images,
        'test_masks': test_masks,
        'entrenamiento_paths': entrenamiento,
        'val_paths': val,
        'test_paths': test
    }