import numpy as np
import glob
from sklearn.model_selection import train_test_split
import cv2
import os

seed = 42
np.random.seed(seed)

ruta = "C:/Users/benja/Desktop/ia/proyecto1-ia/dataset"

# Lista de imágenes excluyendo las que son máscaras
imagen_path = sorted([p for p in glob.glob(ruta + "/*.jpg") if "_expert" not in p])

# Partición de datos por imagen
entrenamiento_val, test = train_test_split(imagen_path, test_size=0.2, random_state=seed)
entrenamiento, val = train_test_split(entrenamiento_val, test_size=0.25, random_state=seed)
print("Entrenamiento:", len(entrenamiento), "Validación:", len(val), "Test:", len(test))

def carga_imagen(paths):
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

train_images, train_masks = carga_imagen(entrenamiento)
val_images, val_masks = carga_imagen(val)
test_images, test_masks = carga_imagen(test)

def normalize(images):
    return [img / 255.0 for img in images]

train_images = normalize(train_images)
val_images = normalize(val_images)
test_images = normalize(test_images)

# Corrección de iluminación (opcional)
def corregir_iluminacion(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# Muestreo equilibrado de píxeles para entrenamiento
def muestreo_equilibrado(imagenes, mascaras, n=10000):
    lesion = []
    no_lesion = []
    for img, mask in zip(imagenes, mascaras):
        lesion_idx = np.where(mask == 1)
        no_lesion_idx = np.where(mask == 0)
        lesion.extend(list(zip(img[lesion_idx], mask[lesion_idx])))
        no_lesion.extend(list(zip(img[no_lesion_idx], mask[no_lesion_idx])))
    lesion = lesion[:n//2]
    no_lesion = no_lesion[:n//2]
    datos = lesion + no_lesion
    np.random.shuffle(datos)
    X, y = zip(*datos)
    return np.array(X), np.array(y)

X_entrenamiento, y_entrenamiento = muestreo_equilibrado(train_images, train_masks)

print("✓ Datos cargados, particionados y preprocesados correctamente")