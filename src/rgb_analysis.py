import numpy as np
import matplotlib.pyplot as plt

def analizar_canales_rgb(imagenes, mascaras):
    """Extrae píxeles de lesión y no-lesión para análisis"""
    lesion_pixels = []
    no_lesion_pixels = []
    
    for img, mask in zip(imagenes, mascaras):
        lesion_coords = np.where(mask == 1)
        no_lesion_coords = np.where(mask == 0)
        
        lesion_rgb = img[lesion_coords]
        no_lesion_rgb = img[no_lesion_coords]
        
        lesion_pixels.extend(lesion_rgb)
        no_lesion_pixels.extend(no_lesion_rgb)
    
    return np.array(lesion_pixels), np.array(no_lesion_pixels)

def mostrar_estadisticos_rgb(lesion_pixels, no_lesion_pixels):
    """Muestra estadísticos por canal RGB"""
    canales = ['R', 'G', 'B']
    print(f"\nEstadísticos por canal:")
    print("-" * 60)

    for i, canal in enumerate(canales):
        lesion_canal = lesion_pixels[:, i]
        no_lesion_canal = no_lesion_pixels[:, i]
        
        print(f"\nCanal {canal}:")
        print(f"  Lesión    - Media: {np.mean(lesion_canal):.4f}, Std: {np.std(lesion_canal):.4f}")
        print(f"  No-lesión - Media: {np.mean(no_lesion_canal):.4f}, Std: {np.std(no_lesion_canal):.4f}")

def crear_histogramas_rgb(lesion_pixels, no_lesion_pixels, filename='histogramas_rgb.png'):
    """Crea histogramas para cada canal RGB"""
    canales = ['R', 'G', 'B']
    colors = ['red', 'green', 'blue']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (canal, color) in enumerate(zip(canales, colors)):
        lesion_canal = lesion_pixels[:, i]
        no_lesion_canal = no_lesion_pixels[:, i]
        
        axes[i].hist(no_lesion_canal, bins=50, alpha=0.7, label='No-lesión', color='lightblue', density=True)
        axes[i].hist(lesion_canal, bins=50, alpha=0.7, label='Lesión', color=color, density=True)
        axes[i].set_title(f'Histograma Canal {canal}')
        axes[i].set_xlabel(f'Intensidad {canal}')
        axes[i].set_ylabel('Densidad')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def analisis_completo_rgb(train_images, train_masks):
    """Función principal para análisis completo RGB"""
    # Extraer píxeles de entrenamiento
    lesion_pixels, no_lesion_pixels = analizar_canales_rgb(train_images, train_masks)

    print(f"\nPíxeles extraídos:")
    print(f"Lesión: {len(lesion_pixels):,}")
    print(f"No-lesión: {len(no_lesion_pixels):,}")

    # Mostrar estadísticos
    mostrar_estadisticos_rgb(lesion_pixels, no_lesion_pixels)
    
    # Crear histogramas
    crear_histogramas_rgb(lesion_pixels, no_lesion_pixels)
    
    print("\n✓ Análisis de histogramas y estadísticos RGB completado")
    
    return lesion_pixels, no_lesion_pixels