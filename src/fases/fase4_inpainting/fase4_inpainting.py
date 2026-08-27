import os
import cv2
import numpy as np
import torch
import gc
from simple_lama_inpainting import SimpleLama
from PIL import Image

def limpiar_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ejecutar_fase_3(ruta_imagen, directorio_mascaras, ruta_salida):
    print("\n[Fase 3] Iniciando Inpainting con LaMa...")
    
    # 1. Buscar todas las máscaras generadas en la Fase 2
    archivos_mascara = [f for f in os.listdir(directorio_mascaras) if f.endswith('.png') and "mask" in f]
    
    if not archivos_mascara:
        print("[Fase 3] No se encontraron máscaras para procesar. Abortando.")
        return
    
    print(f"[Fase 3] Fusionando {len(archivos_mascara)} máscaras...")
    
    # 2. Cargar la imagen original y crear una máscara vacía (negra) del mismo tamaño
    imagen_cv = cv2.imread(ruta_imagen)
    alto, ancho = imagen_cv.shape[:2]
    mascara_global = np.zeros((alto, ancho), dtype=np.uint8)
    
    # 3. Sumar todas las máscaras en una sola
    for archivo in archivos_mascara:
        ruta_mask = os.path.join(directorio_mascaras, archivo)
        mask_individual = cv2.imread(ruta_mask, cv2.IMREAD_GRAYSCALE)
        # Dilatamos un poco la máscara para asegurar que los bordes del objeto se borren bien
        kernel = np.ones((5,5), np.uint8)
        mask_dilatada = cv2.dilate(mask_individual, kernel, iterations=2)
        mascara_global = cv2.bitwise_or(mascara_global, mask_dilatada)
    
    # Guardamos la máscara global para diagnóstico
    ruta_mask_global = os.path.join(directorio_mascaras, "mascara_global_lama.jpg")
    cv2.imwrite(ruta_mask_global, mascara_global)
    
    # 4. Convertir a formato PIL para LaMa
    img_pil = Image.fromarray(cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mascara_global).convert('L')
    
    # 5. Ejecutar LaMa
    print("[Fase 3] Cargando modelo LaMa en GPU y borrando objetos...")
    lama = SimpleLama()
    imagen_limpia = lama(img_pil, mask_pil)
    
    # 6. Guardar resultado
    imagen_limpia.save(ruta_salida)
    print(f"[Fase 3] Imagen limpia generada con éxito en: {ruta_salida}")
    
    # Purgar memoria
    del lama
    limpiar_vram()

if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.abspath(os.path.join(ruta_actual, "../../.."))
    
    ruta_entrada = os.path.join(ruta_raiz, "outputs", "prep_foto_rio.jpg")
    dir_mascaras = os.path.join(ruta_raiz, "outputs")
    ruta_salida = os.path.join(ruta_raiz, "outputs", "fondo_limpio_lama.jpg")
    
    ejecutar_fase_3(ruta_entrada, dir_mascaras, ruta_salida)