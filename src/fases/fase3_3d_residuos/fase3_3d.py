import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
import trimesh
import gc

# --- RUTAS PRINCIPALES ---
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.abspath(os.path.join(ruta_actual, "../../.."))

# Añadir la carpeta TripoSR al path para poder importar sus módulos
ruta_triposr = os.path.join(ruta_raiz, "TripoSR")
sys.path.append(ruta_triposr)

# Ahora sí podemos importar tsr
from tsr.system import TSR

def procesar_recorte_transparente(ruta_img, ruta_mask):
    # Cargar imagen y máscara
    img = cv2.imread(ruta_img)
    mask = cv2.imread(ruta_mask, cv2.IMREAD_GRAYSCALE)
    
    # Encontrar el Bounding Box del objeto
    coordenadas = cv2.findNonZero(mask)
    if coordenadas is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coordenadas)
    
    # Recortar imagen y máscara
    img_crop = img[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    
    # Convertir a RGBA y aplicar transparencia
    img_rgba = cv2.cvtColor(img_crop, cv2.COLOR_BGR2BGRA)
    img_rgba[:, :, 3] = mask_crop  # El canal Alpha es la máscara
    
    # Hacer la imagen cuadrada (padding) para que TripoSR la entienda mejor
    tamaño_max = max(w, h)
    img_cuadrada = np.zeros((tamaño_max, tamaño_max, 4), dtype=np.uint8)
    
    # Centrar el recorte en la imagen cuadrada
    offset_x = (tamaño_max - w) // 2
    offset_y = (tamaño_max - h) // 2
    img_cuadrada[offset_y:offset_y+h, offset_x:offset_x+w] = img_rgba
    
    # Convertir a formato PIL (RGBA)
    img_pil = Image.fromarray(cv2.cvtColor(img_cuadrada, cv2.COLOR_BGRA2RGBA))
    return img_pil

def ejecutar_fase_3(ruta_imagen, directorio_io):
    print("\n[Fase 3] Iniciando Reconstrucción 3D con TripoSR...")
    
    archivos_mascara = [f for f in os.listdir(directorio_io) if f.endswith('.png') and "mask" in f]
    
    if not archivos_mascara:
        print("[Fase 3] No hay máscaras para procesar.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Fase 3] Cargando modelo TripoSR en {device}...")
    
    # Inicializar TripoSR
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt"
    )
    model.to(device)
    model.renderer.set_chunk_size(8192)
    #model.model.to(device)

    # Procesar cada residuo detectado
    for i, archivo in enumerate(archivos_mascara):
        ruta_mask = os.path.join(directorio_io, archivo)
        print(f"[Fase 3] Procesando objeto {i+1}/{len(archivos_mascara)}: {archivo}")
        
        imagen_pil = procesar_recorte_transparente(ruta_imagen, ruta_mask)
        
        if imagen_pil is None:
            continue
            
        # Inferencia 3D
        with torch.no_grad():
            # Asegurar que la imagen sea RGB (eliminar canal alfa si existe)
            if imagen_pil.mode != "RGB":
                imagen_pil = imagen_pil.convert("RGB")
            scene_codes = model(imagen_pil, device=device)
            mesh = model.extract_mesh(scene_codes, has_vertex_color=True)[0]
        
        # Guardar como .obj
        ruta_obj = os.path.join(directorio_io, f"residuo_3d_{i+1}.obj")
        mesh.export(ruta_obj)
        print(f"  -> Guardado en: {ruta_obj}")

    print("[Fase 3] Reconstrucción 3D completada.")
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    ruta_entrada = os.path.join(ruta_raiz, "outputs", "prep_foto_rio.jpg")
    directorio_io = os.path.join(ruta_raiz, "outputs")
    
    ejecutar_fase_3(ruta_entrada, directorio_io)