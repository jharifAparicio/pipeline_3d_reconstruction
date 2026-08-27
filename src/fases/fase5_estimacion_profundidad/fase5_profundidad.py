import os
import torch
import numpy as np
import cv2
import json
from PIL import Image
import depth_pro

def ejecutar_fase_4_metrico(ruta_imagen_limpia, ruta_salida_nps, ruta_salida_json):
    print("\n[Fase 4] Iniciando Estimación de Profundidad Métrica (Depth Pro)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Fase 4] Usando dispositivo: {device}")

    # 1. Cargar el modelo Depth Pro oficial y sus transformaciones
    print("[Fase 4] Cargando modelo Depth Pro en GPU...")
    model, transform = depth_pro.create_model_and_transforms(
        device=device, 
        precision=torch.float16
    )
    model.eval()

    # 2. Cargar y preprocesar la imagen limpia del fondo
    print("[Fase 4] Procesando imagen para métrica absoluta...")
    image, _, f_px = depth_pro.load_rgb(ruta_imagen_limpia)
    image_tensor = transform(image).to(device)

    # 3. Inferencia métrica (devuelve distancia real en metros)
    print("[Fase 4] Calculando matriz de profundidad en metros...")
    with torch.no_grad():
        prediction = model.infer(image_tensor, f_px=f_px)
        
    # Extraer el mapa de profundidad en metros y la distancia focal estimada
    depth_meters = prediction["depth"].detach().cpu().numpy()  # Valores en metros
    
    # Extraer focal y asegurar que sea un float estándar de Python para el JSON
    focallength_px = float(prediction["focallength_px"].detach().cpu().item() if isinstance(prediction["focallength_px"], torch.Tensor) else prediction["focallength_px"])
    
    print(f"[Fase 4] Distancia focal estimada: {focallength_px:.2f} px")
    print(f"[Fase 4] Rango métrico detectado: {depth_meters.min():.2f}m hasta {depth_meters.max():.2f}m")

    # 4. Guardar los datos métricos exactos en formato numpy (.npy)
    np.save(ruta_salida_nps, depth_meters)
    print(f"[Fase 4] Mapa métrico guardado con precisión absoluta en: {ruta_salida_nps}")

    # 5. Exportar metadatos (focal length) a JSON para la Fase 6
    metadatos = {
        "focallength_px": focallength_px
    }
    with open(ruta_salida_json, 'w') as f:
        json.dump(metadatos, f, indent=4)
    print(f"[Fase 4] Metadatos guardados en: {ruta_salida_json}")

    # 6. Guardar también una previsualización en escala de grises para diagnóstico visual
    depth_normalized = cv2.normalize(depth_meters, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_vis = depth_normalized.astype(np.uint8)
    ruta_vis = ruta_salida_nps.replace(".npy", "_vis.jpg")
    cv2.imwrite(ruta_vis, depth_vis)
    print(f"[Fase 4] Previsualización visual guardada en: {ruta_vis}")

    # Purgar VRAM
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.abspath(os.path.join(ruta_actual, "../../.."))
    
    ruta_entrada = os.path.join(ruta_raiz, "outputs", "fondo_limpio_lama.jpg")
    ruta_salida_metrica = os.path.join(ruta_raiz, "outputs", "mapa_profundidad_metrico.npy")
    ruta_salida_metadatos = os.path.join(ruta_raiz, "outputs", "metadata_depth.json")
    
    ejecutar_fase_4_metrico(ruta_entrada, ruta_salida_metrica, ruta_salida_metadatos)