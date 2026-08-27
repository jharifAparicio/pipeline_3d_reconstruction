import os
import cv2
import numpy as np
import torch
import gc
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def limpiar_vram():
    """Fuerza la liberación absoluta de la memoria de video."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ejecutar_fase_2(input_path: str, output_dir: str, sam_checkpoint: str):
    print("\n[Fase 2] Iniciando Detección Semántica y Segmentación...")
    
    # --- PARTE A: YOLO-WORLD (Detección) ---
    print("[Fase 2] Cargando YOLO-World en GPU...")
    # Ultralytics descargará yolov8s-world.pt automáticamente la primera vez
    model_yolo = YOLO("yolov8s-worldv2.pt")
    
    # Definimos qué queremos buscar (Open-Vocabulary)
    clases_objetivo = [
        # Botellas y envases líquidos
        "plastic bottle",
        "clear plastic bottle",
        "glass bottle",
        "green glass bottle",
        "aluminum can",
        "plastic jug",
        "jerrycan",

        # Contenedores grandes y barriles
        "plastic barrel",
        "blue barrel",
        "oil drum",
        "plastic container",
        "bucket",

        # Escombros sueltos y empaques
        "styrofoam",          
        "foam debris",
        "trash bag",
        "black garbage bag",
        "plastic wrapper",
        "food packaging",
        "cardboard",

        # Caucho y otros
        "tire",
        "rubber wheel",
        "discarded shoe",
        "fabric rag"
    ]
    model_yolo.set_classes(clases_objetivo)
    
    print("[Fase 2] Ejecutando inferencia YOLO...")
    resultados = model_yolo.predict(
        input_path, 
        conf=0.03, 
        iou=0.4, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if len(resultados) > 0:
        ruta_debug = os.path.join(output_dir, "yolo_debug.jpg")
        resultados[0].save(filename=ruta_debug)
        print(f"[Fase 2] Imagen de diagnóstico guardada en: {ruta_debug}")
    
    bboxes = []
    if len(resultados) > 0 and resultados[0].boxes is not None:
        # Extraemos las coordenadas [x_min, y_min, x_max, y_max]
        bboxes = resultados[0].boxes.xyxy.cpu().numpy()
        print(f"[Fase 2] Se detectaron {len(bboxes)} posibles residuos.")
    
    # ¡CRÍTICO! Destruimos YOLO antes de cargar SAM
    del model_yolo
    limpiar_vram()
    print("[Fase 2] YOLO purgado de la VRAM.")

    if len(bboxes) == 0:
        print("[Fase 2] No se encontró basura. Abortando segmentación.")
        return []

    # --- PARTE B: SAM 2D (Segmentación exacta) ---
    print("[Fase 2] Cargando SAM en GPU...")
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    # Preparamos la imagen para SAM (OpenCV usa BGR, SAM necesita RGB)
    imagen_bgr = cv2.imread(input_path)
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(imagen_rgb)

    rutas_mascaras = []
    
    print("[Fase 2] Generando máscaras individuales...")
    for i, box in enumerate(bboxes):
        # SAM predice la silueta basándose en el Bounding Box de YOLO
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mascara_binaria = (masks[0] * 255).astype(np.uint8)
        
        # Guardamos la máscara
        nombre_base = os.path.basename(input_path).split('.')[0]
        ruta_mascara = os.path.join(output_dir, f"{nombre_base}_mask_{i:03d}.png")
        cv2.imwrite(ruta_mascara, mascara_binaria)
        rutas_mascaras.append(ruta_mascara)
    
    print(f"[Fase 2] {len(rutas_mascaras)} máscaras guardadas en {output_dir}")
    
    # Destruimos SAM y limpiamos nuevamente
    del predictor
    del sam
    limpiar_vram()
    print("[Fase 2] SAM purgado de la VRAM.")
    
    return rutas_mascaras

if __name__ == "__main__":
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.abspath(os.path.join(ruta_actual, "../../.."))
    
    # Usamos la imagen preprocesada que generaste en la Fase 1
    ruta_entrada = os.path.join(ruta_raiz, "outputs", "prep_foto_rio.jpg")
    ruta_salida = os.path.join(ruta_raiz, "outputs")
    ruta_pesos_sam = os.path.join(ruta_raiz, "weights", "sam_vit_b_01ec64.pth")
    
    if not os.path.exists(ruta_pesos_sam):
        print("ERROR: Falta el archivo de pesos de SAM en la carpeta weights/")
    else:
        ejecutar_fase_2(ruta_entrada, ruta_salida, ruta_pesos_sam)