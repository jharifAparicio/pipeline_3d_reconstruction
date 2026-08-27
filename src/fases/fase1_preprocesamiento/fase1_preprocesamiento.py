import os
import cv2
import numpy as np

def ejecutar_fase_1(input_path: str, output_dir: str, target_size: int = 1024) -> str:
    """
    Preprocesa la imagen original: redimensiona conservando la relación de aspecto
    y estandariza el espacio de color para que los modelos la procesen sin deformaciones.
    """
    print(f"\n[Fase 1] Iniciando preprocesamiento...")
    print(f"[Fase 1] Cargando imagen: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: No se encontró la imagen en {input_path}")
        
    # 1. Leer imagen (OpenCV la lee en formato BGR por defecto)
    img_bgr = cv2.imread(input_path)
    
    # 2. Calcular nueva resolución manteniendo la relación de aspecto exacta
    h, w = img_bgr.shape[:2]
    escala = target_size / max(h, w)
    
    # Si la imagen ya es más pequeña que 1024, no la escalamos para no perder calidad
    if escala < 1.0:
        nuevo_w = int(w * escala)
        nuevo_h = int(h * escala)
        
        # Usamos INTER_AREA porque reduce el tamaño sin emborronar los bordes (vital para YOLO y SAM)
        img_procesada = cv2.resize(img_bgr, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
        print(f"[Fase 1] Imagen redimensionada de {w}x{h} a {nuevo_w}x{nuevo_h}")
    else:
        img_procesada = img_bgr
        print(f"[Fase 1] La imagen ya está optimizada ({w}x{h}). Se mantiene tamaño original.")

    # 3. Guardar el tensor de salida
    nombre_archivo = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"prep_{nombre_archivo}")
    
    cv2.imwrite(output_path, img_procesada)
    print(f"[Fase 1] Tensor guardado con éxito en: {output_path}")
    
    # Devolvemos la ruta de la nueva imagen para que la Fase 2 la recoja automáticamente
    return output_path

if __name__ == "__main__":
    # Calculamos la ruta raíz del proyecto dinámicamente
    # __file__ es el script actual, subimos 3 niveles para llegar a pipeline_3d/
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.abspath(os.path.join(ruta_actual, "../../.."))
    
    ruta_entrada = os.path.join(ruta_raiz, "inputs", "foto_rio.jpg")
    ruta_salida = os.path.join(ruta_raiz, "outputs")
    
    os.makedirs(os.path.join(ruta_raiz, "inputs"), exist_ok=True)
    os.makedirs(ruta_salida, exist_ok=True)
    
    try:
        resultado = ejecutar_fase_1(ruta_entrada, ruta_salida)
    except Exception as e:
        print(e)