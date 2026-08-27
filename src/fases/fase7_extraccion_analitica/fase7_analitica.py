import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path

def ejecutar_fase_7(directorio_io):
    print("[Fase 7] Iniciando Extracción Analítica de Residuos...")
    
    dir_salida = Path(directorio_io) / "fase7_analitica"
    dir_salida.mkdir(parents=True, exist_ok=True)
    
    # Rutas de fases anteriores
    dir_fase3 = Path(directorio_io) / "fase3_3d"
    ruta_metadata_depth = Path(directorio_io) / "fase5_profundidad" / "metadata_depth.json"
    
    # Cargar metadatos métricos si existen para escalar posiciones
    escala_metrica = 1.0
    if ruta_metadata_depth.exists():
        with open(ruta_metadata_depth, "r") as f:
            data_depth = json.load(f)
            escala_metrica = data_depth.get("factor_escala", 1.0)

    reporte_analitico = {
        "total_residuos_detectados": 0,
        "inventario": []
    }

    archivos_obj = list(dir_fase3.glob("*.obj"))
    reporte_analitico["total_residuos_detectados"] = len(archivos_obj)

    for i, archivo_obj in enumerate(archivos_obj):
        # Cargar malla 3D con Open3D para análisis volumétrico
        mesh = o3d.io.read_triangle_mesh(str(archivo_obj))
        
        # Calcular propiedades geométricas básicas
        volumen = mesh.get_volume() * (escala_metrica ** 3)
        centro = mesh.get_center() * escala_metrica
        bbox = mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_extents() * escala_metrica
        
        item_info = {
            "id": i,
            "archivo": archivo_obj.name,
            "volumen_estimado_m3": float(volumen),
            "centro_espacial_xyz": [float(centro[0]), float(centro[1]), float(centro[2])],
            "dimensiones_cm": [
                float(extents[0] * 100), 
                float(extents[1] * 100), 
                float(extents[2] * 100)
            ]
        }
        reporte_analitico["inventario"].append(item_info)

    # Guardar reporte analítico global
    ruta_reporte = dir_salida / "reporte_residuos.json"
    with open(ruta_reporte, "w") as f:
        json.dump(reporte_analitico, f, indent=4)

    print(f"[Fase 7] Análisis completado. Reporte guardado en: {ruta_reporte}")

if __name__ == "__main__":
    directorio_base = "outputs"
    ejecutar_fase_7(directorio_base)