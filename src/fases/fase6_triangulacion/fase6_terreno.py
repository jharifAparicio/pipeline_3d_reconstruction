import open3d as o3d
import numpy as np
import cv2
import json
import os

def generar_malla_terreno(ruta_rgb, ruta_depth, ruta_focal, ruta_salida):
    print("Iniciando Fase 6: Triangulación de Malla de Terreno (CPU)...")
    
    # 1. Cargar profundidad primero para obtener sus dimensiones exactas
    depth_np = np.load(ruta_depth)
    h, w = depth_np.shape

    # 2. Cargar imagen RGB y reescalarla al tamaño exacto del mapa de profundidad
    color_img = cv2.imread(ruta_rgb)
    color_img = cv2.resize(color_img, (w, h), interpolation=cv2.INTER_AREA)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # 3. Leer la distancia focal estimada por Depth Pro
    with open(ruta_focal, 'r') as f:
        meta_depth = json.load(f)
        focallength_px = meta_depth['focallength_px']
    
    print(f"Usando distancia focal estimada por Depth Pro: {focallength_px:.2f} px")

    # El centro óptico (cx, cy) se asume en el centro exacto de la imagen
    cx, cy = w / 2.0, h / 2.0

    # 3. Convertir a estructuras de Open3D
    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, 
        depth_o3d, 
        depth_scale=1.0, 
        depth_trunc=50.0, 
        convert_rgb_to_intensity=False
    )

    # 4. Inyectar los intrínsecos dinámicos de Depth Pro
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, 
        height=h, 
        fx=focallength_px, 
        fy=focallength_px, 
        cx=cx, 
        cy=cy
    )

    # 5. Generar Nube de Puntos
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 6. Estimar normales y reconstruir la malla (Poisson)
    print("Calculando normales y triangulando superficie...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 7. Exportar
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    o3d.io.write_triangle_mesh(ruta_salida, mesh)
    
    print(f"Malla de terreno exportada con éxito en: {ruta_salida}")

if __name__ == "__main__":
    generar_malla_terreno(
        ruta_rgb="inputs/foto_rio.jpg",
        ruta_depth="outputs/mapa_profundidad_metrico.npy",
        ruta_focal="outputs/metadata_depth.json",
        ruta_salida="outputs/fase6_terreno/Malla_Terreno.obj"
    )