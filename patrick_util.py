import numpy as np
import open3d as o3d


def get_o3d_sphere(color=[0.3, 1.0, 0.3], pos=[0, 0, 0], radius=0.06):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color(color)
    mean = np.asarray(mesh_sphere.vertices).mean(axis=0)
    diff = np.asarray(pos) - mean
    mesh_sphere.translate(diff)
    return mesh_sphere


def set_camera_extrinsic(vis, transform=np.eye(4)):
    """
    :param vis: Open3D visualizer object
    :param transform: 4x4 numpy defining a rigid transform where the camera should go
    """

    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = transform
    ctr.convert_from_pinhole_camera_parameters(cam)

