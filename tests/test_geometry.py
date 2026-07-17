import numpy as np

from conceptgraphs_uav.geometry import project_depth_to_world


def test_image_down_projects_to_ned_down():
    depth = np.full((3, 3), 10.0, dtype=np.float32)
    points = project_depth_to_world(
        depth,
        position=np.zeros(3, dtype=np.float32),
        quaternion_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
        fov_degrees=90,
        stride=1,
    )
    # Row-major point order: bottom-center is index 7. In NED it must have +Z.
    assert points[7, 2] > 0
    assert points[1, 2] < 0
