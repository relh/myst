#!/usr/bin/env python
# -*- coding: utf-8 -*-

# First, I'll load the provided point cloud file to understand its structure and content.
import open3d as o3d

# Load the point cloud file
file_path = './gaussians.ply'
point_cloud = o3d.io.read_point_cloud(file_path)

# Check the basic information of the point cloud
point_cloud_info = {
    "Number of points": len(point_cloud.points),
    "Has colors": point_cloud.colors is not None,
    "Has normals": point_cloud.normals is not None
}
point_cloud_info

import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

rr.init("rerun_example_my_data", spawn=True)

#positions = np.zeros((10, 3))
#positions[:,0] = np.linspace(-10,10,10)

#colors = np.zeros((10,3), dtype=np.uint8)
#colors[:,0] = np.linspace(0,255,10)

rr.log(
    "my_points",
    rr.Points3D(point_cloud.points, colors=point_cloud.normals, radii=0.01)
)
