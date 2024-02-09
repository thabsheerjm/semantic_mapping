#!/usr/bin/env python3

import numpy as np 
import torch
import open3d as o3d
import cv2
import os
from PIL import Image

from argparse import ArgumentParser
import shutil
import torch

from torchvision import models, transforms
from torchvision.transforms import functional as F


class SemanticSegmentation:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0','deeplabv3_resnet50',pretrained=True)
    


    def read_calibration_data():
        pass

    

def read_calib_data():
    filepath = "calibration/calib.txt"
    file = open(filepath, "r")
    for i, row in enumerate(file):
        if i == 0:
            key, value = row.split(':')
            P = np.array([float(x) for x in value.split()]).reshape((3,4))
        if i == 1:
            key, value = row.split(':')
            K = np.array([float(x) for x in value.split()]).reshape((3,3))
        if i == 2:
            key, value = row.split(':')
            R0 = np.array([float(x) for x in value.split()]).reshape((3,3))
        if i == 3:
            key, value = row.split(':')
            Tr_cam_to_lidar = np.array([float(x) for x in value.split()]).reshape((3,4))
        if i == 4:
            key, value = row.split(':')
            D = np.array([float(x) for x in value.split()]).reshape((1,5))
    return P, K, R0, Tr_cam_to_lidar, D


def Projection_lidar_to_cam():
    P, K, R, Tr_cam_to_lidar, D = read_calib_data()
    R_cam_to_lidar = Tr_cam_to_lidar[:3,:3].reshape(3,3)
    t_cam_to_lidar = Tr_cam_to_lidar[:3,3].reshape(3,1)
    R_cam_to_lidar_inv = np.linalg.inv(R_cam_to_lidar)
    t_new = -np.dot(R_cam_to_lidar_inv, t_cam_to_lidar)
    Tr_lidar_to_cam = np.vstack((np.hstack((R_cam_to_lidar_inv, t_new)), np.array([0., 0., 0., 1.])))
    R_rect = np.eye(4)
    R_rect[:3, :3] = R.reshape(3, 3)
    P_ = P.reshape((3, 4))
    proj_mat = P_ @ R_rect @ Tr_lidar_to_cam
    return proj_mat

def generate_color_palette(num_classes):
    np.random.seed(42)  # For reproducible colors for each classes
    return np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

def visualize_pointcloud(pointcloud, count, save_path):
    xyz = pointcloud[:, 0:3]
    semantics = pointcloud[:,3:]
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(semantics)
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.capture_screen_image(os.path.join(save_path, f"{count}.png"))
    visualizer.destroy_window()


def project_lidar_on_image(P, lidar_pts, size):
    n = lidar_pts.shape[0]
    pts_3d = np.hstack((lidar_pts, np.ones((n, 1))))  # homogeneous co-ordinates
    pts_2d = np.dot(pts_3d, P.T)
    depth = pts_3d[:,2]
    depth[depth == 0] = -1e-6
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d = pts_2d[:, :2]
    inliers_idx = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < size[0]) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < size[1]))
    return pts_2d[inliers_idx], depth[inliers_idx], lidar_pts[inliers_idx]

def make_video(fps, path, video_file):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(path, fps=fps)
    clip.write_videofile(video_file)
    shutil.rmtree(path)

def make_directory(save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)


def semantics(img_path, model):
    """
    Performs semantic segmentation on the given image and returns the class predictions
    and an RGB image where each pixel's color corresponds to its class's color.
    """
    # Load and preprocess the input image
    input_image = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    model.eval()
    with torch.no_grad():
        output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
    
    # Get unique classes and generate a color palette
    unique_classes = torch.unique(output_predictions)
    num_classes = len(unique_classes)
    color_palette = generate_color_palette(num_classes)
    
    # Map each class to a color
    rgb_image = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    for i, cls in enumerate(unique_classes):
        rgb_image[output_predictions == cls] = color_palette[i]

    return output_predictions.cpu().numpy(), rgb_image


def main():
    #caluclate the transformation from LiDAR to camera
    p_lidar_to_cam = Projection_lidar_to_cam()

    save_path_img = os.path.join('Results', 'projected_clouds')
    make_directory(save_path_img)

    save_path_cloud = os.path.join('Results', 'painted_clouds')
    make_directory(save_path_cloud)

    rgb_folder_path = os.path.join( 'Data', 'rgb_data')
    pcd_folder_path = os.path.join( 'Data', 'pointcloud')
    img_list= os.listdir(rgb_folder_path)
    img_list.sort()
    pcd_list = os.listdir(pcd_folder_path)
    pcd_list.sort()

    img_paths = []
    pcd_paths = []

    for a in range(len(img_list)):

        img_paths.append(os.path.join(rgb_folder_path, img_list[a]))
        pcd_paths.append(os.path.join(pcd_folder_path, pcd_list[a]))

    count = 10000
    for i in range(len(img_list)):

        img_path = img_paths[i]
        img = cv2.imread(img_path)
        pcd = o3d.io.read_point_cloud(pcd_paths[i])
        pcd_array = np.asarray(pcd.points)

        # remove all the points from the point cloud that are behind the camera (means backside of the car which is -ve x axis)
        idx = pcd_array[:,0] >= 0 
        pcd_array= pcd_array[idx]
        #projet lidar points on RGB image
        pts_2D,depth, pts_3D_img = project_lidar_on_image(p_lidar_to_cam, pcd_array, (img.shape[1], img.shape[0]))
        N = pts_3D_img.shape[0]

        #predict semantic segmentation of RGB images using DeepLavV3+ pretrained model
        pred, semantic_rgb = semantics(img_path,model)
        cloud_color = np.zeros((N,3), dtype=np.float32)
        fused_img = img.copy()
        for j in range(pts_2D.shape[0]):
            if j >= 0:

                x = np.int32(pts_2D[j, 0])
                y = np.int32(pts_2D[j, 1])
                
                # get the color corresponsing to the label
                class_color = np.float64(semantic_rgb[y, x]) 
                #draw the point cloud projections on the image
                cv2.circle(fused_img, (x,y), 2, color=tuple(class_color), thickness=1)

                # assign color to point clouds
                cloud_color[j] = class_color/255.0

        # stacked_img = np.vstack((img,fused_img))
         # Ensure semantic_rgb is resized to match img dimensions if they don't already
        semantic_rgb_resized = cv2.resize(semantic_rgb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Stack the original image and the semantic segmentation image vertically
        stacked_img = np.vstack((img, semantic_rgb_resized))
        cv2.imwrite(save_path_img + "/" + str(count) + ".png",stacked_img)

        # colored point cloud
        semantic_pointcloud = np.hstack((pts_3D_img[:,:3], cloud_color))

        
        count+=1

    video_file = "semantic_map.mp4"
    make_video(10, save_path_img, video_file)

if __name__ == '__main__':
    main()