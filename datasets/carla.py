import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
import random
import cv2


def rot_from_origin(c2w, rotation=10):
    rot = c2w[:3, :3]
    pos = c2w[:3, -1:]
    rot_mat = get_rotation_matrix(rotation)
    pos = torch.mm(rot_mat, pos)
    rot = torch.mm(rot_mat, rot)
    c2w = torch.cat((rot, pos), -1)
    return c2w


def get_rotation_matrix(rotation):
    phi = rotation * (np.pi / 180.0)
    x = np.random.uniform(-phi, phi)
    y = np.random.uniform(-phi, phi)
    z = np.random.uniform(-phi, phi)

    rot_x = torch.Tensor(
        [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
    )
    rot_y = torch.Tensor(
        [[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]]
    )
    rot_z = torch.Tensor(
        [
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1],
        ]
    )
    rot_mat = torch.mm(rot_x, torch.mm(rot_y, rot_z))
    return rot_mat


TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
    )
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    """
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    """
    assert (
        R1.shape[-1] == 3
        and R2.shape[-1] == 3
        and R1.shape[-2] == 3
        and R2.shape[-2] == 3
    )
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1)
            / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )


def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    num_select=4,
    tar_id=-1,
    angular_dist_method="vector",
    scene_center=(0, 0, 0),
):
    """
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    """
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == "matrix":
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3]
        )
    elif angular_dist_method == "vector":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == "dist":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception("unknown angular distance calculation method!")

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids

def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.03
    # radii = 0
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def read_poses(transform_file, idx=None):
    with open(transform_file, "r") as f:
        data = json.load(f)

    focal = data["fl_x"]
    img_wh = (data['w'], data['h'])
    all_c2w = []
    img_paths = []
    
    if idx is None:
        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"])
            # all_c2w.append(convert_pose_PD_to_NeRF(c2w))
            all_c2w.append((c2w))
            img_paths.append(os.path.join(os.path.dirname(transform_file), frame['file_path']))

    else:
        if isinstance(idx,int):
            idx = [idx]
        for idx_ in idx:
            frame = data["frames"][idx_]
            c2w = np.array(frame["transform_matrix"])
            # all_c2w.append(convert_pose_PD_to_NeRF(c2w))
            all_c2w.append((c2w))
            img_paths.append(os.path.join(os.path.dirname(transform_file), frame['file_path']))

    all_c2w = np.array(all_c2w)
    all_c2w[:, 0:3, 1:3] *= -1
    all_c2w[:, :3, 3] *= 0.1

    rot_y = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])

    all_c2w = np.matmul(rot_y, all_c2w) #(N,4,4)

    rot = np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]]).T

    rot = rot @ np.array([[-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]).T
    
    all_c2w = np.matmul(all_c2w, rot)

    return all_c2w, focal, img_wh, img_paths


class Carla(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(640, 480),
        white_back=False,
        model_type="Vanilla",
        eval_inference=None,
        optimize=None,
        encoder_type="resnet",
        contract=True,
        finetune_lpips=False,
        dataset_config=None
    ):
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.white_back = white_back
        self.base_dir = root_dir
        self.eval_inference = eval_inference
        self.optimize = optimize
        self.encoder_type = encoder_type
        self.contract = contract
        self.finetune_lpips = finetune_lpips
        self.data = []
        if len(dataset_config) == 1:
            dataset_config = dataset_config[0]
        self.dataset_config = dataset_config

        self.num_src_views = 6

        if dataset_config["town"] == "all":
            towns = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
        else:
            towns = dataset_config["town"]
        for town in towns:
            if dataset_config["weather"] == "all":
                weathers = [folder for folder in os.listdir(os.path.join(root_dir, town)) if os.path.isdir(os.path.join(root_dir, town, folder))]
            else:
                weathers = dataset_config["weather"]
            for weather in weathers:
                if dataset_config["vehicle"] == "all":
                    vehicles = [folder for folder in os.listdir(os.path.join(root_dir, town, weather)) if os.path.isdir(os.path.join(root_dir, town, weather, folder))]
                else:
                    vehicles = dataset_config["vehicle"]
                for vehicle in vehicles:
                    if dataset_config["spawn_point"] == ["all"]:
                        spawn_points = [folder for folder in os.listdir(os.path.join(root_dir, town, weather, vehicle)) if "spawn_point_" in folder]
                    else:
                        spawn_points = [f"spawn_point_{i}" for i in dataset_config["spawn_point"]]
                    for spawn_point in spawn_points:
                        if dataset_config["step"] == ["all"]:
                            steps = [folder for folder in os.listdir(os.path.join(root_dir, town, weather, vehicle, spawn_point)) if "step_" in folder]
                        else:
                            steps = [f"step_{i}" for i in dataset_config["step"]]
                        for step in steps:
                            self.data.append(self.get_path(town, weather, vehicle, spawn_point, step))

        # for multi scene training
        if self.encoder_type == "resnet":
            self.img_transform = T.Compose(
                [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            # for custom CNN MVS nerf style
            self.img_transform = T.Compose(
                [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )

        if self.encoder_type == "resnet":
            # roughly 10 epochs to see one full data i.e. 210*240*100*25
            self.samples_per_epoch = 9600
            # self.samples_per_epoch = 1875
        else:
            self.samples_per_epoch = 9600
            # self.samples_per_epoch = 1875
            # back to 10 epochs to see one full data due to increase sampling size i.e. 128 and 256 for coarse and fine
            # roughly 2 epochs to see one full data i.e. 210*240*100*25

        # self.samples_per_epoch = 1000
        #
        w, h = self.img_wh
        if self.eval_inference is not None:
            # num = 3
            num = 99
            # num = 40
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])

        self.model_type = model_type
        self.near = 0.0
        self.far = 3.0

    def get_path(self, town, weather, vehicle, spawn_point, step):
        data_path = os.path.join(self.base_dir, town, weather, vehicle, spawn_point, step)
        data = dict(
            town = town,
            weather = weather,
            vehicle = vehicle,
            spawn_point = spawn_point,
            step = step,
            nuscenes = os.path.join(data_path, "nuscenes"),
            sphere = os.path.join(data_path, "sphere"),
        )

        return data

    def read_data(
        self,
        instance_dir,
        idx=None
    ):
        src_dir = os.path.join(instance_dir['sphere'], "transforms/transforms_ego_train.json")
        split = "test" if self.split =="val" else self.split
        superv_dir = os.path.join(instance_dir['sphere'], f"transforms/transforms_ego_{split}.json")


        src_c2w, src_focal, src_img_wh, src_img_paths = read_poses(src_dir, idx=np.random.choice(80, self.num_src_views, replace=False))
        superv_c2w, superv_focal, superv_img_wh, superv_img_paths = read_poses(superv_dir, idx=idx)
        

        # SOURCE IMGS
        src_imgs = [Image.open(src_img_path) for src_img_path in src_img_paths]

        src_c2w = torch.FloatTensor(src_c2w) #(6,4,4)
        src_c = np.array(src_img_wh) / 2.0 # 400,300



        # SUPERVISION IMGS
        w, h = self.img_wh # (400,300)

        superv_imgs = [Image.open(superv_img_path).resize((w, h), Image.LANCZOS) for superv_img_path in superv_img_paths]
        
        superv_focal *= w / superv_img_wh[0]  #200,  modify focal length to match size self.img_wh

        superv_pose = torch.FloatTensor(superv_c2w) 
        superv_c2w = superv_pose[:,:3, :4]      


        superv_directions = get_ray_directions(h, w, superv_focal)  # (h, w, 3)
        all_superv_rays_o = []
        all_superv_rays_d = []
        all_superv_view_dirs = []
        all_superv_radii = []

        for c2w in superv_c2w:
            rays_o, view_dirs, rays_d, radii = get_rays(
                superv_directions, c2w, output_view_dirs=True, output_radii=True
            )
            all_superv_rays_o.append(rays_o)
            all_superv_rays_d.append(rays_d)
            all_superv_view_dirs.append(view_dirs)
            all_superv_radii.append(radii)

        rays_o = torch.cat(all_superv_rays_o, dim=0)
        rays_d = torch.cat(all_superv_rays_d, dim=0)
        view_dirs = torch.cat(all_superv_view_dirs, dim=0)
        radii = torch.cat(all_superv_radii, dim=0)
        return (
            rays_o, # (120000, 3)
            view_dirs, # (120000, 3)
            rays_d, # (120000, 3)
            src_imgs, # 6 x PIL imgs,  800*600
            radii, # [0.0058] * 500
            src_c2w, # (6,4,4)
            torch.tensor(src_focal, dtype=torch.float32), #  (400)
            torch.tensor(src_c, dtype=torch.float32), # tensor([400., 300.])
            superv_imgs # [Pil img 400x300]
        )

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train":
            if self.optimize is not None:
                return 3
            else:
                return self.samples_per_epoch
            # return len(self.ids)
        elif self.split == "val":
            if self.eval_inference is not None:
                return len(self.data) * 99
                # return 3
            else:
                return len(self.data)
        else:
            if self.eval_inference is not None:
                return len(self.data) * 99
                # return 40
                # return 3
            else:
                return len(self.data)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            train_idx = random.randint(0, len(self.data) - 1)
            instance_dir = self.data[train_idx]

            # imgs = list()
            # poses = list()
            # focals = list()
            # all_c = list()

            # rays = list()
            # view_dirs = list()
            # rays_d = list()
            # rgbs = list()
            # radii = list()

            if self.encoder_type == "resnet":
                ray_batch_size = 800
            else:
                ray_batch_size = 800

            if self.optimize is not None or self.finetune_lpips:
                # ======================================================\n\n\n
                # #Load desitnation view data from one camera view
                # ======================================================\n\n\n
                # Load desitnation view data
                (
                    rays_o,
                    view_dirs,
                    rays_d,
                    src_imgs,
                    radii,
                    src_poses,
                    src_focal,
                    c,
                    superv_imgs
                ) = self.read_data(instance_dir, idx=torch.randint(0,80).item())

                H, W, _ = np.array(superv_imgs[0]).shape
                camera_radii = torch.FloatTensor(radii)
                cam_rays = torch.FloatTensor(rays_o)
                cam_view_dirs = torch.FloatTensor(view_dirs)
                cam_rays_d = torch.FloatTensor(rays_d)

                img_gt = Image.fromarray(np.uint8(superv_imgs[0]))
                img_gt = T.ToTensor()(img_gt)
                rgbs = img_gt.permute(1, 2, 0).flatten(0, 1)[...,:2]

                radii = camera_radii.view(-1)
                rays = cam_rays.view(-1, cam_rays.shape[-1])
                rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
                view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

                imgs = []
                for img in src_imgs:
                    img = Image.fromarray(np.uint8(img)[...,:3])
                    img = T.ToTensor()(img)
                    imgs.append(self.img_transform(img))
                src_imgs = torch.stack(imgs, dim=0)

                patch = self.finetune_lpips
                if patch:
                    # select 64 by 64 patch for LPIPS loss
                    width = self.img_wh[0]
                    height = self.img_wh[1]
                    x = np.random.randint(0, height - 30 + 1)
                    y = np.random.randint(0, width - 30 + 1)
                    rgbs = rgbs.view(height, width, 3)[
                        x : x + 30, y : y + 30, :
                    ].reshape(-1, 3)
                    nocs_2ds = nocs_2ds.view(height, width, 3)[
                        x : x + 30, y : y + 30, :
                    ].reshape(-1, 3)
                    masks = masks.view(height, width)[x : x + 30, y : y + 30].reshape(
                        -1, 1
                    )
                    radii = radii.view(height, width)[x : x + 30, y : y + 30].reshape(
                        -1, 1
                    )
                    rays = rays.view(height, width, 3)[
                        x : x + 30, y : y + 30, :
                    ].reshape(-1, 3)
                    rays_d = rays_d.view(height, width, 3)[
                        x : x + 30, y : y + 30, :
                    ].reshape(-1, 3)
                    view_dirs = view_dirs.view(height, width, 3)[
                        x : x + 30, y : y + 30, :
                    ].reshape(-1, 3)
                else:
                    pix_inds = torch.randint(0, H * W, (ray_batch_size,))
                    rgbs = rgbs[pix_inds, ...]
                    nocs_2ds = nocs_2ds[pix_inds]
                    masks = masks[pix_inds]
                    radii = radii[pix_inds]
                    rays = rays[pix_inds]
                    rays_d = rays_d[pix_inds]
                    view_dirs = view_dirs[pix_inds]

            else:
                # ======================================================\n\n\n
                # #Load desitnation view data from 20 dest views
                # ======================================================\n\n\n                

                (
                    rays_o, # (2400000,3)
                    view_dirs, # (2400000,3)
                    rays_d, # (2400000,3)
                    src_imgs,  # [PIL img 800x600] * 6
                    radii, # (2400000)
                    src_poses, # (6,4,4)
                    src_focal, #(), 400.
                    c, # [400,300]
                    superv_imgs #[PIL img 400x300] * 20
                ) = self.read_data(instance_dir, idx=np.random.choice(79, 20, replace=False))

                H, W, _ = np.array(superv_imgs[0]).shape
                camera_radii = torch.FloatTensor(radii)
                cam_rays = torch.FloatTensor(rays_o)
                cam_view_dirs = torch.FloatTensor(view_dirs)
                cam_rays_d = torch.FloatTensor(rays_d)
                rgbs = []
                for superv_img in superv_imgs:
                    img_gt = Image.fromarray(np.uint8(superv_imgs[0]))
                    img_gt = T.ToTensor()(img_gt)
                    rgb = img_gt.permute(1, 2, 0).flatten(0, 1)[...,:3]
                    rgbs.append(rgb)

                radii = camera_radii.view(-1)
                rays = cam_rays.view(-1, cam_rays.shape[-1])
                rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
                view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])
                rgbs = torch.stack(rgbs) # (20, 120000,3)
                rgbs = rgbs.view(-1, 3) # (2400000, 3) [0->1]

                imgs = []
                for img in src_imgs:
                    img = Image.fromarray(np.uint8(img)[...,:3])
                    img = T.ToTensor()(img)
                    imgs.append(self.img_transform(img))
                src_imgs = torch.stack(imgs, dim=0) #(6,3,600,800)

                pix_inds = torch.randint(
                    0, len(rgbs), (ray_batch_size,)
                )
                rgbs = rgbs.reshape(-1, 3)[pix_inds, ...]
                radii = radii.reshape(-1, 1)[pix_inds]
                rays = rays.reshape(-1, 3)[pix_inds]
                rays_d = rays_d.reshape(-1, 3)[pix_inds]
                view_dirs = view_dirs.reshape(-1, 3)[pix_inds]

            if self.model_type == "Vanilla":
                sample = {
                    "src_imgs": src_imgs,
                    "rays": rays,
                    "rgbs": rgbs,
                }
            else:
                sample = {}
                sample["src_imgs"] = src_imgs # (6,3,600,800) [-1 -> 1]
                sample["src_poses"] = src_poses #(6,4,4)
                sample["src_focal"] = src_focal.repeat(self.num_src_views) # [400] * 6
                sample["src_c"] = c.repeat(self.num_src_views,1) #(6,2)
                sample["rays_o"] = rays #(500,3)
                sample["rays_d"] = rays_d #(500,3)
                sample["viewdirs"] = view_dirs #(500,3)
                sample["target"] = rgbs #(500,3)
                sample["radii"] = radii #(500,1)
                sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = torch.zeros_like(sample["rays_o"])

            return sample

        # elif self.split == 'val': # create data for each image separately
        elif self.split == "val":
            if self.eval_inference is not None:
                instance_dir = self.data[0]
            else:
                instance_dir = self.data[idx]

            # Load destination view data
            (
                rays_o,
                view_dirs,
                rays_d,
                src_imgs,
                radii,
                src_poses,
                src_focal,
                src_c,
                superv_imgs
            ) = self.read_data(instance_dir, idx=np.random.randint(0,19)) # random view

            H, W, _ = np.array(superv_imgs[0]).shape
            camera_radii = torch.FloatTensor(radii)
            cam_rays = torch.FloatTensor(rays_o)
            cam_view_dirs = torch.FloatTensor(view_dirs)
            cam_rays_d = torch.FloatTensor(rays_d)

            img_gt = Image.fromarray(np.uint8(superv_imgs[0]))
            img_gt = T.ToTensor()(img_gt)
            rgbs = img_gt.permute(1, 2, 0).flatten(0, 1)[...,:3]

            radii = camera_radii.view(-1)
            rays = cam_rays.view(-1, cam_rays.shape[-1])
            rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
            view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

            imgs = []
            for img in src_imgs:
                img = Image.fromarray(np.uint8(img)[...,:3])
                img = T.ToTensor()(img)
                imgs.append(self.img_transform(img))
            src_imgs = torch.stack(imgs, dim=0) # (6,3,600,800)


            if self.model_type == "Vanilla":
                sample = {
                    "src_imgs": src_imgs,
                    "rays": rays,
                    "rgbs": rgbs,
                }
            else:
                sample = {}
                sample["src_imgs"] = src_imgs # (6,3,600,800) [-1 -> 1]
                sample["src_poses"] = src_poses # (6,4,4)
                sample["src_focal"] = src_focal.repeat(self.num_src_views) #(6)
                sample["src_c"] = src_c.repeat(self.num_src_views,1) # (6,2)
                sample["rays_o"] = rays # (120000,3)
                sample["rays_d"] = rays_d # (120000,3)
                sample["viewdirs"] = view_dirs # (120000,3)
                sample["target"] = rgbs # (120000,3) [0->1]
                sample["radii"] = radii
                sample["multloss"] = torch.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = torch.zeros_like(sample["rays_o"])

            return sample

