#!/usr/bin/env python

import sys
sys.path
sys.path.append('/home/russell/git/vMAP/')

import torch.multiprocessing as mp
import shutil
from cfg import Config
import argparse
from functorch import vmap
import vis
import dataset
import matplotlib.pyplot as plt
import open3d
import utils
from vmap import *
import loss
import time



class IMapWrapper:
    def __init__(self, args):
        #############################################
        # init config
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.first_image = True

        log_dir = args.logdir
        config_file = args.config
        save_ckpt = args.save_ckpt
        os.makedirs(log_dir, exist_ok=True)  # saving logs
        shutil.copy(config_file, log_dir)

        # config params
        self.cfg = Config(config_file)
        self.n_sample_per_step = self.cfg.n_per_optim
        self.n_sample_per_step_bg = self.cfg.n_per_optim_bg

        self.cam_info = cameraInfo(self.cfg)

        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
            width=self.cfg.W,
            height=self.cfg.H,
            fx=self.cfg.fx,
            fy=self.cfg.fy,
            cx=self.cfg.cx,
            cy=self.cfg.cy)

        # init obj_dict
        self.obj_dict = {}   # only objs
        self.vis_dict = {}   # including bg
        self.obj_id = 0

        self.keyframe_counter = 0

        self.optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))],
                                           lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        self.latest_mesh = open3d.geometry.TriangleMesh()
        self.mesh_available = False
        # todo for meshing
        # size = 2
        # processes = []
        # for rank in range(size):
        #     p = mp.Process(target=self.meshing, args=())
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #         p.join()

    def getMesh(self):
        if self.mesh_available:
            return self.latest_mesh
        else:
            return None

    def meshing(self):

        if ((self.keyframe_counter % self.cfg.n_vis_iter) == 0 and (self.keyframe_counter >= 10)):
            print("DOING MESHING")
            for obj_id, obj_k in self.vis_dict.items():
                bound = obj_k.get_bound(self.intrinsic_open3d)
                if bound is None:
                    print("get bound failed obj ", obj_id)
                    continue
                adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//self.cfg.live_voxel_size+1, self.cfg.grid_dim))
                mesh = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=adaptive_grid_dim)
                if mesh is None:
                    print("meshing failed obj ", obj_id)
                    continue

                # save to dir
                # obj_mesh_output = os.path.join(log_dir, "scene_mesh")
                # os.makedirs(obj_mesh_output, exist_ok=True)
                # mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.obj".format(frame_id, str(obj_id))))

                # live vis
                self.latest_mesh = vis.trimesh_to_open3d(mesh)
                self.mesh_available = True
                # vis3d.add_geometry(open3d_mesh)
                # vis3d.add_geometry(bound)
                # # update vis3d
                # vis3d.poll_events()
                # vis3d.update_renderer()
        else:
            print(f"NOT MESHING{self.keyframe_counter} {self.cfg.n_vis_iter}")

        # with performance_measure("saving ckpt"):
        #     if save_ckpt and ((((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len - 1) or
        #                        (cfg.live_mode and time.time() - last_frame_time > cfg.keep_live_time)) and frame_id >= 10):
        #         for obj_id, obj_k in vis_dict.items():
        #             ckpt_dir = os.path.join(log_dir, "ckpt", str(obj_id))
        #             os.makedirs(ckpt_dir, exist_ok=True)
        #             bound = obj_k.get_bound(intrinsic_open3d)   # update bound
        #             obj_k.save_checkpoints(ckpt_dir, frame_id)
        #         # save current cam pose
        #         cam_dir = os.path.join(log_dir, "cam_pose")
        #         os.makedirs(cam_dir, exist_ok=True)
        #         torch.save({"twc": twc, }, os.path.join(cam_dir, "twc_frame_{}".format(frame_id) + ".pth"))

    def add_images_and_pose(self, rgb_image, depth_image, camera_pose):

        with performance_measure(f"Adding Images to Dictionary"):
            state = torch.ones_like(depth_image, dtype=torch.uint8)
            bbox = torch.tensor([0, depth_image.shape[0], 0, depth_image.shape[1]], dtype=torch.float32)

            if (self.first_image):
                self.first_image = False

                scene_obj = sceneObject(self.cfg, self.obj_id, rgb_image, depth_image, state,
                                        bbox, camera_pose, self.keyframe_counter)

                # scene_obj.init_obj_center(intrinsic_open3d, depth, state, twc)
                self.obj_dict.update({self.obj_id: scene_obj})
                self.vis_dict.update({self.obj_id: scene_obj})

                self.optimiser.add_param_group(
                    {"params": scene_obj.trainer.fc_occ_map.parameters(),
                     "lr": self.cfg.learning_rate, "weight_decay": self.cfg.weight_decay})
                self.optimiser.add_param_group(
                    {"params": scene_obj.trainer.pe.parameters(),
                     "lr": self.cfg.learning_rate, "weight_decay": self.cfg.weight_decay})

            else:
                scene_obj = self.vis_dict[self.obj_id]
                scene_obj.append_keyframe(rgb_image, depth_image, state, bbox, camera_pose, self.keyframe_counter)

            self.keyframe_counter += 1

       ##################################################################
        # training data preperation, get training data for all objs
        Batch_N_gt_depth = []
        Batch_N_gt_rgb = []
        Batch_N_depth_mask = []
        Batch_N_obj_mask = []
        Batch_N_input_pcs = []
        Batch_N_sampled_z = []

        with performance_measure(f"Sampling over {len(self.obj_dict.keys())} objects,"):
            for obj_id, obj_k in self.obj_dict.items():

                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z \
                    = obj_k.get_training_samples(self.cfg.n_iter_per_frame * self.cfg.win_size, self.cfg.n_samples_per_frame,
                                                 self.cam_info.rays_dir_cache)

                # merge first two dims, sample_per_frame*num_per_frame
                Batch_N_gt_depth.append(gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]]))
                Batch_N_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]]))
                Batch_N_depth_mask.append(valid_depth_mask)
                Batch_N_obj_mask.append(obj_mask)
                Batch_N_input_pcs.append(
                    input_pcs.reshape(
                        [input_pcs.shape[0] * input_pcs.shape[1],
                         input_pcs.shape[2],
                         input_pcs.shape[3]]))
                Batch_N_sampled_z.append(sampled_z.reshape(
                    [sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]]))

        ####################################################
        # training
        assert len(Batch_N_input_pcs) > 0
        # move data to GPU  (n_obj, n_iter_per_frame, win_size*num_per_frame, 3)
        with performance_measure(f"stacking and moving to gpu: "):
            Batch_N_input_pcs = torch.stack(Batch_N_input_pcs).to(self.cfg.training_device)
            Batch_N_gt_depth = torch.stack(Batch_N_gt_depth).to(self.cfg.training_device)
            Batch_N_gt_rgb = torch.stack(Batch_N_gt_rgb).to(self.cfg.training_device) / 255.  # todo
            Batch_N_depth_mask = torch.stack(Batch_N_depth_mask).to(self.cfg.training_device)
            Batch_N_obj_mask = torch.stack(Batch_N_obj_mask).to(self.cfg.training_device)
            Batch_N_sampled_z = torch.stack(Batch_N_sampled_z).to(self.cfg.training_device)

        with performance_measure(f"Training over {len(self.obj_dict.keys())} objects,"):
            for iter_step in range(self.cfg.n_iter_per_frame):
                data_idx = slice(iter_step*self.n_sample_per_step, (iter_step+1)*self.n_sample_per_step)
                batch_input_pcs = Batch_N_input_pcs[:, data_idx, ...]
                batch_gt_depth = Batch_N_gt_depth[:, data_idx, ...]
                batch_gt_rgb = Batch_N_gt_rgb[:, data_idx, ...]
                batch_depth_mask = Batch_N_depth_mask[:, data_idx, ...]
                batch_obj_mask = Batch_N_obj_mask[:, data_idx, ...]
                batch_sampled_z = Batch_N_sampled_z[:, data_idx, ...]

                # for loop training
                batch_alpha = []
                batch_color = []
                for k, obj_id in enumerate(self.obj_dict.keys()):
                    obj_k = self.obj_dict[obj_id]
                    embedding_k = obj_k.trainer.pe(batch_input_pcs[k])
                    alpha_k, color_k = obj_k.trainer.fc_occ_map(embedding_k)
                    batch_alpha.append(alpha_k)
                    batch_color.append(color_k)

                batch_alpha = torch.stack(batch_alpha)
                batch_color = torch.stack(batch_color)

                # step loss
                batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                                     batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                                     batch_obj_mask.detach(), batch_depth_mask.detach(),
                                                     batch_sampled_z.detach())

                # with performance_measure(f"Backward"):
                batch_loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad(set_to_none=True)
