import time
import loss
from vmap import *
import utils
import open3d
import matplotlib.pyplot as plt
import dataset
import vis
from functorch import vmap
import argparse
from cfg import Config
import shutil

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# Bit operations
BIT_MOVE_16 = int(2**16)
BIT_MOVE_8 = int(2**8)

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]


# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
# https://github.com/felixchenfy/open3d_ros_pointcloud_conversion
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255).astype(int) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]
        colors = colors.reshape(colors.shape[0],1)
        cloud_data = np.concatenate((points, colors), axis=1, dtype=object)
        # cloud_data=np.c_[points, colors, dtype='f4, f4, f4, i4']
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

if __name__ == "__main__":
    #############################################
    # init config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--logdir', default='./logs/debug',
                        type=str)
    parser.add_argument('--config',
                        default='./configs/Replica/config_replica_room0_vMAP.json',
                        type=str)
    parser.add_argument('--save_ckpt',
                        default=False,
                        type=bool)
    args = parser.parse_args()

    log_dir = args.logdir
    config_file = args.config
    save_ckpt = args.save_ckpt
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params
    n_sample_per_step = cfg.n_per_optim
    n_sample_per_step_bg = cfg.n_per_optim_bg

    # param for vis
    vis3d = open3d.visualization.Visualizer()
    vis3d.create_window(window_name="3D mesh vis",
                        width=cfg.W,
                        height=cfg.H,
                        left=600, top=50)
    view_ctl = vis3d.get_view_control()
    view_ctl.set_constant_z_far(10.)

    # set camera
    cam_info = cameraInfo(cfg)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=cfg.W,
        height=cfg.H,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy)

    # init obj_dict
    obj_dict = {}   # only objs
    vis_dict = {}   # including bg

    optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    rospy.init_node('imap')
    reconstruction_pub = rospy.Publisher('reconstruction', PointCloud2, queue_size=10)
    input_pointcloud_pub = rospy.Publisher('input_pointcloud', PointCloud2, queue_size=10)

    print("breakpoint test")

    #############################################
    # init data stream

    # load dataset
    dataloader = dataset.init_loader(cfg)
    dataloader_iterator = iter(dataloader)
    dataset_len = len(dataloader)


    for frame_id in tqdm(range(dataset_len)):
        print("*********************************************")
        if rospy.is_shutdown():
            print('rospy.is_shutdown()')
            break
        # get new frame data
        with performance_measure(f"getting next data"):
            sample = next(dataloader_iterator)


        if sample is not None:  # new frame
            last_frame_time = time.time()
            with performance_measure(f"Appending data"):
                rgb = sample["image"].to(cfg.data_device)
                depth = sample["depth"].to(cfg.data_device)
                twc = sample["T"].to(cfg.data_device)
                bbox_dict = sample["bbox_dict"]
                
                if "frame_id" in sample.keys():
                    live_frame_id = sample["frame_id"]
                else:
                    live_frame_id = frame_id

                inst = sample["obj"].to(cfg.data_device)
                obj_ids = torch.unique(inst)

                # append new frame info to objs in current view
                for obj_id in obj_ids:
                    if obj_id == -1:    # unsured area
                        continue
                    obj_id = int(obj_id)
                    # convert inst mask to state
                    state = torch.zeros_like(inst, dtype=torch.uint8, device=cfg.data_device)
                    state[inst == obj_id] = 1
                    state[inst == -1] = 2
                    bbox = bbox_dict[obj_id]

                    if obj_id in vis_dict.keys():
                        scene_obj = vis_dict[obj_id]
                        scene_obj.append_keyframe(rgb, depth, state, bbox, twc, live_frame_id)
                    else: # init scene_obj
                        if len(obj_dict.keys()) >= cfg.max_n_models:
                            print("models full!!!! current num ", len(obj_dict.keys()))
                            continue
                        print("init new obj ", obj_id)

                        scene_obj = sceneObject(cfg, obj_id, rgb, depth, state, bbox, twc, live_frame_id)
                        # scene_obj.init_obj_center(intrinsic_open3d, depth, state, twc)
                        obj_dict.update({obj_id: scene_obj})
                        vis_dict.update({obj_id: scene_obj})

                        optimiser.add_param_group({"params": scene_obj.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                        optimiser.add_param_group({"params": scene_obj.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})


        ##################################################################
        # training data preperation, get training data for all objs
        Batch_N_gt_depth = []
        Batch_N_gt_rgb = []
        Batch_N_depth_mask = []
        Batch_N_obj_mask = []
        Batch_N_input_pcs = []
        Batch_N_sampled_z = []

        with performance_measure(f"Sampling over {len(obj_dict.keys())} objects,"):
            for obj_id, obj_k in obj_dict.items():
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z \
                    = obj_k.get_training_samples(cfg.n_iter_per_frame * cfg.win_size, cfg.n_samples_per_frame,
                                                 cam_info.rays_dir_cache)
                # merge first two dims, sample_per_frame*num_per_frame
                Batch_N_gt_depth.append(gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]]))
                Batch_N_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]]))
                Batch_N_depth_mask.append(valid_depth_mask)
                Batch_N_obj_mask.append(obj_mask)
                Batch_N_input_pcs.append(input_pcs.reshape([input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                Batch_N_sampled_z.append(sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]]))

                # vis sampled points in open3D
                # rgb_np = rgb.cpu().numpy().astype(np.uint8).transpose(1,0,2)
                # depth_np = depth.cpu().numpy().astype(np.float32).transpose(1,0)
                # twc_np = twc.cpu().numpy()
                # rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                #     open3d.geometry.Image(rgb_np),
                #     open3d.geometry.Image(depth_np),
                #     depth_trunc=100,
                #     depth_scale=1,
                #     convert_rgb_to_intensity=False,
                # )

                # T_CW = np.linalg.inv(twc_np)
                # # input image pc
                # input_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
                #     image=rgbd,
                #     intrinsic=intrinsic_open3d,
                #     extrinsic=T_CW)
                # input_pc.points = open3d.utility.Vector3dVector(np.array(input_pc.points) - obj_k.obj_center.cpu().numpy())

                # ros_msg = convertCloudFromOpen3dToRos(input_pc, frame_id="map")
                # ros_msg.header.stamp = rospy.Time.now()
                # print("Publishing Input Data")
                # input_pointcloud_pub.publish(ros_msg)


        ####################################################
        # training
        assert len(Batch_N_input_pcs) > 0
        # move data to GPU  (n_obj, n_iter_per_frame, win_size*num_per_frame, 3)
        with performance_measure(f"stacking and moving to gpu: "):

            Batch_N_input_pcs = torch.stack(Batch_N_input_pcs).to(cfg.training_device)
            Batch_N_gt_depth = torch.stack(Batch_N_gt_depth).to(cfg.training_device)
            Batch_N_gt_rgb = torch.stack(Batch_N_gt_rgb).to(cfg.training_device) / 255. # todo
            Batch_N_depth_mask = torch.stack(Batch_N_depth_mask).to(cfg.training_device)
            Batch_N_obj_mask = torch.stack(Batch_N_obj_mask).to(cfg.training_device)
            Batch_N_sampled_z = torch.stack(Batch_N_sampled_z).to(cfg.training_device)
            if cfg.do_bg:
                bg_input_pcs = bg_input_pcs.to(cfg.training_device)
                bg_gt_depth = bg_gt_depth.to(cfg.training_device)
                bg_gt_rgb = bg_gt_rgb.to(cfg.training_device) / 255.
                bg_valid_depth_mask = bg_valid_depth_mask.to(cfg.training_device)
                bg_obj_mask = bg_obj_mask.to(cfg.training_device)
                bg_sampled_z = bg_sampled_z.to(cfg.training_device)

        with performance_measure(f"Training over {len(obj_dict.keys())} objects,"):
            for iter_step in range(cfg.n_iter_per_frame):
                data_idx = slice(iter_step*n_sample_per_step, (iter_step+1)*n_sample_per_step)
                batch_input_pcs = Batch_N_input_pcs[:, data_idx, ...]
                batch_gt_depth = Batch_N_gt_depth[:, data_idx, ...]
                batch_gt_rgb = Batch_N_gt_rgb[:, data_idx, ...]
                batch_depth_mask = Batch_N_depth_mask[:, data_idx, ...]
                batch_obj_mask = Batch_N_obj_mask[:, data_idx, ...]
                batch_sampled_z = Batch_N_sampled_z[:, data_idx, ...]
                # if cfg.training_strategy == "forloop":
                    # for loop training
                batch_alpha = []
                batch_color = []
                for k, obj_id in enumerate(obj_dict.keys()):
                    obj_k = obj_dict[obj_id]
                    embedding_k = obj_k.trainer.pe(batch_input_pcs[k])
                    alpha_k, color_k = obj_k.trainer.fc_occ_map(embedding_k)
                    batch_alpha.append(alpha_k)
                    batch_color.append(color_k)

                batch_alpha = torch.stack(batch_alpha)
                batch_color = torch.stack(batch_color)



            # step loss
            # with performance_measure(f"Batch LOSS"):
                batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                     batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                     batch_obj_mask.detach(), batch_depth_mask.detach(),
                                     batch_sampled_z.detach())

            # with performance_measure(f"Backward"):

                batch_loss.backward()
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)



        ####################################################################
        # live vis mesh
        if (((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len-1) or
            (cfg.live_mode and time.time()-last_frame_time>cfg.keep_live_time)) and frame_id >= 10:
            vis3d.clear_geometries()
            for obj_id, obj_k in vis_dict.items():
                bound = obj_k.get_bound(intrinsic_open3d)
                if bound is None:
                    print("get bound failed obj ", obj_id)
                    continue
                adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim))
                mesh = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=adaptive_grid_dim)
                if mesh is None:
                    print("meshing failed obj ", obj_id)
                    continue

                # save to dir
                obj_mesh_output = os.path.join(log_dir, "scene_mesh")
                os.makedirs(obj_mesh_output, exist_ok=True)
                mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.obj".format(frame_id, str(obj_id))))

                # live vis
                open3d_mesh = vis.trimesh_to_open3d(mesh)
                # vis3d.add_geometry(open3d_mesh)
                # vis3d.add_geometry(bound)
                # # update vis3d
                # vis3d.poll_events()
                # vis3d.update_renderer()
                
                pc = open3d.geometry.PointCloud()
                pc.points = open3d_mesh.vertices
                pc.colors = open3d_mesh.vertex_colors

                ros_msg = convertCloudFromOpen3dToRos(pc, frame_id="map")
                ros_msg.header.stamp = rospy.Time.now()
                print("Publishing Reconstruction Data")
                reconstruction_pub.publish(ros_msg)



        with performance_measure("saving ckpt"):
            if save_ckpt and ((((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len - 1) or
                               (cfg.live_mode and time.time() - last_frame_time > cfg.keep_live_time)) and frame_id >= 10):
                for obj_id, obj_k in vis_dict.items():
                    ckpt_dir = os.path.join(log_dir, "ckpt", str(obj_id))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    bound = obj_k.get_bound(intrinsic_open3d)   # update bound
                    obj_k.save_checkpoints(ckpt_dir, frame_id)
                # save current cam pose
                cam_dir = os.path.join(log_dir, "cam_pose")
                os.makedirs(cam_dir, exist_ok=True)
                torch.save({"twc": twc,}, os.path.join(cam_dir, "twc_frame_{}".format(frame_id) + ".pth"))


