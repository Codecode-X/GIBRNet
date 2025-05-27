import numpy as np
import torch

def collate_fn(batch):

    N = len(batch)
    max_node_num = 4
    head_num = 2
    sq_len=5

    head_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    pos_batch = np.zeros((N, sq_len, max_node_num, 6))
    attmat_batch = np.zeros((N, sq_len, max_node_num, max_node_num))
    atomic_label_batch = np.zeros((N))
    scene_batch = np.zeros((N, sq_len, 3, 360,640))
    nodes_depth_batch = np.zeros((N, sq_len, 4, 3, 224, 224))
    gaze_vec_batch = np.zeros((N,sq_len,2,2))
    face_kp_batch=np.zeros((N, sq_len, 2, 68, 3))
    ID_batch=np.zeros((N, sq_len))
    position_01_batch = np.zeros((N,sq_len,360,640))
    item = []

    for i, (head_patch_sq, pos_sq, attmat_sq, atomic_label, scene_sq, rec, nodes_depth_sq, gaze_vec_sq, position_01) in enumerate(batch):

            head_batch[i, ...] = head_patch_sq
            pos_batch[i, ...] = pos_sq
            attmat_batch[i, ...] = attmat_sq
            atomic_label_batch[i] = atomic_label
            item.append(rec)
            scene_batch[i, ...] = scene_sq
            nodes_depth_batch[i,...] = nodes_depth_sq
            gaze_vec_batch[i,...] = gaze_vec_sq
            position_01_batch[i,...] = position_01


    head_batch = torch.FloatTensor(head_batch)
    pos_batch = torch.FloatTensor(pos_batch)
    attmat_batch = torch.FloatTensor(attmat_batch)
    scene_batch = torch.FloatTensor(scene_batch)
    atomic_label_batch = torch.LongTensor(atomic_label_batch)
    nodes_depth_batch = torch.FloatTensor(nodes_depth_batch)
    gaze_vec_batch = torch.FloatTensor(gaze_vec_batch)
    position_01_batch = torch.FloatTensor(position_01_batch)


    return head_batch, pos_batch, attmat_batch, atomic_label_batch, scene_batch, item, nodes_depth_batch, gaze_vec_batch,position_01_batch