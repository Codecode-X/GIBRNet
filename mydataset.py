import os
import numpy as np
import torch.utils.data
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor

class mydataset(torch.utils.data.Dataset):

    def __init__(self, dict, is_train):
        self.dict = dict   
        self.is_train = is_train 
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.atomic_label = {'single': 0, 'mutual': 1, 'avert': 2, 'refer': 3, 'follow': 4, 'share': 5}
        self.round_cnt = {'single': 0, 'mutual': 0, 'avert': 0, 'refer': 0, 'follow': 0, 'share': 0}
        self.seq_len = 5
        self.head_num=2
        self.node_num=4

        self.test_list = [] 
        for atomic_class_name in self.dict.keys():
            self.test_list.extend(self.dict[atomic_class_name])
        self.single_list = self.dict['single']
        self.mutual_list = self.dict['mutual']
        self.avert_list = self.dict['avert']
        self.refer_list = self.dict['refer']
        self.follow_list = self.dict['follow']
        self.share_list = self.dict['share']

    

    def __getitem__(self, index):

        if self.is_train:

            if index % 6 == 0:
                mode = 'single'
            elif index % 6 == 1:
                mode = 'mutual'
            elif index % 6 == 2:
                mode = 'avert'
            elif index % 6 == 3:
                mode = 'refer'
            elif index % 6 == 4:
                mode = 'follow'
            elif index % 6 == 5:
                mode = 'share'
            rec = self.dict[mode][index // 6 - self.round_cnt[mode] * len(self.dict[mode])]


            node_patch_sq = np.zeros(shape=(self.seq_len,4,3,224,224))
            pos_sq = np.zeros(shape=(self.seq_len,4,6))
            position_01 = np.zeros(shape=(self.seq_len,360,640))
            attmat_sq = np.zeros(shape=(self.seq_len,4,4))
            scene_sq = np.zeros(shape=(self.seq_len,3,360,640))
            nodes_depth_sq = np.zeros(shape=(self.seq_len,4,3,224,224))
            gaze_vec_sq = np.zeros(shape=(self.seq_len,2,2))
            vid, nid1, nid2, start_fid, end_fid, atomic_class_name = rec
            atomic_label = self.atomic_label[atomic_class_name]
            if index==0:
                print('******************************************** train ********************************************')

            video = np.load('./data/ant_processed/vid_{}_ant_all.npy'.format(vid),allow_pickle=True)

            for sq_id in range(self.seq_len): 
                fid=start_fid+sq_id 
                frame = video[fid]['ant']   
                attmat = video[fid]['attmat']   
                frame_id = video[fid]['ant'][0]['frame_ind']  

                info_path = './data/all'
                img_path = os.path.join(info_path,'img/{}/{}.png'.format(vid,frame_id))
                img = cv2.imread(img_path)
                img_depth_path = os.path.join(info_path,'depth_img/{}/{}.png'.format(vid, frame_id))
                img_depth = cv2.imread(img_depth_path)
                scene_sq[sq_id,...] = self.transforms(img).numpy()
                for node_i in [0,1]:
                    nid=[nid1, nid2][node_i]
                    pos = frame[nid]['pos']   #[xmin,ymin,xmax,ymax]
                   
                    head = cv2.resize(img[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))   #[224,224,3]
                    
                    head = ToTensor()(head).numpy()   #[3,224,224]
                    node_patch_sq[sq_id, node_i, ...] = head
                    pos_vec = np.array([float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                                        (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, node_i, :] = pos_vec

                    head_depth = cv2.resize(img_depth[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    head_depth = ToTensor()(head_depth).numpy()
                    nodes_depth_sq[sq_id,node_i,...] = head_depth

                tar_ind1=np.argwhere(attmat[nid1,:]==1).flatten()  
                tar_ind2=np.argwhere(attmat[nid2,:]==1).flatten()  

                if tar_ind1.size == 0:
                    pass
                else:
                    pos = frame[int(tar_ind1)]['pos']
                    for i in range(len(pos)):  
                        pos[i] = str( int(pos[i]) + np.random.randint(-1,2) )
                    if int(pos[0])<0: pos[0] = '0'
                    if int(pos[1])<0: pos[1] = '0'
                    if int(pos[2])>640: pos[2] = '640'
                    if int(pos[3])>360: pos[3] = '360'
                    target = cv2.resize(img[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    target = ToTensor()(target).numpy()
                    node_patch_sq[sq_id, 2, ...] = target
                    pos_vec = np.array([float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                                        (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, 2, :] = pos_vec
                    target_depth = cv2.resize(img_depth[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    target_depth = ToTensor()(target_depth).numpy()
                    nodes_depth_sq[sq_id,2,...] = target_depth

                if tar_ind2.size ==0:
                    pass
                else:
                    pos = frame[int(tar_ind2)]['pos']
                    for i in range(len(pos)): 
                        pos[i] = str( int(pos[i]) + np.random.randint(-1,2) )
                    if int(pos[0])<0: pos[0] = '0'
                    if int(pos[1])<0: pos[1] = '0'
                    if int(pos[2])>640: pos[2] = '640'
                    if int(pos[3])>360: pos[3] = '360'
                    target = cv2.resize(img[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    target = ToTensor()(target).numpy()
                    node_patch_sq[sq_id, 3, ...] = target
                    pos_vec = np.array([float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                                        (float(pos[0]) + float(pos[2])) / (2 * 640.),(float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, 3, :] = pos_vec
                    target_depth = cv2.resize(img_depth[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    target_depth = ToTensor()(target_depth).numpy()
                    nodes_depth_sq[sq_id, 3, ...] = target_depth
             
                if tar_ind1.size != 0: 
                    if int(tar_ind1) == nid2:
                        attmat_sq[sq_id, 0, 1] = 1 
                    else:
                        attmat_sq[sq_id, 0, 2] = 1 


                if tar_ind2.size != 0: 
                    if int(tar_ind2) == nid1:
                        attmat_sq[sq_id, 1, 0] = 1 
                    elif tar_ind1.size != 0 and int(tar_ind2) == int(tar_ind1):  
                        attmat_sq[sq_id, 0, 3] = 1
                        attmat_sq[sq_id, 1, 2] = 1
                        attmat_sq[sq_id, 1, 3] = 1
                    else:
                        attmat_sq[sq_id, 1, 3] = 1  


            if (index // 6 - self.round_cnt[mode] * len(self.dict[mode])) == (len(self.dict[mode]) - 1):
                self.round_cnt[mode] += 1

            return node_patch_sq, pos_sq, attmat_sq, atomic_label, scene_sq, rec, nodes_depth_sq, gaze_vec_sq, position_01
        else:
            rec = self.test_list[index]
            print('len',len(self.test_list))
            node_patch_sq = np.zeros((self.seq_len, 4, 3, 224, 224))  # [5,4,3,224,224]
            pos_sq = np.zeros((self.seq_len, self.node_num, 6))  # [5,4,6]
            position_01 = np.zeros((self.seq_len, 360, 640))
            attmat_sq = np.zeros((self.seq_len, self.node_num, self.node_num))  # [5,4,4]
            scene_sq = np.zeros((self.seq_len, 3, 360, 640))
            nodes_depth_sq = np.zeros((self.seq_len, 4, 3, 224, 224))
            gaze_vec_sq = np.zeros((self.seq_len, 2, 2))
            vid, nid1, nid2, start_fid, end_fid, atomic_class_name = rec
            vid = int(vid)
            atomic_label = self.atomic_label[atomic_class_name]
            if index==0:
                print('******************** test *******************')
            video = np.load('./data/ant_processed/vid_{}_ant_all.npy'.format(vid),allow_pickle=True)

            for sq_id in range(self.seq_len): 
                fid = start_fid + sq_id 
                frame = video[fid]['ant']  
                attmat = video[fid]['attmat']  
                frame_id = video[fid]['ant'][0]['frame_ind']  
                frame_id = int(frame_id)
                info_path = './data/all'
                img_path = os.path.join(info_path, 'img/{}/{}.png'.format(vid, frame_id))
                img = cv2.imread(img_path) 
                img_depth_path = os.path.join(info_path, 'depth_img/{}/{}.png'.format(vid, frame_id))
                img_depth_feature = cv2.imread(img_depth_path) 
                scene_sq[sq_id, ...] = ToTensor()(img).numpy()
                for node_i in [0, 1]:
                    nid = [nid1, nid2][node_i]
                    pos = frame[nid]['pos'] 
                    pos[0] = int(pos[0])
                    pos[1] = int(pos[1])
                    pos[2] = int(pos[2])
                    pos[3] = int(pos[3])
                    cv2.rectangle(img, (int(pos[0]), int(pos[1])), (int(pos[2]), int(pos[3])), (0, 0, 255), 2)
                   
                    head = cv2.resize(img[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))  # [224,224,3]
                    head = ToTensor()(head).numpy()
                    node_patch_sq[sq_id, node_i, ...] = head
                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.), (float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, node_i, :] = pos_vec
                    head_depth = cv2.resize(img_depth_feature[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    head_depth = ToTensor()(head_depth).numpy()
                    nodes_depth_sq[sq_id, node_i, ...] = head_depth

                tar_ind1 = np.argwhere(attmat[nid1, :] == 1).flatten() 
                tar_ind2 = np.argwhere(attmat[nid2, :] == 1).flatten()  

                if tar_ind1.size == 0:
                    pass
                else:
                    pos = frame[int(tar_ind1)]['pos'] 
                    cv2.rectangle(img, (int(pos[0]) + 2, int(pos[1]) + 2), (int(pos[2]) - 2, int(pos[3]) - 2),(0, 255, 255),2)
                    
                    for i in range(len(pos)):  
                        pos[i] = str(int(pos[i]) + np.random.randint(-1, 2))
                    if int(pos[0]) < 0: pos[0] = '0'
                    if int(pos[1]) < 0: pos[1] = '0'
                    if int(pos[2]) > 640: pos[2] = '640'
                    if int(pos[3]) > 360: pos[3] = '360'

                    target = cv2.resize(img[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :], (224, 224))
                    target = ToTensor()(target).numpy()
                    node_patch_sq[sq_id, 2, ...] = target

                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.), (float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, 2, :] = pos_vec

                    target_depth = cv2.resize(img_depth_feature[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    target_depth = ToTensor()(target_depth).numpy()
                    nodes_depth_sq[sq_id, 2, ...] = target_depth

                if tar_ind2.size == 0:
                    pass
                else:
                    pos = frame[int(tar_ind2)]['pos']
                    cv2.rectangle(img, (int(pos[0]) + 4, int(pos[1]) + 4), (int(pos[2]) - 4, int(pos[3]) - 4),(0, 255, 255),2)
                    for i in range(len(pos)): 
                        pos[i] = str(int(pos[i]) + np.random.randint(-1, 2))
                    if int(pos[0]) < 0: pos[0] = '0'
                    if int(pos[1]) < 0: pos[1] = '0'
                    if int(pos[2]) > 640: pos[2] = '640'
                    if int(pos[3]) > 360: pos[3] = '360'

                    target = cv2.resize(img[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :], (224, 224))
                    target = self.transforms(target).numpy()
                    node_patch_sq[sq_id, 3, ...] = target

                    pos_vec = np.array(
                        [float(pos[0]) / 640., float(pos[1]) / 360., float(pos[2]) / 640., float(pos[3]) / 360.,
                         (float(pos[0]) + float(pos[2])) / (2 * 640.), (float(pos[1]) + float(pos[3])) / (2 * 360.)])
                    pos_sq[sq_id, 3, :] = pos_vec

                    target_depth = cv2.resize(img_depth_feature[int(pos[1]):int(pos[3]), int(pos[0]):int(pos[2]), :],(224, 224))
                    target_depth = self.transforms(target_depth).numpy()
                    nodes_depth_sq[sq_id, 3, ...] = target_depth

                if tar_ind1.size != 0: 
                    if int(tar_ind1) == nid2:
                        attmat_sq[sq_id, 0, 1] = 1 
                    else:
                        attmat_sq[sq_id, 0, 2] = 1 

                if tar_ind2.size != 0:  
                    if int(tar_ind2) == nid1:
                        attmat_sq[sq_id, 1, 0] = 1  
                    elif tar_ind1.size != 0 and int(tar_ind2) == int(tar_ind1):  
                        attmat_sq[sq_id, 0, 3] = 1
                        attmat_sq[sq_id, 1, 2] = 1
                        attmat_sq[sq_id, 1, 3] = 1
                    else:
                        attmat_sq[sq_id, 1, 3] = 1  

            return node_patch_sq, pos_sq, attmat_sq, atomic_label, scene_sq, rec, nodes_depth_sq, gaze_vec_sq, position_01

    def __len__(self):
        if self.is_train:
            max_len = 0
            for event in ['single', 'mutual', 'avert', 'refer', 'follow', 'share']:
                max_len = max(len(self.dict[event]), max_len)
            return max_len * 6
        else:
            return len(self.test_list)