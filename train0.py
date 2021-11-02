from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import numpy as np
import random
import copy
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
#torch.set_printoptions(linewidth= 120)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import tensorboard
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import math
from os.path import join
import glob
import torch.utils.data as data
import cupy as cp

from augmenter import augment
from visulization import filter_sem
from visulization import PIXELS_PER_METER
from logger import Logger

PIXELS_PER_METER= 3

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = np.array(image)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    result = torch.from_numpy(result)
    result = result.permute(2,0,1)

    return result
    
def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])


from torchvision import transforms
from os.path import join
import os
class SingleDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.augmenter = augment(0.5)
        self.seg_channels = [4,6,7,10,18]
        self.direc = datapath # json path
        self.crop_size = 64
        self.crop_top = 8
        self.crop_bottom = 8
        self.margin = (96- self.crop_size) // 2

        self.mapping = {
            (220, 20, 60): 1,
            (153, 153, 153): 2,
            (157, 234, 50): 3,
            (128, 64, 128): 4,
            (244, 35, 232): 5,
            (0, 0, 142): 6,
            (220, 220, 0): 7,
            # Exception : 0
        }

        f= open(join(datapath,'data.json'))
        self.datei = json.load(f) # data is a dictionary
        self.num_frames = len(self.datei)
        self.num_plans = 6
        self.cam_idx = 0    # 3 frontal camera
        self.size = self.datei["len"] - self.num_plans - 6
        del self.datei["len"]
        self.transform = transforms.Compose([transforms.ToTensor()])
        f.close()


    def __len__(self):
        return self.size
        
         
    def __getitem__(self, idx):

        if idx > self.size:
            idx = random.randint(0 , self.size)

        # print ('idx: ', idx)
        locs = self.__class__.access('loc', self.datei, idx, T = self.num_plans+1)
        locs = locs[:,:2]
        

        rot = self.__class__.access('rot', self.datei, idx)
        spd = self.__class__.access('spd', self.datei, idx)
        cmd = self.__class__.access('cmd', self.datei, idx)

        rgb = cv2.imread(self.direc + '/rgbs/wide_{}_{}'.format(self.cam_idx, format(idx, '05d') + '.jpg' ))
        sem = cv2.imread(self.direc + '/rgbs/wide_sem_{}_{}'.format(self.cam_idx, format(idx, '05d') + '.png' ))
        sem = self.map2label(sem)
        
        # Augment rgb
        rgb = rgb.reshape(240,480,3)
        rgb = self.augmenter(images=rgb[None,...,::-1])[0] #!not sure

        sem = sem[self.crop_top:-self.crop_bottom]
        rgb = rgb[self.crop_top:-self.crop_bottom]


        lbl0 = self.__class__.access('lbl_00', self.datei, idx)
        lbl1 = self.__class__.access('lbl_01', self.datei, idx)
        lbl2 = self.__class__.access('lbl_02', self.datei, idx)
        lbl3 = self.__class__.access('lbl_03', self.datei, idx)
        lbl4 = self.__class__.access('lbl_04', self.datei, idx)
        lbl5 = self.__class__.access('lbl_05', self.datei, idx)
        lbl6 = self.__class__.access('lbl_06', self.datei, idx)
        lbl7 = self.__class__.access('lbl_07', self.datei, idx)
        lbl8 = self.__class__.access('lbl_08', self.datei, idx)
        lbl9 = self.__class__.access('lbl_09', self.datei, idx)
        lbl10 = self.__class__.access('lbl_10', self.datei, idx)
        lbl11 = self.__class__.access('lbl_11', self.datei, idx)


        lbl0 = cv2.imread(self.direc + str(lbl0), 0)
        lbl1 = cv2.imread(self.direc + str(lbl1), 0)
        lbl2 = cv2.imread(self.direc + str(lbl2), 0)
        lbl3 = cv2.imread(self.direc + str(lbl3), 0)
        lbl4 = cv2.imread(self.direc + str(lbl4), 0)
        lbl5 = cv2.imread(self.direc + str(lbl5), 0)
        lbl6 = cv2.imread(self.direc + str(lbl6), 0)
        lbl7 = cv2.imread(self.direc + str(lbl7), 0)
        lbl8 = cv2.imread(self.direc + str(lbl8), 0)
        lbl9 = cv2.imread(self.direc + str(lbl9), 0)
        lbl10 = cv2.imread(self.direc + str(lbl10), 0)
        lbl11 = cv2.imread(self.direc + str(lbl11), 0)


        lbl = np.stack((lbl0, lbl1, lbl2, lbl3, lbl4, lbl5, lbl6, lbl7, lbl8, lbl9, lbl10, lbl11), 0)
        if self.transform is not None:
            lbl = self.transform(lbl)        
        #print(lbl.size())
        lbl = lbl.permute(0,2,1)



        # Rotate BEV
        yaw = float(rot)
        lbl = rotate_image(lbl, yaw+90)
        lbl = lbl[:,self.margin:self.margin+self.crop_size, self.margin:self.margin+self.crop_size]

        #Rotate locs
        dloc = rotate_points(locs[1:] - locs[0:1], -yaw-90)*PIXELS_PER_METER - [-self.crop_size/2, -self.crop_size/2]

        return rot, lbl, dloc, spd, int(cmd), rgb, sem


    
    def map2label(self,sem):
        resem = torch.zeros(sem.shape[:2], dtype=torch.long)
        for clr, label in self.mapping.items():
            idx = (sem[:, :, 0] == clr[0]) & (sem[:, :, 1] == clr[1]) & (sem[:, :, 2] == clr[2])
            resem[idx] = label
        return resem

        
    @staticmethod
    def access(tag, datei, index, T = 0):
        stack = []
        if tag == "loc":
            for i in range (T):
                stack.append(datei[str(index + i)]["loc"][:2])      
            return np.asarray(stack)


        elif re.match(r"lbl_*",tag):
            return datei[str(index)][tag]

        else:
            stack.append(datei[str(index)][tag])
            return np.asarray(stack)



import tqdm



# Network
import torch
from model import PointModel, LBC
from visulization import visualize_birdview


# Train
def main(args):

    

    # Concate Datasets
    root_dir = '/root/dataset/rails1M/main_trajs6_converted2'
    list_of_datasets= []
    flag = 0
    
    print('start loading')
    for data_path in glob.glob(f'{root_dir}/**'):
        if SingleDataset(data_path).size <= 0: continue
        list_of_datasets.append(SingleDataset(data_path)) 

        flag += 1
        # around 6757 in total
        if flag > 30:
          break
    print('finished. Flag= ',flag)

    multiple_datasets = data.ConcatDataset(list_of_datasets)
    train_loader = DataLoader(multiple_datasets, batch_size= args.batch_size, shuffle= False)
    
    
    global_it = 0
    print('in main function')

    if args.mode == 'bev':
        lbc = LBC(args)
        logger = Logger(args)

        for epoch in range(args.num_epochs):
            for rots, lbls, dlocs, spds, cmds, rgbs, sems in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
                info = lbc.train(rots, lbls, spds, dlocs,  cmds, rgbs, sems, train='bev')
                global_it += 1

                if global_it % args.num_iters_per_log == 0:
                    print('>> Loss: ', info['loss'])
                    logger.log_bev(global_it, lbls, info)
            torch.save(lbc.bev_model.state_dict(), os.path.join('bev_model_{}.th'.format(epoch+1)))
        print('finished training bev, global_it =', global_it, 'epoch:', epoch)
        
    

    elif args.mode == 'cam':
        lbc = LBC(args)
        logger = Logger(args)
        lbc.bev_model.load_state_dict(torch.load('bev_model_20.th', map_location=args.device))
        lbc.bev_model.eval()

        for epoch in range(args.num_epochs):
            for rots, lbls, dlocs, spds, cmds, rgbs, sems in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
            
                info = lbc.train(rots, lbls, spds, dlocs,  cmds, rgbs, sems, train='cam')
                global_it += 1

                if global_it % args.num_iters_per_log == 0:
                    print('>> Loc loss: ',info['loc_loss'])
                    print('>> Seg loss: ',info['seg_loss'])
                    logger.log_rgb(global_it, rgbs, lbls, info)

            torch.save(lbc.rgb_model.state_dict(), os.path.join('rgb_model_{}.th'.format(epoch+1)))
        print('finished training rgb, global_it =', global_it, 'epoch:', epoch)
        
        