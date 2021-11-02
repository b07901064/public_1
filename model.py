# Network
import torch
from torch._C import dtype
import torch.nn as nn
from resnet import resnet18, resnet34
from spatial_softmax import SpatialSoftmax
import torch.nn.functional as F
from _numpy import to_numpy
from normalization import Normalize
from segmentation import SegmentationHead
from converter import Converter


class PointModel(nn.Module):
    def __init__(self, backbone, pretrained=False, height=96, width=96, input_channel=3, output_channel=20, num_labels=8):
        super().__init__()
        
        #print('input channel:', input_channel)
        #print('height', height)
        self.kh = height//32
        self.kw = width//32
        self.num_labels = num_labels
        
        # backbone: only takes lbls
        self.backbone = eval(backbone)(pretrained=pretrained, num_channels=input_channel)

        self.spd_encoder = nn.Sequential(
            nn.Linear(1,128),
            nn.ReLU(True),
            nn.Linear(128,128),
        )
        self.upconv = nn.Sequential(
            # how input channel = 640 ?
            nn.ConvTranspose2d(640,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,output_channel,1,1,0),
            SpatialSoftmax(height//4, width//4),
        )
        
    def forward(self, bev, spd):
        bev = (bev>0).float() #torch.Size([8, 12, 64, 64])
        inputs = self.backbone(bev/255.) #torch.Size([8, 512, 2, 2])
        spd_embds = self.spd_encoder(spd[:,None]) #torch.Size([8, 1, 1, 128])
        spd_embds = spd_embds.permute (0,3,2,1) #torch.Size([8, 128, 1, 1])
        spd_embds = spd_embds.repeat(1, 1, self.kh,self.kw) #torch.Size([8, 128, 2, 2])

        outputs = self.upconv(torch.cat([inputs, spd_embds], 1)) #torch.Size([8,18,2])
        return outputs

class RGBPointModel(PointModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.seg_head = SegmentationHead(512, self.num_labels)
        self.img_size = nn.Parameter(torch.tensor([self.kw*32,self.kh*32]).float(), requires_grad=False) # not sure what this is but it's not used
        
    def forward(self, rgb, spd, pred_seg= True):
        inputs = self.backbone(self.normalize(rgb/255.))
        #print('inputs: ',inputs.size()) #torch.Size([64, 512, 7, 15])
        #spd_embds = self.spd_encoder(spd[:,None])[...,None,None].repeat(1,1,self.kh,self.kw)
        spd_embds = self.spd_encoder(spd[:,None]) #torch.Size([8, 1, 1, 128])
        spd_embds = spd_embds.permute (0,3,2,1) #torch.Size([8, 128, 1, 1])
        spd_embds = spd_embds.repeat(1, 1, self.kh,self.kw) #torch.Size([8, 128, 2, 2])
        #print('spd_embds size :', spd_embds.size())
        points = self.upconv(torch.cat([inputs, spd_embds], 1))
        
        points[...,1] = (points[...,1] + 1)/2
        
        if pred_seg:
            segs = self.seg_head(inputs)
            return points, segs
        else:
            return points




from torch import optim
import numpy as np
import cupy as cp
#from utils import _numpy


class LBC:
    def __init__(self, args):
        
        self.crop_top = 8
        self.crop_bottom = 8
        self.num_plan = 6
        self.crop_size = 64
        self.num_cmds = 6
        self.seg_weight = 0.05
        self.bev_model_dir = '/root/Documents/github/public0/bev_model'

        # Save configs
        self.device = torch.device(args.device)
        self.T = self.num_plan # T in LBC
        
        # Create models
        self.bev_model = PointModel(
            'resnet18',
            height=64, width=64,
            input_channel=12,
            output_channel= self.T*self.num_cmds
        ).to(self.device)

        self.rgb_model = RGBPointModel(
            'resnet34',
            pretrained=True,
            height=240-self.crop_top-self.crop_bottom, width=480,
            output_channel=self.T*self.num_cmds
        ).to(self.device)
        
        self.bev_optim = optim.Adam(self.bev_model.parameters(), lr= args.lr)
        self.rgb_optim = optim.Adam(self.rgb_model.parameters(), lr= args.lr)
        
        self.converter = Converter(offset=6.0, scale=[1.5,1.5]).to(self.device) ## TODO/ dunno
        
    def train(self, rots, lbls, spds, locs, cmds, rgbs, sems, train='image'):

        lbls = lbls.float().to(self.device)
        rgbs = rgbs.permute(0,3,1,2).float().to(self.device)
        sems = sems.to(self.device)
        rots = rots.float().to(self.device)
        locs = locs.float().to(self.device)
        spds = spds.float().to(self.device)
        cmds = cmds.long().to(self.device)
        
        if train == 'bev':
            return self.train_bev(rots, lbls, spds, locs, cmds)
        elif train == 'cam':
            return self.train_rgb(rots, lbls, spds, locs, cmds, rgbs,sems)
        
        else:
            raise NotImplementedError
            
    def train_bev(self, rots, lbls, spds, locs, cmds):

        pred_locs = self.bev_model(lbls, spds).view(-1,self.num_cmds,self.T,2)

        # Scale pred locs
        pred_locs = (pred_locs+1) * self.crop_size/2
        pred_locs = pred_locs.gather(1, cmds[:,None,None,None].repeat(1,1,self.T,2)).squeeze(1)
        loss = F.mse_loss(pred_locs, locs)

        self.bev_optim.zero_grad()
        loss.backward()
        self.bev_optim.step()
        
        return dict(
            loss=float(loss),
            cmds= to_numpy(cmds),
            locs= to_numpy(locs),
            pred_locs= to_numpy(pred_locs),
        )


    def train_rgb(self, rots, lbls, spds, locs, cmds, rgbs, sems):
        
        with torch.no_grad():
            tgt_bev_locs = (self.bev_model(lbls, spds).view(-1,self.num_cmds,self.T,2)+1) * self.crop_size/2
        
        pred_rgb_locs, pred_sems = self.rgb_model(rgbs, spds)
        pred_rgb_locs = (pred_rgb_locs.view(-1,self.num_cmds,self.T,2)+1) * self.rgb_model.img_size/2
    
        tgt_rgb_locs = self.bev_to_cam(tgt_bev_locs)    # not actually used
        pred_bev_locs = self.cam_to_bev(pred_rgb_locs)
        
        # calculate the loss between tgt_bev_locs and pred_bev_locs
        # the privileged model TEACHES the student 
        act_loss = F.l1_loss(pred_bev_locs, tgt_bev_locs, reduction='none').mean(dim=[2,3])
        
        turn_loss = (act_loss[:,0]+act_loss[:,1]+act_loss[:,2]+act_loss[:,3])/4
        lane_loss = (act_loss[:,4]+act_loss[:,5]+act_loss[:,3])/3
        foll_loss = act_loss[:,3]   #follow
        
        is_turn = (cmds==0)|(cmds==1)|(cmds==2)
        is_lane = (cmds==4)|(cmds==5)

        loc_loss = torch.mean(torch.where(is_turn, turn_loss, foll_loss) + torch.where(is_lane, lane_loss, foll_loss))
        
        # multip_branch_losses = losses.mean(dim=[1,2,3])
        # single_branch_losses = losses.mean(dim=[2,3]).gather(1, cmds[:,None]).mean(dim=1)
        
        # loc_loss = torch.where(cmds==3, single_branch_losses, multip_branch_losses).mean()
        
        #print('pred sems: ', pred_sems.size())  #torch.Size([64, 7, 56, 120])
        #print('sems: ', sems.size())    #torch.Size([64, 3, 224, 480])
        #seg_loss = self.CE_loss(temp, sems)
        temp = F.interpolate(pred_sems.float(),scale_factor=4)
        seg_loss = F.cross_entropy(temp, sems)
        loss = loc_loss + self.seg_weight * seg_loss
        #loss = loc_loss

        self.rgb_optim.zero_grad()
        loss.backward()
        self.rgb_optim.step()
        
        return dict(
            loc_loss=float(loc_loss),
            seg_loss=float(seg_loss),
            cmds= to_numpy(cmds),
            tgt_rgb_locs= to_numpy(tgt_rgb_locs),
            tgt_bev_locs= to_numpy(tgt_bev_locs),
            pred_rgb_locs= to_numpy(pred_rgb_locs),
            pred_bev_locs= to_numpy(pred_bev_locs),
            tgt_sems= np.uint8(to_numpy(sems)),
            pred_sems= to_numpy(pred_sems.argmax(1)),
        )

    @staticmethod
    def CE_loss(X,y):
        m = y.shape[0]
        print('y: ',y.size())   #torch.Size([32, 224, 480])
        print('X: ',X.size())   #torch.Size([32, 7, 224, 480])
        exps = torch.exp(X)
        loss = torch.mean(torch.log(torch.sum(exps, dim=1)) - torch.gather(X, 1, y.unsqueeze(1)))

        return loss


    def bev_to_cam(self, bev_coords):
        
        bev_coords = bev_coords.clone()
        bev_coords[...,1] =  self.crop_size/2 - bev_coords[...,1]
        bev_coords[...,0] = -self.crop_size/2 + bev_coords[...,0]
        world_coords = torch.flip(bev_coords, [-1])
        
        cam_coords = self.converter.world_to_cam(world_coords)
        
        return cam_coords
        
    def cam_to_bev(self, cam_coords):
        world_coords = self.converter.cam_to_world(cam_coords)
        bev_coords = torch.flip(world_coords, [-1])
        bev_coords[...,1] *= -1
        
        return bev_coords + self.crop_size/2
