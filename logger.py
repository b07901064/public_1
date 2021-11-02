import cv2
import ray
import numpy as np
import wandb
import matplotlib.pyplot as plt
from visulization import visualize_birdview, visualize_semantic_processed, visualize_semantic
from matplotlib.patches import Circle

class Logger:
    def __init__(self, config):
        wandb.init(project=config.project, config=config)
    
    @property
    def log_dir(self):
        return wandb.run.dir
    
    def log_bev(self, it, lbls, info, num_log=16):
        
        #print('lbl size:', lbls.size())
        #print(lbls[:num_log].size())
        lbls = lbls[:num_log].numpy()
        bevs = np.stack([visualize_birdview(lbl, num_channels=6) for lbl in lbls], 0)

        cmds = info.pop('cmds')
        locs = info.pop('locs')
        pred_locs = info.pop('pred_locs')
        
        # Draw on bevs
        for loc, pred_loc, bev, cmd in zip(locs, pred_locs, bevs, cmds):
            #print(loc)
            for t in range(loc.shape[0]):
                #print(loc.shape[0])
                #print(loc[0])
                #print(loc[1])
                #print(loc[2])
                gx, gy = loc[t].astype(int)
                px, py = pred_loc[t].astype(int)
                cv2.circle(bev, (gx, gy), 3, (0,255,0), 1) #Green
                cv2.circle(bev, (px, py), 4, (255,0,255), 1) #Pink
        
        
        info.update({'it': it, 'visuals': [wandb.Image(bev) for bev in bevs]})
        wandb.log(info)
    

    def log_rgb(self, it, rgbs, lbls, info):
        
        #rgbs = rgbs.permute(0,2,3,1)
        #print('rgbs size:', rgbs.size())
        rgb = rgbs[0].numpy()
        #print('rgb size:', rgb.shape)
        lbl = lbls[0].numpy()
        
        cmd = info.pop('cmds')[0]
        tgt_rgb_loc = info.pop('tgt_rgb_locs')[0]
        tgt_bev_loc = info.pop('tgt_bev_locs')[0]
        pred_rgb_loc = info.pop('pred_rgb_locs')[0]
        pred_bev_loc = info.pop('pred_bev_locs')[0]
        #print('pred sem: ',info.pop('pred_sems')[0])
        #print('target:', info.pop('tgt_sems')[0])
        pred_sem = visualize_semantic_processed(info.pop('pred_sems')[0])
        tgt_sem = visualize_semantic_processed(info.pop('tgt_sems')[0])
        #pred_sem = visualize_semantic(info.pop('pred_sems')[0])
        #tgt_sem = visualize_semantic(info.pop('tgt_sems')[0])
        
        f, [sem_axes, rgb_axes, lbl_axes] = plt.subplots(3,tgt_rgb_loc.shape[0], figsize=(30,15))
        
        sem_axes[0].imshow(tgt_sem)
        sem_axes[1].imshow(pred_sem)
        #print(tgt_rgb_loc.shape)    #(6,6,2)
        #print(rgb_axes.shape)       #(6,)
        
        for i in range(tgt_rgb_loc.shape[0]):
            rgb_ax = rgb_axes[i]
            lbl_ax = lbl_axes[i]
        
            rgb_ax.imshow(rgb)
            lbl_ax.imshow(visualize_birdview(lbl, num_channels=6))
            rgb_ax.set_title({0:'Left',1:'Right',2:'Straight',3:'Follow'}.get(cmd,'???'))
            for tgt_rgb, tgt_bev, pred_rgb, pred_bev in zip(tgt_rgb_loc[i],tgt_bev_loc[i],pred_rgb_loc[i],pred_bev_loc[i]):
                
                rgb_ax.add_patch(Circle(tgt_rgb, 4, color='lime'))   #Green
                rgb_ax.add_patch(Circle(pred_rgb, 4, color= 'fuchsia'))   #Pink
                
                lbl_ax.add_patch(Circle(tgt_bev, 1, color='lime'))   #Green
                lbl_ax.add_patch(Circle(pred_bev, 1, color= 'fuchsia'))   #Pink
                
        info.update({'global_it': it, 'visuals': plt})
        wandb.log(info)
        plt.close('all')

    


