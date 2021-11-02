import torch
import os
import tqdm
from os.path import join

from dataloader import loaddata
from model import PointModel, LBC
from augmenter import augment
from visulization import filter_sem
from visulization import PIXELS_PER_METER
from logger import Logger



def main(args):  
    global_it = 0
    print('in main function')

    if args.mode == 'bev':
        lbc = LBC(args)
        logger = Logger(args)
        train_loader = loaddata(args)

        for epoch in range(args.num_epochs):
            for rots, lbls, dlocs, spds, cmds in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
                info = lbc.train_bev(rots, lbls, spds, dlocs,  cmds)
                global_it += 1

                if global_it % args.num_iters_per_log == 0:
                    print('>> Loss: ', info['loss'])
                    logger.log_bev(global_it, lbls, info)
            torch.save(lbc.bev_model.state_dict(), os.path.join('bev_model_{}.th'.format(epoch+1)))
        print('finished training bev, global_it =', global_it, 'epoch:', epoch)
        
    

    elif args.mode == 'cam':
        lbc = LBC(args)
        logger = Logger(args)
        train_loader = loaddata(args)
        lbc.bev_model.load_state_dict(torch.load('bev_model_20.th', map_location=args.device))
        lbc.bev_model.eval()

        for epoch in range(args.num_epochs):
            for rots, lbls, dlocs, spds, cmds, rgbs, sems in tqdm.tqdm(train_loader, desc=f'Epoch {epoch}'):
            
                info = lbc.train_rgb(rots, lbls, spds, dlocs,  cmds, rgbs, sems)
                global_it += 1

                if global_it % args.num_iters_per_log == 0:
                    print('>> Loc loss: ',info['loc_loss'])
                    print('>> Seg loss: ',info['seg_loss'])
                    logger.log_rgb(global_it, rgbs, lbls, info)

            torch.save(lbc.rgb_model.state_dict(), os.path.join('rgb_model_{}.th'.format(epoch+1)))
        print('finished training rgb, global_it =', global_it, 'epoch:', epoch)
        
        