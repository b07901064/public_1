from train0 import main
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--project', default='carla_lbc')
parser.add_argument('--config-path', default='experiments/config_nocrash_lbc.yaml')
parser.add_argument('--device', choices=['cpu', 'cuda:1'], default= 'cuda:1')
parser.add_argument('--mode', choices = ['bev', 'cam'], default = 'bev')

# Training data config
parser.add_argument('--batch-size', type=int, default= 256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num-epochs', type=int, default=20)


# Logging config
parser.add_argument('--num-iters-per-log', type=int, default=100)

args = parser.parse_args()

main(args)






''' 
mode: bev
batch size can use 256 or possibly more

mode: cam
batch size can use 56 (about the limit)

specify camera index under dataloader.SingleDataset 'self.cam_idx = 0/1/2' 
specify number of plans under dataloader.SingleDataset 'self.num_plans = 6'
specify amount of data to be loaded under dataloader line ~184
'''