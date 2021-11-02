from train0 import main
import argparse
import os

#os.environ['CUDA_VISIBLE_DEVICES']= '1'

parser = argparse.ArgumentParser()

parser.add_argument('--project', default='carla_lbc')
parser.add_argument('--config-path', default='experiments/config_nocrash_lbc.yaml')
parser.add_argument('--device', choices=['cpu', 'cuda:1'], default= 'cuda:1')
parser.add_argument('--mode', choices = ['bev', 'cam'], default = 'cam')

# Training data config
parser.add_argument('--batch-size', type=int, default= 256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num-epochs', type=int, default=20)


# Logging config
parser.add_argument('--num-iters-per-log', type=int, default=100)

args = parser.parse_args()

main(args)
