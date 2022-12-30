import os, sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--keep_ratio', type=float, default=0.1)
parser.add_argument('--branch_point', type=int, default=20)
parser.add_argument('--gpu', type=str, default='gypsum-1080ti')

args = parser.parse_args() 

cmd = '#!/bin/sh\n'
cmd += f'#SBATCH --job-name={args.branch_point}/{args.keep_ratio}\n'
# cmd += f'#SBATCH --job-name=f/{args.keep_ratio}\n'
cmd += f'#SBATCH -o assets/slurm/slurm-%j.out  # %j = job ID\n'
cmd += f'#SBATCH --partition={args.gpu}\n'
cmd += f'#SBATCH --gres=gpu:1\n'
cmd += f'#SBATCH -c 6\n'
cmd += f'#SBATCH --mem 20GB\n' 

# cmd += f'python train_flat_model.py --keep_ratio={args.keep_ratio} --num_workers=3'
cmd += f'python train_branched_model.py --keep_ratio={args.keep_ratio} --branch_point={args.branch_point} --num_workers=3'

print(cmd)

with open('job_train.sh', 'w') as f:
    f.write(cmd)

os.system('sbatch job_train.sh')