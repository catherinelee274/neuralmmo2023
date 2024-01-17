# Neural MMO Competition 2023

## Test that training runs
python train.py --local-mode true

## Run training. This is very memory intensive!
## You can change --num-envs  and --rollout-batch-size to adjust memory usage
## Also check out --device and --seed
## The checkpoints are saved under --runs_dir with --run_name
python train.py --run-name <YOUR_RUN_NAME> --device <YOUR_DEVICE> --seed <YOUR_NUMBER> --num-envs 1 --rollout-batch-size 2**14

## Evaluate checkpoints. After training, copy your checkpoints into policies
## The below command will compare your checkpoints against the baseline policy
python evaluate.py -p policies

## To generate a replay, create a directory with your checkpoints then run
python evaluate.py -p <YOUR_DIR> -r


NMMO Baseline Repository:
├── reinforcement_learning
│   ├── config.py
│   └── policy.py --> Your policy goes here
├── requirements.txt
└── train.py --> Train your policy here
