export SLURM_ACCOUNT=bbkn-holli
export SLURM_TIMELIMIT=01:00:00
csrun_wse --single-task-nodes 2 --cyclic python-ws ./mnist_estimator_1_cs.py --mode train --cs_ip ${CS2} --params config/params_1.yaml
