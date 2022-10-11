export SLURM_ACCOUNT=bblg-holli
export SLURM_TIMELIMIT=02:00:00
csrun_wse --single-task-nodes 2 --total-nodes 4 --cyclic python ./mnist_estimator_1_cs.py --mode train --cs_ip ${CS2} --params config/params_1.yaml
