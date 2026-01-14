# Copying files to the remote machine
adjust the scp script `copy_files.sh` by putting your ssh key instead of YOUR-SSH-KEY field, your actual path of the folder  (YOUR_LOCAL_PATH) and the user name (YOUR-USER-NAME). Change these fields and upload the files under the folder as the remote machine `does not allow downloading files`.



# How to run
`sbatch run.sh`

# Python script
finetunes a language model using specified configurations and resources.

# Partition Information for SLURM Jobs
More info here [NHR-Documentation](https://user.nhr.zib.de/wiki/spaces/PUB/pages/430579/Slurm+partition+GPU+A100)

Line 2 of the SLURM job script contains the partition information: ```#SBATCH --partition=gpu-a100:test```

The run.sh script uses `gpu-a100:test` - which is the test partition.

Available partitions are:
- `gpu-a100`: `34` nodes, where each node has `4` A100 GPUs
- `gpu-a100:test`: `2` nodes, where each node has `4` A100 GPUs
- `gpu-a100:shared`: `5` nodes, where each node has `4` A100 GPUs (shared access)


# Email Notification
To receive email notifications about the job status, you can add the following lines to your SLURM job script: `#SBATCH --mail-user=YOUR_EMAIL_ADDRESS` and `#SBATCH --mail-type=ALL`. Replace `YOUR_EMAIL_ADDRESS` with your actual email address. This will send you emails for all job events, including job start, end, and failure.