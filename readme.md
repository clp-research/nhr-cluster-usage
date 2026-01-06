# NHR Cluster Usage

Full documentation about the usage of the cluster and the application process can be found here:

https://user.nhr.zib.de/wiki/spaces/PUB/pages/426682/Application+Guide

## Application

To apply for the account visit the NHR portal homepage at this link: https://portal.nhr.zib.de/

Generally, you can apply for two types of accounts:
1) Compute Project Account - A project account where multiple users can bill their usage to.
2) User Account - individual user account -- This is what students for their projects should apply for.

The application process is generally straight forward. Upon completing the form in the portal, an application document will be generated. This needs to be signed by the user, and then forwarded to Prof. David Schlangen to obtain his signature and stamp. Upon obtaining this, the document needs to be emailed to NHR at the address specified in the document. 

In order to access the account, you need to generate a password-protected ssh key, and upload it to the platform. Instructions can be found here: https://user.nhr.zib.de/wiki/spaces/PUB/pages/427064/SSH+Login


## Usage

User accounts are credited with 75000 core-h per quarter. Upon request, they can be credit with another 300k core-h each quarter. These additional core-h are added to the account upon request. For Uni Potsdam students, requests should be made to Dr. Nicholas Charron via email at <charron@zib.de>.

Usually, the last couple of days in the quarter the cluster undergoes maintenance, and cannot be accessed -- users will be notified of this in advance.

### Remote Usage

If you are using VScode, I recommend using the Remote - SSH plugin to set up a connection. It will allow you to access your workspace inside your IDE. Unlike Jarvis, NHR does not provide a web UI to interact with the files. 


### Partitions
Overview of costs and partitions can be found here: https://user.nhr.zib.de/wiki/spaces/PUB/pages/426811/Accounting

The amount of credit hours user accounts are credited with each quarter can go a long way. A single GPU costs 150 core-h per hour -- so the initial 75k core-h is sufficenet for about 500h of compute on a single A100 GPU.

Typically, each GPU Node contains 4 A100 GPUs. Based on how many GPUs you want to use, and for how long, you need to be aware of different GPU Partitions and their limitations. Full documentation is here: https://user.nhr.zib.de/wiki/spaces/PUB/pages/430577/Compute+partitions

Generally, there are 3 GPU partitions that you will be using:

1) gpu-a100:test -> small GPU partition with a time limit set to 1 hour -- this is where you should be testing your code before scheduling a longer job.
2) gpu-a100:shared -> Shared GPU partition -- you will use this if you plan to use fewer than 4 GPUs for your job. 
3) gpu-a100 -> You will use this if you want exclusive access to all 4 GPUs. 

The typical job wall is 24 hours, although there are ways to extend this to 48 Hours. NHR provides documentation on how to do this.

### Slurm

Access to the GPU is managed through slurm. Instructions on using slurm can be found here: https://user.nhr.zib.de/wiki/spaces/PUB/pages/426680/Slurm+usage

These are the most common slurm commands you will use:

1) salloc -> used to request resource allocation. Once you receive the resources, you can ssh into the allocated machine for interactive usage.
2) sbatch -> Used to submit a batch script and schedule a longer running job.
3) squeue -u USERNAME -> used to check the status of your batched job. Run without -u argument to see the generall queue. 
4) sinfo -> use to see the status of the compute nodes. 


### File Systems

Full documentation available here: https://user.nhr.zib.de/wiki/spaces/PUB/pages/426848/File+Systems

There is a limit to how many files and amount of storage you can use on different file systems. Use show-quota to check your usage. If you exceed it, you will not be able to write any new files untill you reduce your usage. 


### Multi GPU Training

In order to run multi-gpu training, you will need to prepare your training script to support it. There are several methods that you should look into:

1) Data Parallel
2) Model Parallel
3) Distributed Data Parallel (DDP)

DDP is the most common setup -- on each GPU, a complete instance of the model is loaded, while the data is sharded into batches, each GPU receiving it's batch of the dataset. 

Huggingface's Accelerate library is very easy to integrate and works quite nicely. Further information can be found here: https://huggingface.co/blog/pytorch-ddp-accelerate-transformers


### Internet Access

Compute nodes do not have access to any online services at all. While you can SSH into the compute node, you will not be allowed to make any API calls to services such as Huggingface, OpenAI, Wandb. 

Since the process doesn't immediately crash as the script waits for the service to respond, it can waste quite a few credits this way. Here are a couple of things to take into consideration:

1) Download the checkpoints from Huggingface into your cache folder before you sbatch your training script.
2) Do not push to HF Hub while training -- you need to do this after the training is completed.
3) Use WANDB and other loggers in offline mode. 
4) Use only local models -- no API service will work and there are no workarounds. 
