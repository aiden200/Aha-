## Instructions for Cloud distributed training(Paperspace)

### Machines

Make sure to create everything in the same region. I used `West Coast (CA1)`.

1. Create 1x Private network. Assign both computers to the private network when creating the machines.
2. Create 8x nodes of `P4000x2` (multi-GPU) with `ML-in-a-Box` as operating system
3. Create 1 Network drive (250 GB)


### S3 Setup
I'm keeping my data in a AWS S3 bucket since we have around ~5TB worth of data. We want to mount a S3 bucket to our local device. We're going to use `boto3`, since we are training and streaming data 

1. Start by installing s3fs
   ```bash
   sudo apt update
   sudo apt install -y s3fs
   ```
2. Set AWS credentials
   ```bash
   echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ~/.passwd-s3fs
   chmod 600 ~/.passwd-s3fs
   ```
3. Create a Mount point
   ```bash
   mkdir ~/my_s3_mount
   ```
4. Mount your S3 Bucket
   ```bash
   s3fs your-bucket-name ~/my_s3_mount -o use_path_request_style -o url=https://s3.amazonaws.com -o allow_other
   ```
   You can also unmount your bucket using
   ```bash
   fusermount -u ~/my_s3_mount
   ```


### Setup

Login on each machine and perform the following operations:

1. `sudo apt-get update`
2. `sudo apt-get install net-tools`
3. If you get an error about `seahorse` while installing `net-tools`, do the following:
   1. sudo rm /var/lib/dpkg/info/seahorse.list
   2. sudo apt-get install seahorse --reinstall
4. Get each machine's private IP address using `ifconfig`
5. Add IP and hostname mapping of all the slave nodes on `/etc/hosts` file of the master node
6. Mount the network drive
   1. `sudo apt-get install smbclient`
   2. `sudo apt-get install cifs-utils`
   3. `sudo mkdir /mnt/training-data`
   4. Replace the following values on the command below:
      1. `NETWORD_DRIVE_IP` with the IP address of the network drive
      2. `NETWORK_SHARE_NAME` with the name of the network share
      3. `DRIVE_USERNAME` with the username of the network drive
   5. `sudo mount -t cifs //NETWORD_DRIVE_IP/NETWORK_SHARE_NAME /mnt/training-data -o uid=1000,gid=1000,rw,user,username=NETWORK_DRIVE_USERNAME`
      1. Type the drive's password when prompted
7. `git clone https://github.com/hkproj/pytorch-transformer-distributed`
8. `cd pytorch-transformer-distributed`
9. `pip install -r requirements.txt`
10. Login on Weights & Biases
    1. `wandb login`
    2. Copy the API key from the browser and paste it on the terminal
11. Run the training command from below


For each machine, perform the following operations:




### Training

Run the following command on any machine. Make sure to not run it on both, otherwise they will end up overwriting each other's checkpoints.

We'll specify some terminology and instructions here:
- A cluster is a combination of nodes. In our case we are using 1 cluster and 4 nodes.
- A node is a machine on paperspace, which is a single machine with 1 or more GPUs + CPUs
- `nproc_per_node` is the number of GPUs the specified node is using. In our case, we are using 2
- `nnodes` is the number of nodes we are using. In our case, it's 4
- `node_rank` represents the rank of the node we are using. The master node should be rank 0.
- `rdzv_endpoint` represents the ip:port of the master node coordinating the training.

For the master node, run:

`torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:48123 train.py --batch_size 8 --model_folder "/mnt/training-data/weights"`


Run the following command on each slave node (all nodes excluding the master node) (replace `IP_ADDR_MASTER_NODE` with the IP address of the master node):

`torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=IP_ADDR_MASTER_NODE:48123 train.py --batch_size 8 --model_folder "/mnt/training-data/weights"`

### Monitoring

Login to Weights & Biases to monitor the training progress: https://app.wandb.ai/