## Instructions for Cloud distributed training(Paperspace)

### Terminology
- Nodes: Defined as "machines" in paperspace. Nodes can contain multiple GPUs and CPUs
- Master node: the primary node responsible for updating weights. All other nodes will communicate to this node.
- Slave node: every node which isn't a master node
- Global rank: The rank of the GPU instance across all nodes. If we have 4 nodes with each node containg 2 GPUs, we have 4 x 2 = 8 GPUs. The first GPU will be global rank 0, the last will be global rank 8.
- Local rank: The rank of the GPU instance within the node. In our case, the local rank will be 0 or 1 since each node only contains 2 GPUs
- Paperspace: the platform we will be training our model on


### Machines

Make sure to create everything in the same region. I used `East Coast (NY2)`. I would advice to do the same, since all the powerful clusters are only available in the NY2 region.

1. Create 1x Private network. Assign all machines to the private network upon creation.
2. Create 4x nodes of `A6000x2` (multi-GPU) with `ML-in-a-Box` as operating system - make sure to select the private network created.
3. Create 1 Network drive (250 GB)


### Paperspace Mount Setup
We're going to transfer the local data into paperspace. To do this we are going to need to mount the network drive onto our computer and transfer the data.

1. Install CIFS tools if not installed
```bash
sudo apt update
sudo apt install cifs-utils
```

2. Create a local Mount Point:
```bash
sudo mkdir -p /mnt/paperspace
```

3. Mount the paperspace share:
```bash
sudo mount -t cifs //<network-address>/<share-name> /mnt/paperspace -o username=<your-username>,password=<your-password>,uid=1000,gid=1000
```

4. Copy your local data:
```bash
cp -r /path/to/your/local/data /mnt/paperspace/
```

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

Login to the first machine using `ssh paperspace@[IP addr]`. We are going to set up everything here, and create a snapshot we can use for the other 3 machines so we don't have to perform this setup again. 

perform the following operations:

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
      2. To mount the machine each time its started, open up `etc/fstab` using something like
      ```sudo vim /etc/fstab``` then add in the line `//NETWORD_DRIVE_IP/NETWORK_SHARE_NAME /mnt/training-data cifs username=NETWORK_DRIVE_USERNAME,password=NETWORK_DRIVE_PASSWORD,uid=1000,gid=1000,rw 0 0`
7. Transfer the data into the mounted drive: 
   ```
   rsync -av --progress --ignore-existing /path/to/local/videos/ paperspace@<VM_IP>:/mnt/training-data/
   ```
8. Transfer the code base onto the machine:
   ```
   rsync -av --progress --ignore-existing [AHA DIRECTORY] paperspace@[IP ADDR]:/home/paperspace/Documents
   ```
9. Configure the setup on the machine
   1. Install miniconda:
      ```
      # Download Miniconda
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

      # Run installer
      bash Miniconda3-latest-Linux-x86_64.sh
      ```
   2. Let Conda activate automatically
      ```bash
      ~/miniconda3/bin/conda init
      source ~/.bashrc
      ```
   3. Create a conda environment & install pytorch:
      ```
      conda create -n aha python=3.10
      conda activate aha
      python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu124
      ```
   4. Verify it worked:
      ```
      python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
      ```
      You should get: `2.5.1+cu124 True 12.4`
   5. Install packages:
      ```
      python -m pip install -r requirements.txt
      cd LLaVA_NeXT
      python -m pip install -e ".[train]"
      cd ..
      ```

     
7. `git clone https://github.com/hkproj/pytorch-transformer-distributed`
8. `cd pytorch-transformer-distributed`
9. `pip install -r requirements.txt`
10. Login on Weights & Biases
    1. `wandb login`
    2. Copy the API key from the browser and paste it on the terminal
11. Run the training command from below

### Node communication setup

For each machine, perform the following operation:

Open `/etc/hosts` and paste every other machines IP addr and hostname. You can obtain the IP by running `ifconfig` and hostname with `hostname`. For example, if we had 4 nodes initialized, node 2's `etc/hosts` file should look something like this:

```
127.0.0.1   localhost
127.0.1.1   HOSTNAME
10.5.7.2    psucgene4   // Node 0
10.4.5.6    psuchdr4    // Node 1
10.3.4.54   dlkjfslk    // Node 3
```

Obviously, the IP addr and hostname would be different. Awesome, we should be ready to train now.

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