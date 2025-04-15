## Instructions for Cloud distributed training (Paperspace)

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
2. Create 1 node of `A6000x2` (multi-GPU), 250GB as the drive, with `ML-in-a-Box` as operating system - make sure to select the private network created.
3. Create 1 Network drive (250 GB)


### S3 Mount setup
These are instructions if your data is in a S3 bucket.

1. Login to your machine and identify your user and group-id:
   ```bash
   id -u #UID
   id -g #GID
   ```

2. Install CIFS tools if not installed
   ```bash
   sudo apt update
   sudo apt install cifs-utils
   ```

3. Create a local Mount Point and give access to user:
   ```bash
   sudo mkdir -p /mnt/training-data
   sudo chown paperspace:paperspace /mnt/training-data
   ```

4. Install the S3 Mount application
   ```bash
   wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
   sudo apt-get install -y ./mount-s3.deb
   ```
5. Once you've got Mountpoint for Amazon S3 installed, you can mount your Amazon S3 bucket. You'll need valid AWS credentials to access your bucket. Mountpoint will automatically use credentials from an IAM role associated with your EC2 instance, or the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables. 
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-west-1  # or whatever region your bucket is in
   ```

6. To mount your bucket, run this command, replacing `amzn-s3-demo-bucket` with the name of your bucket:
   ```bash
   mount-s3 --uid=UID --gid=GID amzn-s3-demo-bucket /mnt/training-data
   ```

7. Check if you can access S3:
   ```bash
   ls /mnt/training-data
   ```

8. Change your `video_root` commands in `configs/datasets` to reflect the new data path

9. When you are finished, you can unmount your bucket
   ```bash
   umount /mnt/training-data
   ```

10. *Optional but HEAVILY RECOMMENDED:* Move your S3 bucket to the East Coast Region
      ```bash
      aws s3api create-bucket --bucket [NEW bucket name] --region us-east-1
      aws s3 sync s3://[old bucket] s3://[new bucket]
      ```
      Mount new bucket
      ```bash
      sudo umount /mnt/training-data  # Unmount the old one (if mounted)
      mount-s3 --uid=1000 --gid=1000 [new bucket] /mnt/training-data
      ```   

11. *Optional*: Mount S3 bucket on boot:
   
      1. Create systemd service:
         ```bash
         sudo vim /etc/systemd/system/mount-s3.service
         ```
      2. Paste the following (Update information first):
         ```bash
         [Unit]
         Description=Mount Amazon S3 bucket with Mountpoint
         After=network-online.target
         Wants=network-online.target

         [Service]
         Type=simple
         User=paperspace
         ExecStart=/usr/bin/mount-s3 --foreground --uid=1000 --gid=1000 [your bucket] /mnt/training-data
         Restart=no
         Environment=HOME=/home/paperspace
         Environment=AWS_ACCESS_KEY_ID=your_access_key
         Environment=AWS_SECRET_ACCESS_KEY=your_secret_key
         Environment=AWS_DEFAULT_REGION=us-east-1
         WorkingDirectory=/home/paperspace

         [Install]
         WantedBy=multi-user.target
         ```
      3. Reload and enable the service:
         ```bash
         sudo systemctl daemon-reexec
         sudo systemctl daemon-reload
         sudo systemctl enable mount-s3.service
         ```
      4. Start the service:
         ```bash
         sudo systemctl start mount-s3.service
         ```
      5. Monitor status:
         ```bash
         sudo systemctl status mount-s3.service
         ```



### Setup

Login to the first machine using `ssh paperspace@[IP addr]`. Make sure you have your public ssh key set up on paperspace before performing this operation! We are going to set up everything on the first machine, and create a snapshot we can use for the other 3 machines so we don't have to perform this setup again. 

perform the following operations:

1. `sudo apt-get update`
2. `sudo apt-get install net-tools`
3. If you get an error about `seahorse` while installing `net-tools`, do the following:
   1. sudo rm /var/lib/dpkg/info/seahorse.list
   2. sudo apt-get install seahorse --reinstall
<!-- 4. Get each machine's private IP address using `ifconfig`
5. Add IP and hostname mapping of all the slave nodes on `/etc/hosts` file of the master node -->
4. Transfer the code base onto the machine:
   ```
   rsync -av --progress --ignore-existing [AHA DIRECTORY] paperspace@[IP ADDR]:/home/paperspace/Documents
   ```
5. Configure the setup on the machine
   
   **Note:**: Make sure your pip and python is pointed towards your conda environment!  
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
   3. Create a conda environment and install cuda.:
      ```bash
      conda create -n aha python=3.10
      conda activate aha
      conda install nvidia/label/cuda-[CUDA VERSION]::cuda
      ```
      You can check out the [CUDA Versions here](https://anaconda.org/nvidia/cuda).
   4. Install packages:
      ```
      python -m pip install -r requirements.txt
      cd LLaVA_NeXT
      python -m pip install -e ".[train]"
      cd ..
      ```
      
      <details>

      <summary>If you get a NVIDIA driver too old error, these are the upgrade instructions </summary>

      Delete the current driver:
      ```bash
      sudo apt-get --purge remove "*nvidia*"
      sudo apt autoremove
      sudo reboot now
      ```

      Verify that the driver is no longer installed by running the command `nvidia-smi`


      Reinstall the driver:
      ```bash
      sudo apt update
      sudo apt upgrade
      sudo apt install ubuntu-drivers-common
      sudo ubuntu-drivers list
      sudo apt install -y nvidia-driver-535 (or whatever the latest nvidia-driver-### is)
      sudo reboot now
      ```

      Verify that the new version driver is installed by running the command `nvidia-smi`

      </details>
   5. Install pytorch (run this command before and after 4 if you get an error about the cuda driver being too old):
      ```
      python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu124
      ```
   6. Verify it worked:
      ```
      python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
      ```
      You should get: `2.5.1+cu124 True 12.4`
   7. Install Flash attention:
      ```bash
      MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation --no-cache-dir 
      ```
6. Save this configuration as a template on paperspace, so you can initialize a machine without this whole setup. 
7. Go back to paperspace, spin up 3 other instances, using the same template. Remember to create these machines on the same private network, 250GB worth of disk space, and using the 2xA6000 GPUs nodes. We are now ready to perform distributed training.



### Node communication setup

For each machine, perform the following operations:

1. Get each machine's private IP address using `ifconfig`.
2. Get each machine's hostname using `hostname`

3. Open `/etc/hosts` and paste every other machines IP addr and hostname. For example, if we had 4 nodes initialized, node 2's `etc/hosts` file should look something like this:

   ```
   127.0.0.1   localhost
   127.0.1.1   HOSTNAME
   10.5.7.2    psucgene4   // Node 0
   10.4.5.6    psuchdr4    // Node 1
   10.3.4.54   dlkjfslk    // Node 3
   ```

   Obviously, the IP addr and hostname would be different, this is just an example. Awesome, we should be ready to train now.

### Training

Run the following command on any machine. Make sure to not run it on multiple machines, otherwise they will end up overwriting each other's checkpoints.

We'll specify some terminology and instructions here:
- A cluster is a combination of nodes. In our case we are using 1 cluster and 4 nodes.
- A node is a machine on paperspace, which is a single machine with 1 or more GPUs + CPUs
- `nproc_per_node` is the number of GPUs the specified node is using. In our case, we are using 2
- `nnodes` is the number of nodes we are using. In our case, it's 4
- `node_rank` represents the rank of the node we are using. The master node should be rank 0.
- `rdzv_endpoint` represents the ip:port of the master node coordinating the training.

For the master node, run:



```bash
PYTHONWARNINGS="ignore" torchrun --nproc_per_node 2 --nnodes=4 --node_rank=0 --master_port 29506 \
   --rdzv_id=456 --rdzv_backend=static --rdzv_endpoint=127.0.0.1:48123 \
   train.py --deepspeed configs/deepspeed/zero2.json \
    --bf16 true --tf32 true \
    --dataset_config configs/datasets/actual_paperspace_configuration.json \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
    --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 --gradient_checkpointing true \
    --evaluation_strategy no --prediction_loss_only false \
    --save_strategy steps --save_steps 500 --save_total_limit 5 \
    --learning_rate 0.00002 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --report_to wandb \
    --output_dir outputs/aha \
    > outputs/aha/train.log
```


Run the following command on each slave node (all nodes excluding the master node) (replace `IP_ADDR_MASTER_NODE` with the IP address of the master node) (replace NODE_RANK with the machine's node rank):


```bash
PYTHONWARNINGS="ignore" torchrun --nproc_per_node 2 --nnodes=4 --node_rank=NODE_RANK --master_port 29506 \
   --rdzv_id=456 --rdzv_backend=static --rdzv_endpoint=IP_ADDR_MASTER_NODE:48123 \
   train.py --deepspeed configs/deepspeed/zero2.json \
    --bf16 true --tf32 true \
    --dataset_config configs/datasets/actual_paperspace_configuration.json \
    --llm_pretrained lmms-lab/llava-onevision-qwen2-7b-ov \
    --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 --gradient_checkpointing true \
    --evaluation_strategy no --prediction_loss_only false \
    --save_strategy steps --save_steps 100 --save_total_limit 5 \
    --learning_rate 0.00002 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --logging_steps 1 \
    --report_to wandb \
    --output_dir outputs/aha \
    > outputs/aha/train.log
```

### Monitoring

Login to Weights & Biases to monitor the training progress: https://app.wandb.ai/