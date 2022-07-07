# bloom-inference-jax

## Setting Up a TPU-Manager
The TPU hosts are managed by a single TPU manager. This TPU manager takes the form of a single CPU device.

First, create a CPU VM in the **same region** as that of the TPU pod. This is important to enable the TPU manager to communicate with the TPU hosts. A suitable device config is as follows: 
   1. Region & Zone: TO MATCH TPU ZONE
   2. Machine type: c2-standard-8
   3. CPU platform: Intel Cascade Lake
   4. Boot disk: 256GB balanced persistent disk

SSH into the CPU and set-up a Python environment with the **same Python version** as that of the TPUs. The default TPU Python version is 3.8.10.

```
python3 -m venv /path/to/venv
```
If the above does not work, run the following and then repeat:
   
```
sudo apt-get update
sudo apt-get install python3-virtualenv
```

Check Python version is 3.8.10:
```
python --version
```

Clone the repository and install requirements:
```
git clone https://github.com/huggingface/bloom-jax-inference.git
cd bloom-jax-inference
pip install -r requirements.txt
```


Authenticate `gcloud`, which will require copy-and-pasting a command into a terminal window on a machine with a browser installed:
```
gcloud auth login
```

Now SSH into one of the workers. This will generate an SSH key:
```
gcloud alpha compute tpus tpu-vm ssh patrick-tpu-v3-32 --zone europe-west4-a --worker 0
```
