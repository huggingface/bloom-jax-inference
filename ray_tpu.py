# copied from https://github.com/kingoflolz/mesh-trzansformer-jax

import functools
import os
import subprocess
import time

import glob
import requests
from fabric import Connection
from invoke import run


@functools.lru_cache()
def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
    return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
        "utf-8").strip()


def copy_files_on_tpu(path, dest, is_dir=False):
    if is_dir:
        command = f"gcloud compute tpus tpu-vm scp --recurse ***REMOVED***path***REMOVED*** suraj-tpu-v3-32:***REMOVED***dest***REMOVED*** --quiet --force-key-file-overwrite --strict-host-key-checking=no --worker=all --zone=europe-west4-a"
    else:
        command = f"gcloud compute tpus tpu-vm scp ***REMOVED***path***REMOVED*** suraj-tpu-v3-32:***REMOVED***dest***REMOVED*** --worker=all --zone=europe-west4-a"
    
    return subprocess.check_output(command, shell=True).decode("utf-8").strip()

def run_command_on_tpu(command):
    command = f"gcloud compute tpus tpu-vm ssh suraj-tpu-v3-32 --zone=europe-west4-a --quiet --force-key-file-overwrite --strict-host-key-checking=no --worker=0 --command='***REMOVED***command***REMOVED***'"
    return subprocess.check_output(command, shell=True).decode("utf-8").strip()


def create_tpu(
        name,
        zone,
        type,
        preemptible,
):
    headers = ***REMOVED***
        'Authorization': f'Bearer ***REMOVED***get_bearer()***REMOVED***',
        'Content-Type': 'application/json',
    ***REMOVED***

    try:
        status = check_tpu(name, zone)

        if status["state"] not in ["CREATING", "READY"]:
            print("deleting TPU")
            delete_tpu(name, zone)

            while True:
                try:
                    print("deleting check")
                    print(check_tpu(name, zone)["state"])

                    time.sleep(1)
                except:
                    break
    except:
        pass

    params = (
        ('node_id', name),
    )

    data = ***REMOVED***"accelerator_type":
                type,
            "runtime_version":
                'v2-alpha',
            "network_config":
                ***REMOVED***"enable_external_ips": True***REMOVED***,
            ***REMOVED***

    if preemptible:
        data["schedulingConfig"] = ***REMOVED***"preemptible": True***REMOVED***

    response = requests.post(f'https://tpu.googleapis.com/v2alpha1/projects/***REMOVED***get_project()***REMOVED***/locations/***REMOVED***zone***REMOVED***/nodes',
                             headers=headers, params=params, json=data)

    print(response.json())

    return response.status_code == 200


def check_tpu(name, zone):
    headers = ***REMOVED***
        'Authorization': f'Bearer ***REMOVED***get_bearer()***REMOVED***',
    ***REMOVED***

    response = requests.get(
        f'https://tpu.googleapis.com/v2alpha1/projects/***REMOVED***get_project()***REMOVED***/locations/***REMOVED***zone***REMOVED***/nodes/***REMOVED***name***REMOVED***',
        headers=headers)

    return response.json()


def delete_tpu(name, zone):
    headers = ***REMOVED***
        'Authorization': f'Bearer ***REMOVED***get_bearer()***REMOVED***',
    ***REMOVED***

    response = requests.delete(
        f'https://tpu.googleapis.com/v2alpha1/projects/***REMOVED***get_project()***REMOVED***/locations/***REMOVED***zone***REMOVED***/nodes/***REMOVED***name***REMOVED***',
        headers=headers)

    return response.json()


def wait_til(name, zone, state):
    while True:
        ret = check_tpu(name, zone)

        print("wait_til check")
        print(ret)

        matches = True
        for k, expected_v in state.items():
            if k not in ret:
                matches = False
                continue
            if ret[k] != expected_v:
                matches = False

        if "error" in ret:
            return False

        if ret["state"] == "TERMINATED":
            return False

        if matches:
            return True

        time.sleep(1)


def get_connection(
        name,
        zone,
):
    info = check_tpu(name, zone)
    outputs = []
    for i in info["networkEndpoints"]:
        outputs.append(Connection(i["ipAddress"],
                                  connect_kwargs=***REMOVED***
                                      "key_filename": os.path.expanduser('~/.ssh/google_compute_engine'), ***REMOVED***))
    return outputs


def start_ray(conn, address):
    conn.run('sudo rm -rf *.py bloom_inference')
    # run_command_on_tpu("sudo rm -rf *.py bloom_inference")

    conn.run("mkdir bloom_inference bloom_inference/bloom_inference -p")
    # run_command_on_tpu("mkdir bloom_inference -p")
    
    for i in glob.glob("*.py"):
        conn.put(i, "bloom_inference/")

    for i in glob.glob("bloom_inference/*.py"):
        conn.put(i, "bloom_inference/bloom_inference/")

    # copy_files_on_tpu("bloom_inference/*.py", "bloom_inference/")

    # conn.run('sudo pip install -e bloom_inference/', hide=True)
    # conn.run('python3 setup.py install --user', hide=True)

    conn.put("scripts/ray_tpu.sh", "/tmp/ray-tpu.sh")
    conn.sudo('chmod +x /tmp/ray-tpu.sh', hide=True)
    try:
        conn.run('ray stop -f', hide=True)
        # run_command_on_tpu("ray stop -f")
    except:
        pass
    
    time.sleep(1)
    
    out = conn.run(f'bash /tmp/ray-tpu.sh ***REMOVED***address***REMOVED***', hide=True)
    print(out)
    # copy_files_on_tpu("scripts/ray_tpu.sh", "/tmp/ray-tpu.sh")
    # run_command_on_tpu("sudo chmod +x /tmp/ray-tpu.sh")
    # run_command_on_tpu(f"sudo /tmp/ray-tpu.sh ***REMOVED***address***REMOVED***")