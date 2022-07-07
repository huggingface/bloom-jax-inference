# copied from https://github.com/kingoflolz/mesh-trzansformer-jax

import functools
import os
import subprocess
import time

import glob
import requests
from fabric import Connection


@functools.lru_cache()
def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
    return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
        "utf-8").strip()


def check_tpu(name, zone):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)

    return response.json()


def get_connection(
        name,
        zone,
):
    info = check_tpu(name, zone)
    outputs = []
    for i in info["networkEndpoints"]:
        outputs.append(Connection(i["ipAddress"],
                                  connect_kwargs={
                                      "key_filename": os.path.expanduser('~/.ssh/google_compute_engine'), }))
    return outputs


def start_ray(conn, address):
    # start afresh each launch (temporarily)
    conn.run('sudo rm -rf *.py bloom_inference')
    conn.run("mkdir bloom_inference bloom_inference/bloom_inference -p")
    
    # copy files into correct dirs
    for i in glob.glob("*.py"):
        conn.put(i, "bloom_inference/")

    for i in glob.glob("bloom_inference/*.py"):
        conn.put(i, "bloom_inference/bloom_inference/")

    # transfer start-up script from CPU -> hosts
    conn.put("scripts/ray_tpu.sh", "/tmp/ray-tpu.sh")
    conn.sudo('chmod +x /tmp/ray-tpu.sh', hide=True)

    try:
        conn.run('ray stop -f', hide=True)
        # run_command_on_tpu("ray stop -f")
    except:
        pass
    
    time.sleep(1)
    
    # run start-up script
    out = conn.run(f'bash /tmp/ray-tpu.sh {address}', hide=True)
    # display result
    print(out)
