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
    return subprocess.check_output('gcloud config list --format "value(core.project)"', shell=True).decode(
        "utf-8").strip()


def check_tpu(name, zone):
    headers = ***REMOVED***
        "Authorization": f"Bearer ***REMOVED***get_bearer()***REMOVED***",
    ***REMOVED***

    response = requests.get(
        f"https://tpu.googleapis.com/v2alpha1/projects/***REMOVED***get_project()***REMOVED***/locations/***REMOVED***zone***REMOVED***/nodes/***REMOVED***name***REMOVED***",
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
                                  connect_kwargs=***REMOVED***
                                      "key_filename": os.path.expanduser("~/.ssh/google_compute_engine"), ***REMOVED***))
    return outputs


def start_ray(conn, address):
    # start afresh each launch (temporarily)
    conn.run("sudo rm -rf *.py bloom-jax-inference")
    # make directory of structure: bloom_inference/bloom_inference/modeling_bloom
    conn.run("mkdir bloom-jax-inference bloom-jax-inference/bloom_inference bloom-jax-inference/bloom_inference/modeling_bloom -p")

    # copy run files into bloom_inference
    for i in glob.glob("*.py"):
        conn.put(i, "bloom-jax-inference/")

    # copy CPU/TPU manager files into bloom_inference/bloom_inference
    for i in glob.glob("bloom_inference/*.py"):
        conn.put(i, "bloom-jax-inference/bloom_inference/")

    # copy modeling files into bloom_inference/bloom_inference/modeling_bloom
    for i in glob.glob("bloom_inference/modeling_bloom/*.py"):
        conn.put(i, "bloom-jax-inference/bloom_inference/modeling_bloom/")

    # copy modeling files into bloom_inference/bloom_inference/modeling_bloom
    for i in glob.glob("*.sh"):
        conn.put(i, "bloom-jax-inference/")

    # copy key files into bloom_inference
    conn.put("key.json", "bloom-jax-inference/")

    # transfer start-up script from CPU -> hosts and give permissions
    conn.put("scripts/ray_tpu.sh", "/tmp/ray-tpu.sh")
    conn.sudo("chmod +x /tmp/ray-tpu.sh", hide=True)

    try:
        conn.run("ray stop -f", hide=True)
    except:
        pass
    
    time.sleep(1)
    
    # run start-up script
    out = conn.run(f"bash /tmp/ray-tpu.sh ***REMOVED***address***REMOVED***", hide=False)
    # display result
    print(out)
