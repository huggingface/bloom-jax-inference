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
    headers = {
        "Authorization": f"Bearer {get_bearer()}",
    }

    response = requests.get(
        f"https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}",
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
                                      "key_filename": os.path.expanduser("~/.ssh/google_compute_engine"), }))
    return outputs


def start_ray(conn, address):
    # start afresh each launch (temporarily)
    conn.run("sudo rm -rf *.py bloom_inference")
    # make directory of structure: bloom_inference/bloom_inference/modeling_bloom
    conn.run(
        "mkdir bloom_inference bloom_inference/bloom_inference bloom_inference/scripts bloom_inference/bloom_inference/modeling_bloom -p")

    # copy run files into bloom_inference
    for i in glob.glob("*.py"):
        conn.put(i, "bloom_inference/")

    # copy CPU/TPU manager files into bloom_inference/bloom_inference
    for i in glob.glob("bloom_inference/*.py"):
        conn.put(i, "bloom_inference/bloom_inference/")

    # copy CPU/TPU manager files into bloom_inference/bloom_inference
    for i in glob.glob("scripts/*.sh"):
        conn.put(i, "bloom_inference/scripts/")

    # copy modeling files into bloom_inference/bloom_inference/modeling_bloom
    for i in glob.glob("bloom_inference/modeling_bloom/*.py"):
        conn.put(i, "bloom_inference/bloom_inference/modeling_bloom/")

    # transfer start-up script from CPU -> hosts and give permissions
    conn.sudo("chmod +x bloom_inference/scripts/ray_tpu.sh", hide=True)

    try:
        conn.run("ray stop -f", hide=True)
    except:
        pass

    time.sleep(1)

    # run start-up script
    out = conn.run(f"bash bloom_inference/scripts/ray_tpu.sh {address}", hide=True)
    # display result
    print(out)
