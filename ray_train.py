import numpy as np
import ray

import argparse
import os
import sys
from six.moves import shlex_quote

import cv2
import go_vncdriver
import logging
import signal
import time
from a3c import A3C
from envs import create_env
import distutils.version
import tensorflow as tf

from worker import run, FastSaver, cluster_spec
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('-w', '--num-workers', default=1, type=int,
                        help="Number of workers")
    parser.add_argument('-r', '--remotes', default=None,
                        help='The address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
    parser.add_argument('-e', '--env-id', type=str, default="PongDeterministic-v3",
                        help="Environment id")
    parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                        help="Log directory path")

    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")
    args = parser.parse_args()
    ray.init(num_workers=args.num_workers + 1)

    args.job_name = "server"
    args.task = 0
    server = start.remote(args)
    workers = []
    args.job_name = "worker"
    for i in range(args.num_workers):
        args.task = i
        workers.append(start.remote(args))

    ray.get(server)

@ray.remote
def start(args):
    """
Setting up Tensorflow for data parallel work
"""
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    main()
