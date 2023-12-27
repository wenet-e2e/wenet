#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-10-27] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import argparse
import logging
import os
import random
import socket
import subprocess
from threading import Thread
"""
Requirements:

```bash
apt install pdsh OR yum install pdsh
```

Usage:
```bash
num_nodes=2 python3 tools/ssh_launcher.py -n ${num_nodes} -H hostfile '
NCCL_SOCKET_IFNAME=eth0 bash run.sh \
  --stage 4 --stop_stage 4 \
  --HOST_NODE_ADDR ${HOST_NODE_ADDR}:26555 \
  --num_nodes ${num_node}
'
```

where `hostfile` looks like this:
```txt
servername1
servername2
```
OR
```txt
IP1
IP2
```

NOTE:
1. Ensure SSH passwordless login is enabled between different machines.
2. HOST_NODE_ADDR is automatically parsed in function `submit()` and passed to ENV

"""


def parse_hostfile(hostfile, convert_ip=True):
    hosts = []
    with open(hostfile) as f:
        for h in f.readlines():
            h = h.strip()
            if len(h) > 0:
                if not convert_ip:
                    hosts.append(h.strip())
                else:
                    try:
                        ip = socket.gethostbyname(h)
                        hosts.append(ip)
                    except Exception as e:
                        print("error host ", h, " error: ", e)
    return hosts


def find_available_ports(ip, port_num=1, port=9091, port_end=9999):
    max_retry = 100

    port_list = []
    for i in range(0, max_retry):
        if i + 1 == max_retry:
            raise Exception("faild to bind a port")
        port = random.randint(port, port_end)
        print("local port ", port)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", port))
            port_list.append(port)
            print("success port ", port)
            sock.close()
            if len(port_list) == port_num:
                break
        except socket.error:
            print("error port ", port)
            continue
    return port_list


def get_env(pass_envs):
    envs = []
    for k, v in list(pass_envs.items()):
        if "()" in k or "()" in v:
            continue
        envs.append("export " + str(k) + "=" + str(v) + ";")
    return " ".join(envs)


def submit(nworker, hostfile, port, sshport, cmd):
    if nworker <= 0:
        return
    ip_ports = []
    if hostfile is not None:
        hosts = parse_hostfile(hostfile)
        ip_ports = [h + ":" + str(port) for h in hosts]
    else:
        assert nworker == 1
        local_host = "127.0.0.1"
        hosts = [local_host] * nworker
        ports = find_available_ports(local_host, nworker)
        for p in ports:
            ip_ports.append(local_host + ":" + str(p))

    local_dir = os.getcwd() + "/"
    working_dir = local_dir

    # thread func to run the job
    def run(prog):
        print("launch prog ", prog)
        try:
            subprocess.check_call(prog, shell=True)
        except subprocess.CalledProcessError as e:
            logging.info("subprocess({}) failed({})! {}".format(
                e.cmd, e.returncode, e.output))
            os._exit(-1)

    pass_envs = os.environ.copy()
    if "HOST_NODE_ADDR" not in os.environ:
        pass_envs["HOST_NODE_ADDR"] = hosts[0]

    thread_list = []
    for i in range(nworker):
        node = hosts[i % len(hosts)]

        # ssh_port_arg = " -p " + str(sshport) + " "
        prog = get_env(pass_envs) + " cd " + working_dir + "; " + cmd
        prog = (
            "ssh -o StrictHostKeyChecking=no "
            # + ssh_port_arg   # no port available in aidi
            + node + " '" + prog + "'")
        thread = Thread(target=run, args=(prog, ))
        thread.setDaemon(True)
        thread.start()
        thread_list.append(thread)

    for t in thread_list:
        t.join()
        print("thread join success")
    print("process end success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nworker",
        type=int,
        required=True,
        help="number of worker process to be launched",
    )
    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        help="the hostfile of workers")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=9876,
        help="the port used for every worker, used when distribute-training",
    )
    parser.add_argument(
        "-sp",
        "--sshport",
        type=int,
        default=443,
        help="the port used for ssh connect, used when distribute-training",
    )
    parser.add_argument("command",
                        nargs="+",
                        help="command for plugin program")
    args = parser.parse_args()
    cmd = " ".join(args.command)
    submit(args.nworker, args.hostfile, args.port, args.sshport, cmd)
