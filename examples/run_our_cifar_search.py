import pdb

import pandas as pd
from tensorflow.python.client import device_lib
import os
import subprocess
import time
import multiprocessing
from itertools import product
import numpy as np



def get_available_cpus():
    return multiprocessing.cpu_count()

def run_our_cpu():
    result_dir = 'results'
    dataset = 'cifar10'
    available_cpus = get_available_cpus()
    num_cpus = available_cpus
    delta_gd = [0.1, 1, 2, 5]

    commands = []
    for delta in product(delta_gd):
        cmd = f'python run_our_{dataset}.py --delta {delta[0]}'
        commands.append(cmd)

    procs_by_cpu = [None] * num_cpus

    while len(commands) > 0:
        for idx in range(num_cpus):
            proc = procs_by_cpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this CPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(cmd, shell=True)
                procs_by_cpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for all processes to complete
    for proc in procs_by_cpu:
        if proc is not None:
            proc.wait()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpus

def run_our():
    result_dir = 'results'
    dataset = 'cifar10'
    available_gpus = get_available_gpus()
    num_gpus = len(available_gpus)
    delta_gd = [1]


    commands = []
    for delta in product(delta_gd):
        # file_name = f'{result_dir}/ours_{dataset}_resnet_mb128_lr{lr}_delta{delta}_rf{freq}_s1.csv'
        # if os.path.exists(file_name):
        #     df = pd.read_csv(file_name)
        #     if df.shape[0] >= 100:
        #         continue
        cmd = f'python run_our_{dataset}.py --delta {delta}'

        commands.append(cmd)

    procs_by_gpu = [None] * num_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for all processes to complete
    for proc in procs_by_gpu:
        if proc is not None:
            proc.wait()


def check_search():
    check_dir = 'results'
    dataset = 'cifar100'
    method = 'ours'
    lrs = [0.1, 0.01]
    delta_gd = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    resize_freq = [20, 10, 5, 1]

    for lr, delta, freq in product(lrs, delta_gd, resize_freq):
        file_name = f'{check_dir}/{method}_{dataset}_resnet_mb128_lr{lr}_delta{delta}_rf{freq}_s1.csv'
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            if df.shape[0] < 100:
                print(f'{lr} {delta} {freq} {df.shape[0]}')
            # print(f'{delta} {freq} {df.shape[0]}')
        else:
            print(f'lr {lr}, delta {delta}, freq {freq} not found')


def check_seed():

    # cifar10 combos
    combinations = [(0.1, 0.01, 20), (0.1, 0.01, 10), (0.1, 0.005,20),
                    (0.01, 0.005, 20), (0.01, 0.001, 20),(0.01, 0.0005,20)]

    # cifar100 combos
    #Rank 1: ('ours', 'cifar100', 'resnet', 128, 0.1, 0.005, '20'); val_accuracy = 61.9; test_accuracy = 62.28; end_batch = 2048.0
    # Rank 2: ('ours', 'cifar100', 'resnet', 128, 0.1, 0.01, '20'); val_accuracy = 61.52; test_accuracy = 62.44; end_batch = 2048.0
    # Rank 3: ('ours', 'cifar100', 'resnet', 128, 0.1, 0.01, '10'); val_accuracy = 58.72; test_accuracy = 59.22; end_batch = 2048.0
    # Rank 1: ('ours', 'cifar100', 'resnet', 128, 0.01, 0.0005, '20'); val_accuracy = 55.15; test_accuracy = 55.34; end_batch = 347.0
    # Rank 2: ('ours', 'cifar100', 'resnet', 128, 0.01, 0.0005, '10'); val_accuracy = 54.68; test_accuracy = 55.57; end_batch = 306.0
    # Rank 3: ('ours', 'cifar100', 'resnet', 128, 0.01, 0.001, '20'); val_accuracy = 54.23; test_accuracy = 54.9; end_batch = 686.0

    combinations = [(0.1, 0.005, 20), (0.1, 0.01, 20), (0.1, 0.01, 10),
                    (0.01, 0.0005, 20), (0.01, 0.0005, 10), (0.01, 0.001, 20)]
    seeds = [1, 2, 3, 4, 5]
    for (lr, delta, freq), seed in product(combinations, seeds):
        filename = f'results/ours_cifar100_resnet_mb128_lr{lr}_delta{delta}_rf{freq}_s{seed}.csv'
        if not os.path.exists(filename):
            print(f'{lr} {delta} {freq} {seed} not found')

def run_runs(method = 'our', dataset = 'cifar10', n_trials = 5):
    result_dir = 'results'
    available_cpus = get_available_cpus()
    # num_cpus = available_cpus
    num_cpus = 4
    delta_gd = 1
    seeds = list(range(1, n_trials + 1))
    commands = []
    # datasets = ['cifar10', 'cifar100']
    # methods = ['our']
    methods = [method]
    datasets = [dataset]
    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                if method == 'our':
                    cmd = f'python run_{method}_{dataset}.py --delta {delta_gd} --manual_seed {seed}'
                else:
                    cmd = f'python run_{method}_{dataset}.py --manual_seed {seed}'
                commands.append(cmd)

    # pdb.set_trace()
    procs_by_cpu = [None] * num_cpus

    while len(commands) > 0:
        for idx in range(num_cpus):
            proc = procs_by_cpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this CPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(cmd, shell=True)
                procs_by_cpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for all processes to complete
    for proc in procs_by_cpu:
        if proc is not None:
            proc.wait()





def main():
    run_runs(method = 'cabs', dataset = 'cifar10')
    # run_our_cpu()
    # run_adabatch()
    # check_search()
    # check_seed()



if __name__ == '__main__':
    main()
