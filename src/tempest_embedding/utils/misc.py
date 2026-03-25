from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np


DEFAULT_DATASETS = ['CollegeMsg', 'enron', 'TaobaoSmall', 'mooc', 'wikipedia', 'reddit']


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Tempest-powered Neural Temporal Walks')
    parser.add_argument('-d', '--data', type=str, choices=DEFAULT_DATASETS, default='CollegeMsg')
    parser.add_argument('--data_usage', default=1.0, type=float)
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu_cores', type=int, default=1)

    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--tolerance', type=float, default=0)

    parser.add_argument('--n_degree', nargs='*', default=['64', '1'])
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--pos_enc', type=str, default='saw', choices=['saw', 'lp'])
    parser.add_argument('--pos_dim', type=int, default=172)
    parser.add_argument('--walk_mutual', action='store_true')
    parser.add_argument('--walk_linear_out', action='store_true', default=False)
    parser.add_argument('--temporal_bias', default=1e-5, type=float)
    parser.add_argument('--spatial_bias', default=1.0, type=float)
    parser.add_argument('--ee_bias', default=1.0, type=float)
    parser.add_argument('--solver', type=str, default='rk4', choices=['euler', 'rk4', 'dopri5'])
    parser.add_argument('--step_size', type=float, default=0.125)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--negs', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=1)

    # Tempest-facing knobs
    parser.add_argument('--use_gpu_walks', action='store_true', default=False)
    parser.add_argument('--walk_direction', type=str, default='Backward_In_Time', choices=['Backward_In_Time', 'Forward_In_Time'])
    parser.add_argument('--walk_bias', type=str, default='SpatioTemporal')
    parser.add_argument('--initial_edge_bias', type=str, default=None)
    parser.add_argument('--max_walk_len', type=int, default=3)
    parser.add_argument('--walk_padding_value', type=int, default=0)
    parser.add_argument('--max_time_capacity', type=int, default=-1)
    parser.add_argument('--timescale_bound', type=float, default=-1.0)
    return parser


def get_args():
    parser = build_arg_parser()
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv
