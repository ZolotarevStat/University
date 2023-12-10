from typing import Callable, Any
from functools import partial
import json
import math
import torch


def run_precision(func: Callable) -> None:
    cases = [
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[0.5, 0.4, 0.3, 0.2]]),
            "target": torch.Tensor([[1, 0, 1, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 2 / 3,
                10: 2 / 4,
                100: 2 / 4,
            },
        },
        {
            "output": torch.Tensor(
                [
                    [9, 5, 3, 0, 7, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 10],
                    [0, 0, 1, 5, 9, 3, 0, 0, 0, 0, 0, 4, 0, 0, 10, 7, 0, 2, 8, 6],
                    [0, 1, 4, 8, 6, 5, 3, 7, 10, 0, 9, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [7, 8, 0, 0, 1, 0, 4, 0, 10, 0, 0, 6, 0, 0, 0, 9, 2, 3, 5, 0],
                ]
            ),
            "target": torch.Tensor(
                [
                    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                ]
            ),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1 / 4,
                3: (2 / 3 + 1 / 3 + 1 / 3 + 1 / 3) / 4,
                10: (5 / 10 + 3 / 10 + 5 / 10 + 3 / 10) / 4,
                100: (8 / 20 + 8 / 20 + 10 / 20 + 9 / 20) / 4,
            },
        },
    ]
    return _run_tests(func, cases)


def run_recall(func: Callable) -> None:
    cases = [
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            }
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1 / 4,
                3: 3 / 4,
                10: 4 / 4,
                100: 4 / 4,
            }
        },
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1 / 4,
                3: 3 / 4,
                10: 4 / 4,
                100: 4 / 4,
            }
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            }
        },
        {
            "output": torch.Tensor([[0.5, 0.4, 0.3, 0.2]]),
            "target": torch.Tensor([[1, 0, 1, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1 / 2,
                3: 2 / 2,
                10: 2 / 2,
                100: 2 / 2,
            }
        },
        {
            "output": torch.Tensor(
                [
                    [9, 5, 3, 0, 7, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 10],
                    [0, 0, 1, 5, 9, 3, 0, 0, 0, 0, 0, 4, 0, 0, 10, 7, 0, 2, 8, 6],
                    [0, 1, 4, 8, 6, 5, 3, 7, 10, 0, 9, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [7, 8, 0, 0, 1, 0, 4, 0, 10, 0, 0, 6, 0, 0, 0, 9, 2, 3, 5, 0],
                ]
            ),
            "target": torch.Tensor(
                [
                    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                ]
            ),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: (1 / 8) / 4,
                3: (2 / 8 + 1 / 8 + 1 / 10 + 1 / 9) / 4,
                10: (5 / 8 + 3 / 8 + 5 / 10 + 3 / 9) / 4,
                100: 1.0,
            }
        },
    ]
    return _run_tests(func, cases)


def run_map(func: Callable) -> None:
    cases = [
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[0.5, 0.4, 0.3, 0.2]]),
            "target": torch.Tensor([[1, 0, 1, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: (1 + 2 / 3) / 2,
                10: (1 + 2 / 3) / 2,
                100: (1 + 2 / 3) / 2,
            },
        },
        {
            "output": torch.Tensor(
                [
                    [9, 5, 3, 0, 7, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 10],
                    [0, 0, 1, 5, 9, 3, 0, 0, 0, 0, 0, 4, 0, 0, 10, 7, 0, 2, 8, 6],
                    [0, 1, 4, 8, 6, 5, 3, 7, 10, 0, 9, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [7, 8, 0, 0, 1, 0, 4, 0, 10, 0, 0, 6, 0, 0, 0, 9, 2, 3, 5, 0],
                ]
            ),
            "target": torch.Tensor(
                [
                    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                ]
            ),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: (0.0 + 1.0 + 0.0 + 0.0) / 4,
                3: (
                    (1 / 2 + 2 / 3) / 2
                    + 1
                    + 1 / 2
                    + 1 / 3
                ) / 4,
                10: (
                    (1 / 2 + 2 / 3 + 3 / 6 + 4 / 7 + 5 / 8) / 5
                    + (1 + 2 / 9 + 3 / 10) / 3
                    + (1 / 2 + 2 / 4 + 3 / 6 + 4 / 8 + 5 / 9) / 5
                    + (1 / 3 + 2 / 5 + 3 / 8) / 3 
                ) / 4,
                100: (
                    (1 / 2 + 2 / 3 + 3 / 6 + 4 / 7 + 5 / 8 + 6 / 12 + 7 / 14 + 8 / 20) / 8
                    + (1 + 2 / 9 + 3 / 10 + 4 / 12 + 5 / 14 + 6 / 15 + 7 / 16 + 8 / 17) / 8
                    + (1 / 2 + 2 / 4 + 3 / 6 + 4 / 8 + 5 / 9 + 6 / 11 + 7 / 13 + 8 / 16 + 9 / 18 + 10 / 19) / 10
                    + (1 / 3 + 2 / 5 + 3 / 8 + 4 / 12 + 5 / 16 + 6 / 17 + 7 / 18 + 8 / 19 + 9 / 20) / 9
                ) / 4
            },
        },
    ]
    return _run_tests(partial(func, normalized=False), cases)


def run_mnap(func: Callable) -> None:
    cases = [
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[0.5, 0.4, 0.3, 0.2]]),
            "target": torch.Tensor([[1, 0, 1, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: (1 + 2 / 3) / 2,
                10: (1 + 2 / 3) / 2,
                100: (1 + 2 / 3) / 2,
            },
        },
        {
            "output": torch.Tensor(
                [
                    [9, 5, 3, 0, 7, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 10],
                    [0, 0, 1, 5, 9, 3, 0, 0, 0, 0, 0, 4, 0, 0, 10, 7, 0, 2, 8, 6],
                    [0, 1, 4, 8, 6, 5, 3, 7, 10, 0, 9, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [7, 8, 0, 0, 1, 0, 4, 0, 10, 0, 0, 6, 0, 0, 0, 9, 2, 3, 5, 0],
                ]
            ),
            "target": torch.Tensor(
                [
                    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                ]
            ),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: (0.0 + 1.0 + 0.0 + 0.0) / 4,
                3: (
                    (1 / 2 + 2 / 3) / 3
                    + 1 / 3 
                    + (1 / 2) / 3
                    + (1 / 3) / 3
                ) / 4,
                10: (
                    (1 / 2 + 2 / 3 + 3 / 6 + 4 / 7 + 5 / 8) / 8
                    + (1 + 2 / 9 + 3 / 10) / 8
                    + (1 / 2 + 2 / 4 + 3 / 6 + 4 / 8 + 5 / 9) / 10
                    + (1 / 3 + 2 / 5 + 3 / 8) / 9
                ) / 4,
                100: (
                    (1 / 2 + 2 / 3 + 3 / 6 + 4 / 7 + 5 / 8 + 6 / 12 + 7 / 14 + 8 / 20) / 8
                    + (1 + 2 / 9 + 3 / 10 + 4 / 12 + 5 / 14 + 6 / 15 + 7 / 16 + 8 / 17) / 8
                    + (1 / 2 + 2 / 4 + 3 / 6 + 4 / 8 + 5 / 9 + 6 / 11 + 7 / 13 + 8 / 16 + 9 / 18 + 10 / 19) / 10
                    + (1 / 3 + 2 / 5 + 3 / 8 + 4 / 12 + 5 / 16 + 6 / 17 + 7 / 18 + 8 / 19 + 9 / 20) / 9
                ) / 4
            },
        },
    ]
    return _run_tests(partial(func, normalized=True), cases)


def run_ndcg(func: Callable) -> None:
    cases = [
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[0, 0, 0, 0]]),
            "target": torch.Tensor([[1, 1, 1, 1]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: 1.0,
                10: 1.0,
                100: 1.0,
            },
        },
        {
            "output": torch.Tensor([[1, 1, 1, 1]]),
            "target": torch.Tensor([[0, 0, 0, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 0.0,
                3: 0.0,
                10: 0.0,
                100: 0.0,
            },
        },
        {
            "output": torch.Tensor([[0.5, 0.4, 0.3, 0.2]]),
            "target": torch.Tensor([[1, 0, 1, 0]]),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: 1.0,
                3: (1 / math.log2(1 + 1) + 1 / math.log2(3 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)),
                10: (1 / math.log2(1 + 1) + 1 / math.log2(3 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)),
                100: (1 / math.log2(1 + 1) + 1 / math.log2(3 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)),
            },
        },
        {
            "output": torch.Tensor(
                [
                    [9, 5, 3, 0, 7, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 10],
                    [0, 0, 1, 5, 9, 3, 0, 0, 0, 0, 0, 4, 0, 0, 10, 7, 0, 2, 8, 6],
                    [0, 1, 4, 8, 6, 5, 3, 7, 10, 0, 9, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                    [7, 8, 0, 0, 1, 0, 4, 0, 10, 0, 0, 6, 0, 0, 0, 9, 2, 3, 5, 0],
                ]
            ),
            "target": torch.Tensor(
                [
                    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                ]
            ),
            "topk": [1, 3, 10, 100],
            "expected": {
                1: (0.0 + 1.0 + 0.0 + 0.0) / 4,
                3: round((
                    (1 / math.log2(2 + 1) + 1 / math.log2(3 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1))
                    + (1 / math.log2(1 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1))
                    + (1 / math.log2(2 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1))
                    + (1 / math.log2(3 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1))
                ) / 4, 4),
                10: round((
                    (1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1))
                    + (1 / math.log2(1 + 1) + 1 / math.log2(9 + 1) + 1 / math.log2(10 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1))
                    + (1 / math.log2(2 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(9 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(9 + 1) + 1 / math.log2(10 + 1))
                    + (1 / math.log2(3 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(8 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(9 + 1))
                ) / 4, 4),
                100: round((
                    (1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(12 + 1) + 1 / math.log2(14 + 1) + 1 / math.log2(20 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1))
                    + (1 / math.log2(1 + 1) + 1 / math.log2(9 + 1) + 1 / math.log2(10 + 1) + 1 / math.log2(12 + 1) + 1 / math.log2(14 + 1) + 1 / math.log2(15 + 1) + 1 / math.log2(16 + 1) + 1 / math.log2(17 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1))
                    + (1 / math.log2(2 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(9 + 1) + 1 / math.log2(11 + 1) + 1 / math.log2(13 + 1) + 1 / math.log2(16 + 1) + 1 / math.log2(18 + 1) + 1 / math.log2(19 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(9 + 1) + 1 / math.log2(10 + 1))
                    + (1 / math.log2(3 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(12 + 1) + 1 / math.log2(16 + 1) + 1 / math.log2(17 + 1) + 1 / math.log2(18 + 1) + 1 / math.log2(19 + 1) + 1 / math.log2(20 + 1)) / (1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1) + 1 / math.log2(4 + 1) + 1 / math.log2(5 + 1) + 1 / math.log2(6 + 1) + 1 / math.log2(7 + 1) + 1 / math.log2(8 + 1) + 1 / math.log2(9 + 1))
                ) / 4, 4)
            },
        },
    ]
    return _run_tests(func, cases)


def _run_tests(func: Callable, cases: list[dict[str, Any]]) -> None:
    for case in cases:
        expected = case.pop("expected")
        for k in case.pop("topk"):
            actual = func(**case, topk=k)
            torch.testing.assert_close(
                actual,
                torch.tensor(expected[k]) if isinstance(actual, torch.Tensor) else expected[k],
                msg=lambda msg: f"Inputs:{json.dumps({key: (v.tolist() if isinstance(v, torch.Tensor) else v) for key, v in (case | {'topk': k}).items()}, indent=2, ensure_ascii=False)}\n{msg}",
            )