import sys
from argparse import ArgumentParser
from typing import List
import time

import numpy as np
import onnxruntime as ort
from memory_profiler import profile


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_argument("--signal_path", type=str)
    parser.add_argument("--model_path", type=str)
    return parser.parse_args(args)

@profile
def main(args: List[str]):
    argv = parse_args(args)
    model = ort.InferenceSession(argv.model_path)
    xs: np.ndarray = np.load(argv.signal_path)
    xs = xs[-1]
    xs = xs[:xs.size // 2].reshape(1, -1, 1)
    print(xs.shape)
    time_start = time.time()
    ys: np.ndarray = model.run(
        ["time_distributed_1"],
        {"lstm_1_input": xs.astype(np.float32)},
    )[0]
    time_end = time.time()
    print(f"time used {time_end-time_start} secs")


if __name__ == "__main__":
    main(sys.argv[1:])