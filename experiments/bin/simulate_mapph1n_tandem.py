import sys
import multiprocessing
from time import perf_counter

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from pyqumo import MarkovArrival, PhaseType
from pyqumo.cqumo.sim import simulate_tandem


def parse_numpy_array(s: str, n_dims: int = 2) -> np.ndarray:
    """
    Разбирает двумерный массив из того формата, в котором Pandas
    сохраняет ячейку с ndarray, то есть '[[1 2]\n [3 4]]'
    """
    s = s.replace('\n', '').replace('[', '').replace(']', '')
    a = np.fromstring(s, sep=' ')  # вектор, содержащий N^2 элементов
    n = int(len(a)**(1/n_dims))
    return a.reshape((n,)*n_dims)


def apply_solve_interactive_(args):
    df_, max_packets_ = args

    def fn(row):
        ph_s = row['service_s']
        ph_p = row['service_p']
        net_size = row['net_size']
        capacity = row['capacity']
        arrival = MarkovArrival(row['arrival_d0'], row['arrival_d1'], tol=.01)
        services = [PhaseType(ph_s, ph_p, tol=.01) for _ in range(net_size)]

        t_start = perf_counter()
        sim_ret = simulate_tandem(arrival, services, capacity, max_packets_)
        return {
            'delay': sim_ret.delivery_delays[0].avg,
            'elapsed': perf_counter() - t_start,
            'delivery_prob': sim_ret.delivery_prob[0],
            'last_system_size': sim_ret.system_size[-1].mean,
        }
    return df_.apply(fn, axis=1)


if __name__ == '__main__':
    # hack!: for some reason, this works when calling from ipython..
    __spec__ = None

    if len(sys.argv) < 3:
        print('Format: script.py FILE_NAME MAX_PACKETS [N_PROC=1]')
        sys.exit(1)

    file_name = sys.argv[1]
    max_packets = int(sys.argv[2])
    num_proc = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    df = pd.read_csv(file_name, converters={
        'arrival_d0': parse_numpy_array,
        'arrival_d1': parse_numpy_array,
        'service_s': parse_numpy_array,
        'service_p': (lambda s: parse_numpy_array(s, n_dims=1))
    })

    chunks = np.array_split(df, len(df) // 10)

    for chunk in tqdm(chunks):
        with multiprocessing.Pool(num_proc) as pool:
            chunk['__ret'] = \
                pool.map(apply_solve_interactive_, [(chunk, max_packets)])[0]

    df = pd.concat(chunks, ignore_index=True)

    fields = ['delay', 'delivery_prob', 'last_system_size', 'elapsed']

    for field in fields:
        # noinspection PyTypeChecker
        df[f'sim_{field}'] = df.apply(lambda row: row['__ret'][field], axis=1)

    df.to_csv(file_name)
    sys.exit(0)
