import sys
import multiprocessing

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from pyqumo import MarkovArrival, PhaseType
from pyqumo.algorithms.networks.mapph1n_tandem import solve_iterative, \
    reduce_map


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
    df_, reduce_, only_mean_, use_lag_, reduce_arrival_ = args

    def fn(row):
        d0 = row['arrival_d0']
        d1 = row['arrival_d1']
        s = row['service_s']
        p = row['service_p']
        inp = {
            'arrival': MarkovArrival(d0, d1, tol=.01),
            'service': PhaseType(s, p, tol=.01),
            'capacity': row['capacity'],
            'net_size': row['net_size'],
        }

        if reduce_:
            def reducer(map_):
                return reduce_map(map_, use_lag=use_lag_, only_mean=only_mean_)
        else:
            reducer = None

        return solve_iterative(inp, reducer=reducer,
                               reduce_arrival=reduce_arrival_)
    return df_.apply(fn, axis=1)


if __name__ == '__main__':
    # hack!: for some reason, this works when calling from ipython..
    __spec__ = None

    if len(sys.argv) < 6:
        print('Format: script.py FILE_NAME PREFIX REDUCE ONLY_MEAN USE_LAG '
              'REDUCE_ARRIVAL')
        sys.exit(1)

    file_name = sys.argv[1]
    prefix = sys.argv[2]
    reduce = bool(int(sys.argv[3]))
    only_mean = bool(int(sys.argv[4]))
    use_lag = bool(int(sys.argv[5]))
    reduce_arrival = bool(int(sys.argv[6]))

    df = pd.read_csv(file_name, converters={
        'arrival_d0': parse_numpy_array,
        'arrival_d1': parse_numpy_array,
        'service_s': parse_numpy_array,
        'service_p': (lambda s: parse_numpy_array(s, n_dims=1))
    })
    chunks = np.array_split(df, len(df) // 4)

    for chunk in tqdm(chunks):
        with multiprocessing.Pool(1) as pool:
            chunk['__ret'] = pool.map(
                apply_solve_interactive_,
                [(chunk, reduce, only_mean, use_lag, reduce_arrival)])[0]

    df = pd.concat(chunks, ignore_index=True)

    fields = ['skipped', 'delay', 'delivery_prob', 'last_system_size',
              'elapsed', 'max_inp_order', 'max_out_order',
              'm1_err', 'cv_err', 'skew_err', 'lag1_err']

    for field in fields:
        # noinspection PyTypeChecker
        df[f'{prefix}_{field}'] = df.apply(
            lambda row: getattr(row['__ret'], field),
            axis=1)

    df.to_csv(file_name)
    sys.exit(0)
