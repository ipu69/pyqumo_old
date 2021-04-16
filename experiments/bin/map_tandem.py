import multiprocessing
from itertools import product
from typing import Optional, Dict, Any, List

import pebble
from concurrent.futures import TimeoutError
import os
from time import perf_counter
import click

import pandas as pd
import numpy as np
from tabulate import tabulate

import pyqumo

from pyqumo import MarkovArrival, PhaseType, HyperExponential, Erlang, \
    MatrixError, array2string, rel_err
from pyqumo.cqumo.sim import simulate_tandem, Exponential

from pyqumo.algorithms.networks.mapph1n_tandem import reduce_map, \
    solve_iterative, SolveResults


NET_SIZE_COL = 'net_size'
CAPACITY_COL = 'capacity'
MAP_D0_COL = 'arrival_d0'
MAP_D1_COL = 'arrival_d1'
PH_S_COL = 'service_s'
PH_P_COL = 'service_p'


@click.group()
def cli():
    pass


# noinspection PyTypeChecker
@cli.command()
@click.option('--prefix', default='sim',
              help='prefix for CSV fields with results')
@click.option('--tol', default=.05,
              help='tolerance to matrix errors (default: .05')
@click.option('--num-proc', '-j', default=1, help='number of processes')
@click.option('--jupyter', is_flag=True, default=False)
@click.option('--max-packets', default=100000,
              help='number of packets to generate in simulation')
@click.argument('file_name')
def simulate(file_name, max_packets, jupyter, num_proc, tol, prefix):
    df = load_data(file_name)
    if jupyter:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    # Split into chunks. Each chunk is computed in parallel, then chunks
    # are joined.
    source_chunks = np.array_split(df, len(df) // num_proc)
    result_chunks = []
    for chunk in tqdm(source_chunks):
        sub_chunks = np.array_split(chunk, num_proc)
        sub_chunks = [c for c in sub_chunks if len(c) > 0]
        with multiprocessing.Pool(min(len(sub_chunks), num_proc)) as pool:
            results = pool.map(
                apply_simulate_,
                [(sub_chunk, max_packets, tol) for sub_chunk in sub_chunks]
            )
        for sc, res in zip(sub_chunks, results):
            sc['__ret'] = res
        result_chunks.append(pd.concat(sub_chunks, ignore_index=True))
    df = pd.concat(result_chunks, ignore_index=True)
    for field in ['delay', 'delay_accurate', 'delivery_prob',
                  'last_system_size', 'elapsed']:
        field_name = f'{prefix}_{field}' if prefix else field
        df[field_name] = df.apply(lambda row: row['__ret'][field], axis=1)
    save_data(df.drop(['__ret'], axis=1), file_name)


def apply_simulate_(args):
    """
    Apply simulate_tandem() function to dataframe.

    Parameters
    ----------
    args : tuple of DataFrame and max_packets

    Returns
    -------
    DataFrame with '__ret' column containing results.
    """
    df, max_packets, tol = args

    def fn(data):
        params = get_network_params_(data, tol=tol)
        t_start = perf_counter()
        # print(f'- going to simulate {max_packets} packets')
        sim_ret = simulate_tandem(
            params['arrival'],
            [params['service'].copy() for _ in range(params['net_size'])],
            params['capacity'], max_packets)
        return {
            'delay': sum([x.avg for x in sim_ret.response_time]),
            'delay_accurate': sim_ret.delivery_delays[0].avg,
            'elapsed': perf_counter() - t_start,
            'delivery_prob': sim_ret.delivery_prob[0],
            'last_system_size': sim_ret.system_size[-1].mean,
        }

    if isinstance(df, pd.DataFrame):
        return df.apply(fn, axis=1)

    return fn(df)


@cli.group()
def measure():
    pass


@measure.command()
@click.option('--num-proc', '-j', default=1, help='number of processes')
@click.option('--jupyter', is_flag=True, default=False)
@click.option('--num-packets', '-n',
              default='100,200,500,1000,10000,50000,100000',
              help='number of packets to generate in simulation')
@click.option('--num-samples', '-s', default=50, help='number of samples')
@click.option('--tol', default=.05,
              help='tolerance to matrix errors (default: .05')
@click.option('--output', '-o', 'output_file_name', help="Output file name")
@click.option('--precise', default='sim', help='Prefix of precise value')
@click.argument('file_name')
def simtime(file_name, output_file_name, precise, tol, num_samples,
            num_packets, jupyter, num_proc):
    df: pd.DataFrame = load_data(file_name)
    if jupyter:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    tqdm.pandas()
    # Split into chunks. Each chunk is computed in parallel, then chunks
    # are joined.
    df = df.sample(num_samples)
    source_chunks = np.array_split(df, len(df))
    result_chunks = []
    num_proc_ = min(num_proc, len(source_chunks))

    # Extract rows:
    precise_delay_col = f'{precise}_delay'
    precise_last_system_size_col = f'{precise}_last_system_size'
    precise_delivery_prob_col = f'{precise}_delivery_prob'

    records = df.sample(num_samples)[[
        MAP_D0_COL, MAP_D1_COL, PH_S_COL, PH_P_COL, NET_SIZE_COL,
        CAPACITY_COL, precise_delay_col, precise_delivery_prob_col,
        precise_last_system_size_col]
    ].rename(columns={
        precise_delay_col: 'delay',
        precise_delivery_prob_col: 'delivery_prob',
        precise_last_system_size_col: 'last_system_size',
    }).to_dict('records')

    num_packets_ = [int(val) for val in num_packets.split(',')]

    arguments = product(records, num_packets_)
    input_data = [(data, n_pkt, tol) for data, n_pkt in arguments]

    with multiprocessing.Pool(num_proc_) as pool:
        output_data = list(tqdm(
            pool.imap(apply_simulate_, input_data),
            total=len(input_data)))

    # Now results holds a list of dictionaries with fields 'elapsed',
    # 'delay', 'delivery_prob', 'last_system_size'. Find errors and build
    # result dataframe.
    result_data = {
        'delay_error': [],
        'delivery_prob_error': [],
        'last_system_size_error': [],
        'num_packets': [],
        'net_size': [],
        'elapsed': [],
    }
    for out_rec, inp_rec in zip(output_data, input_data):
        for metric in ('delay', 'last_system_size', 'delivery_prob'):
            result_data[f'{metric}_error'].append(
                rel_err(inp_rec[0][metric], out_rec[metric]))
        result_data['num_packets'].append(inp_rec[1])
        result_data['net_size'].append(inp_rec[0]['net_size'])
        result_data['elapsed'].append(out_rec['elapsed'])
    result_df = pd.DataFrame(result_data)

    save_data(result_df, output_file_name)




@cli.command()
@click.option('--verbose', default=0, help='verbosity level')
@click.option('--timeout', '-t', default=None, type=float,
              help='maximum time for a task')
@click.option('--tol', default=.05,
              help='tolerance to matrix errors (default: .05')
@click.option('--prefix', default='',
              help='prefix for CSV fields with results')
@click.option('--jupyter', is_flag=True, default=False)
@click.option('--max-precise-order', default=5000,
              help='when MAP reach this size, stop trying to solve precisely')
@click.option('--reduce-departure/--no-reduce-departure', default=True,
              help='reduce departure processes, or not (default: yes)')
@click.option('--reduce-arrival/--no-reduce-arrival', default=False,
              help='reduce arrival process, or not (default: no)')
@click.option('--num-moments', '-m', default=3,
              help='number of moments to match (1, 2 or 3)')
@click.option('--use-lag/--no-use-lag', default=False,
              help='use lag-1 correlation in reductions, or not (default: no)')
@click.option('--num-samples', type=int, default=0)
@click.argument('file_name')
def solve(file_name, num_samples, use_lag, num_moments, reduce_arrival,
          reduce_departure, max_precise_order, jupyter, prefix, tol, timeout,
          verbose):
    df = load_data(file_name)

    if verbose > 0:
        print('going to solve with:')
        print('file name        : ', file_name)
        print('reduce arrival   : ', yesno(reduce_arrival))
        print('reduce departure : ', yesno(reduce_departure))
        print('num moments      : ', num_moments)
        print('use lag          : ', yesno(use_lag))
        print('prefix           : ', prefix)
        print('max precise order: ', max_precise_order)
        print('tolerance        : ', tol)
        print('num samples      : ', ('all' if num_samples is None
                                      else num_samples))
        print('timeout          : ', '-' if timeout is None else timeout)

    if jupyter:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    # Split into chunks. However, here we launch only one process, since
    # NumPy/SciPy uses multiple processes themselves.
    if num_samples > 0:
        df = df.sample(num_samples)
    chunks = np.array_split(df, len(df))

    for chunk in tqdm(chunks):
        with pebble.ProcessPool(max_workers=1) as pool:
            map_ret = pool.map(
                apply_solve_,
                [{
                    'df': chunk,
                    'reduce_arrival': reduce_arrival,
                    'reduce_departure': reduce_departure,
                    'max_precise_order': max_precise_order,
                    'num_moments': num_moments,
                    'use_lag': use_lag,
                    'tol': tol,
                    'verbosity': verbose,
                }],
                timeout=timeout
            )
            future_results = map_ret.result()
            real_results = []
            while True:
                try:
                    ret = next(future_results)
                    real_results.append(ret)
                except TimeoutError:
                    ret = chunk.apply(
                        lambda _: SolveResults(skipped=True),
                        axis=1)
                    real_results.append(ret)
                except StopIteration:
                    break
            if verbose > 1:
                print("->> RESULTS: ", real_results[0].item(), '\n')
            chunk['__ret'] = real_results

    df = pd.concat(chunks, ignore_index=True)
    fields = ['skipped', 'delay', 'delivery_prob', 'last_system_size',
              'elapsed', 'max_inp_order', 'max_out_order',
              'm1_err', 'cv_err', 'skew_err', 'lag1_err']

    for field in fields:
        col_name = f'{prefix}_{field}' if prefix else field
        # noinspection PyTypeChecker
        df[col_name] = df.apply(
            lambda row: getattr(row['__ret'].item(), field, None),
            axis=1)

    if verbose > 1:
        print("========")
        num_samples_to_draw = min(num_samples, 10) if num_samples is not None \
            else 10
        col_names = [f'{prefix}_{field}' if prefix else field
                     for field in fields]
        print(df[col_names].sample(num_samples_to_draw))

    if num_samples == 0:
        save_data(df.drop(['__ret'], axis=1), file_name)
    else:
        print("I don't save data when using on limited number of samples.\n"
              "Call without --num-samples N to apply to the whole dataset and"
              "save results.")


def apply_solve_(kwargs):
    """
    Apply solve_iterative() function to a dataframe.

    Parameters
    ----------
    kwargs: dict
        expected keys are: 'df', 'reduce_arrival', 'reduce_departure',
        'num_moments', 'use_lag', 'max_precise_order', 'verbosity'

    Returns
    -------

    """
    verbosity = kwargs.get('verbosity')
    if kwargs['reduce_departure'] or kwargs['reduce_arrival']:
        def reducer(arrival):
            return reduce_map(arrival, num_moments=kwargs['num_moments'],
                              use_lag=kwargs['use_lag'])
    else:
        reducer = None

    def fn(data):
        params = get_network_params_(data, kwargs.get('tol', .05))
        if verbosity > 1:
            net_size = params['net_size']
            capacity = params['capacity']
            arrival = params['arrival']
            service = params['service']

            print(f'solving for net_size={net_size}, capacity={capacity};'
                  f'arrival: rate={arrival.rate}, cv={arrival.cv}, '
                  f'lag={arrival.lag(1)}, '
                  f'D0="{array2string(arrival.d0)}", '
                  f'D1="{array2string(arrival.d1)}"; '
                  f'service: rate={service.rate}, cv={service.cv}, '
                  f'S="{array2string(service.s)}", '
                  f'P="{array2string(service.p)}"'
                  )
        return solve_iterative(
            **params,
            reducer=reducer,
            reduce_arrival=kwargs['reduce_arrival'],
            reduce_departure=kwargs['reduce_departure'],
            max_precise_order=kwargs.get('max_precise_order', 8000)
        )

    return kwargs['df'].apply(fn, axis=1)


@cli.command()
@click.option('--verbose', is_flag=True, default=False)
@click.option('--jupyter', is_flag=True, default=False)
@click.option('--overwrite/--append', default=False)
@click.option('--max-order', default=10)
@click.option('--arrival-rate', default=1.0)
@click.option('--min-busy', default=0.1)
@click.option('--max-busy', default=2.0)
@click.option('--min-cv', default=0.5)
@click.option('--max-cv', default=2.0)
@click.option('--max-skew', default=5.0)
@click.option('--min-lag', default=-0.01)
@click.option('--max-lag', default=0.18)
@click.option('--max-capacity', default=10)
@click.option('--min-net-size', default=1)
@click.option('--max-net-size', default=10)
@click.option('--basic/--complex', default=False)
@click.option('--num-samples', '-n', default=500)
@click.argument('file_name')
def generate(file_name, num_samples, basic, max_net_size, min_net_size,
             max_capacity, max_lag, min_lag, max_skew, max_cv, min_cv,
             max_busy, min_busy, arrival_rate, max_order, overwrite, jupyter,
             verbose):
    if verbose:
        args_table_ = [
            ('file name', file_name),
            ('number of samples', f'{num_samples}'),
            ('network size', f'{min_net_size} ... {max_net_size}'),
            ('capacity', f'0 ... {max_capacity}'),
            ('lag-1', f'{min_lag} ... {max_lag}'),
            ('variation coefficient (cv)', f'{min_cv} ... {max_cv}'),
            ('skewness', f'cv-1/cv ... {max_skew}'),
            ('busy', f'{min_busy} ... {max_busy}'),
            ('arrival rate', f'{arrival_rate}'),
            ('maximum order of PH or MAP', max_order),
            ('use basic distributions', yesno(basic)),
            ('jupyter', yesno(jupyter)),
            ('verbose', yesno(verbose)),
        ]
        print(tabulate(args_table_, headers=('Parameter', 'Value')))

    if os.path.exists(file_name) and not overwrite:
        df_base = load_data(file_name)
    else:
        df_base = None
    if jupyter:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    samples = []
    mean_arrival = 1 / arrival_rate
    for _ in tqdm(range(num_samples)):
        desired_busy = np.random.uniform(min_busy * 1.001, max_busy * 0.999)
        mean_service = desired_busy / arrival_rate
        real_busy = 0.0
        arrival = None
        service = None
        while (arrival is None or service is None or
               arrival.skewness > max_skew or
               service.skewness > max_skew or
               arrival.cv < min_cv or arrival.cv > max_cv or
               service.cv < min_cv or service.cv > max_cv or
               real_busy < min_busy or real_busy > max_busy):

            try:
                if basic:
                    arrival_ph = _basic_dist(
                        mean_arrival, min_cv, max_cv).as_ph()
                    service = _basic_dist(mean_service, min_cv, max_cv).as_ph()
                else:
                    arrival_ph = pyqumo.algorithms.random_phase_type(
                        avg=mean_arrival, min_cv=min_cv, max_cv=max_cv,
                        max_skew=max_skew, max_order=max_order)
                    service = pyqumo.algorithms.random_phase_type(
                        avg=mean_service, min_cv=min_cv, max_cv=max_cv,
                        max_skew=max_skew, max_order=max_order)
                arrival = pyqumo.algorithms.random_markov_arrival(
                    arrival_ph, min_lag1=min_lag, max_lag1=max_lag)
            except MatrixError:
                arrival = None
                service = None

            # For some reason, sometimes skewness becomes complex :-(
            if arrival is None or service is None or \
                    isinstance(arrival.skewness, complex) or \
                    isinstance(service.skewness, complex) or \
                    isinstance(arrival.std, complex) or \
                    isinstance(service.std, complex):
                arrival = None
                service = None
            else:
                # Compute busy ratio:
                real_busy = service.mean / arrival.mean

        # Generate network size and capacity:
        capacity = np.random.randint(0, max_capacity + 1)
        net_size = np.random.randint(min_net_size, max_net_size + 1)
        samples.append({
            'capacity': capacity,
            'net_size': net_size,
            # Store arrival and service parameters:
            MAP_D0_COL: arrival.d0,
            MAP_D1_COL: arrival.d1,
            PH_S_COL: service.s,
            PH_P_COL: service.p,
            # Store arrival properties:
            'arrival_mean': arrival.mean,
            'arrival_rate': 1 / arrival.mean,
            'arrival_cv': arrival.cv,
            'arrival_skew': arrival.skewness,
            'arrival_lag1': arrival.lag(1),
            'arrival_order': arrival.order,
            # Store service properties:
            'service_mean': service.mean,
            'service_rate': 1 / service.mean,
            'service_cv': service.cv,
            'service_skew': service.skewness,
            'service_order': service.order,
            # System-wide properties:
            'busy': service.mean / arrival.mean,
        })
    # Build and store dataframe:
    df = pd.DataFrame(samples)
    if df_base is not None:
        df = pd.concat([df_base, df], ignore_index=True)
    save_data(df, file_name)


def _basic_dist(mean: float, min_cv: float, max_cv: float):
    cv = np.random.uniform(min_cv * 1.001, max_cv * 0.999)
    std = cv * mean
    if cv < 0.95:
        return Erlang.fit(mean, std)
    if cv > 1.05:
        return HyperExponential.fit(mean, std)
    return Exponential(mean)


@cli.command()
@click.argument('file_name')
def inspect(file_name):
    df = load_data(file_name, create_dists=True)
    print('Columns')
    print('-------')
    print(df.info())
    print('Data')
    print('----')
    print(df.describe().transpose())


@cli.command()
@click.option('--columns', '-c', default='',
              help='column names, separated with comma')
@click.argument('file_name')
def display(file_name, columns):
    df = load_data(file_name)
    col_names = df.columns if not columns else columns.split(',')
    col_names = [col.strip() for col in col_names]
    print(df[col_names])


#############################################################################
# HELPERS
#############################################################################
def load_data(file_name, create_dists: bool = False, tol=.05) -> pd.DataFrame:
    df = pd.read_csv(file_name, converters={
        MAP_D0_COL: parse_array,
        MAP_D1_COL: parse_array,
        PH_S_COL: parse_array,
        PH_P_COL: (lambda s: parse_array(s, n_dims=1))
    })
    if create_dists:
        df['arrival'] = df.apply(
            lambda r: MarkovArrival(r[MAP_D0_COL], r[MAP_D1_COL], tol=tol),
            axis=1)
        df['service'] = df.apply(
            lambda r: PhaseType(r[PH_S_COL], r[PH_P_COL], tol=tol),
            axis=1)
    return df


def save_data(df: pd.DataFrame, file_name) -> None:
    df.to_csv(file_name, index=False, )


def parse_array(s: str, n_dims: int = 2) -> np.ndarray:
    """
    Разбирает двумерный массив из того формата, в котором Pandas
    сохраняет ячейку с ndarray, то есть '[[1 2]\n [3 4]]'
    """
    s = s.replace('\n', '').replace('[', '').replace(']', '')
    a = np.fromstring(s, sep=' ')  # вектор, содержащий N^2 элементов
    n = int(len(a)**(1/n_dims))
    return a.reshape((n,)*n_dims)


def get_network_params_(data, tol=.05):
    net_size = data[NET_SIZE_COL]
    capacity = data[CAPACITY_COL]
    d0 = data[MAP_D0_COL]
    d1 = data[MAP_D1_COL]
    s = data[PH_S_COL]
    p = data[PH_P_COL]
    return {
        'arrival': MarkovArrival(d0, d1, tol=tol),
        'service': PhaseType(s, p, tol=tol),
        'net_size': net_size,
        'capacity': capacity,
    }


def yesno(x):
    return 'yes' if x else 'no'


if __name__ == '__main__':
    cli()
