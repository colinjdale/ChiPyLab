import matplotlib.pyplot as plt
import numpy as np
from general.pulses import blackman_pulse


def linear_seq(frac, y_f, y_i):
    return y_i + frac * (y_f - y_i)


def exponential_seq(frac, y_f, y_i):
    return y_i + (y_f - y_i) * (1 - np.exp(-5 * frac)) / (1 - np.exp(-5))


def constant_seq(frac, y_f, y_i):
    return y_i


def step_seq(frac, y_f, y_i):
    return y_f


def sine_seq_func(amp, freq):
    def sine_seq(frac, y_f, y_i):
        return y_i + amp * (np.sin(2*np.pi * freq * frac))
    return sine_seq


def blackman_seq_func(amp):
    def blackman_seq(frac, y_f, y_i):
        return y_i + amp * blackman_pulse(frac, 1)
    return blackman_seq


def segment_y(x_seg, func_list, y_f_list, y_i=0):
    y_i_list = [y_i] + y_f_list[:-1]
    y_list = []

    seg = 0
    while seg < len(func_list):
        y = [func_list[seg](xi, y_f_list[seg], y_i_list[seg]) for xi in x_seg]
        y_list = y_list + y
        seg += 1

    y_list = np.array(y_list)
    return y_list


def segment_x(x_seg, scale_list):
    x_list = []
    seg = 0
    x_edges = [0]
    while seg < len(scale_list):
        x = x_edges[seg] + x_seg * scale_list[seg]
        x_edges = x_edges + [x[-1]]
        x_list = x_list + list(x)
        seg += 1

    x_edges = np.array(x_edges)
    x_list = np.array(x_list)
    return (x_list, x_edges)


seq_func_mapping = {'linear': linear_seq,
                    'exp': exponential_seq,
                    'constant': constant_seq,
                    'step': step_seq,
                    }


def calculate_sequence(df, seq_func_mapping=seq_func_mapping, num=1000):
    # Forward fill value columns
    v_cols = [c for c in df.columns if "_val" in c]
    df[v_cols] = df[v_cols].ffill()

    # Fill function columns with constant
    f_cols = [c for c in df.columns if "_func" in c]
    df[f_cols] = df[f_cols].fillna("constant")

    # Get scaling columns, fill blank with 1.0
    df['scale'] = df['scale'].fillna(1.0)

    # Replace all function name strings with functions
    df = df.replace(seq_func_mapping)

    x = np.linspace(1/num, 1, num)

    # Calculate values
    y = [segment_y(x, list(df[pf]), list(df[pv])) \
          for pf, pv in zip(f_cols, v_cols)]
    
    x, x_edges = segment_x(x, list(df['scale']))

    return x, y, x_edges, df
