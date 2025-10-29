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


def segment_plot(x, func_list, y_f_list, y_i=0):
    y = np.zeros_like(x)
    y_i_list = [y_i] + y_f_list[:-1]
    for i, xi in enumerate(x):
        seg = int(np.floor(xi))
        frac = xi - seg
        y[i] = func_list[seg](frac, y_f_list[seg], y_i_list[seg])
    return y


seq_func_mapping = {'linear': linear_seq,
                    'exp': exponential_seq,
                    'constant': constant_seq,
                    'step': step_seq,
                    }


def calculate_sequence(df, x, seq_func_mapping=seq_func_mapping):
    # Forward fill value columns
    v_cols = [c for c in df.columns if "_val" in c]
    df[v_cols] = df[v_cols].ffill()

    # Fill function columns with constant
    f_cols = [c for c in df.columns if "_func" in c]
    df[f_cols] = df[f_cols].fillna("constant")

    # Replace all function name strings with functions
    df = df.replace(seq_func_mapping)

    # Calculate values
    ys = [segment_plot(x, list(df[pf]), list(df[pv])) for pf, pv in zip(f_cols, v_cols)]

    return ys, df
