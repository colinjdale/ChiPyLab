import pandas as pd

def group_by_mean(df, scan_name):
    mean = df.groupby([scan_name]).mean().reset_index()
    sem = df.groupby([scan_name]).sem().reset_index().add_prefix("em_")
    std = df.groupby([scan_name]).std().reset_index().add_prefix("e_")
    df_avg = pd.concat([mean, std, sem], axis=1)
    return df_avg