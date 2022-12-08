from itertools import chain

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from setup import USER_COL, ITEM_COL, RATING_COL

import seaborn as sns
import matplotlib.pyplot as plt


def sparse_ind(df, user_col=USER_COL, item_col=ITEM_COL):
    i = 1 - len(df) / (df[user_col].nunique() * df[item_col].nunique())
    return round(i, 3)


def get_worst_ds(df, n_users_frac, n_items_frac , rand_range=(1, 5), user_col=USER_COL, item_col=ITEM_COL, rating_col=RATING_COL):
    n_users = int(df[user_col].nunique() * n_users_frac)

    new_id = df[user_col].max() + 1
    max_item = df[item_col].max() + 1

    dfs = []
    for i in range(n_users):
        new_id += 1
        n = np.random.randint(*(rand_range))

        dfs.append(
            pd.DataFrame({
            user_col: new_id,
            item_col: np.random.randint(max_item, max_item +20010, n),
            rating_col: np.random.randint(1,6, n)
        }))
    dfs = pd.concat(dfs)
    return pd.concat([dfs, df])

def get_worst_ds2(df, n_users_frac, n_items_frac=.1 , rand_range=(1, 5), user_col=USER_COL, item_col=ITEM_COL, rating_col=RATING_COL):
    n_users = int(df[user_col].nunique() * n_users_frac)
    n_items = int(df[item_col].nunique() * n_items_frac)

    shuffle_users = pd.Series(df[user_col].unique()).sample(frac=1)
    worst_users = shuffle_users[:n_users]

    norm_df = df[~df[user_col].isin(worst_users)]
    spars_df = df[df[user_col].isin(worst_users)]

    spars_df = spars_df.groupby(user_col, as_index=False).apply(lambda x: x.sample(2)).reset_index(drop=1)

    return pd.concat([norm_df, spars_df])

# df_sparse = get_worst_ds(df, 3.99, n_items_frac=3.99)
#
# sparse_ind(df_sparse)

def plot_sparse(ser, bins=22, sort=True, o_range=None, MIN_ITEMS=20, y=None):
    name = ser.name
    nuniq = ser.nunique()

    if sort:
        order = ser.value_counts()
        ser = ser.sort_values(key=lambda x: order.loc[x], ascending=False)

    maper = dict(zip(ser.unique(), range(nuniq)))
    ser = pd.Series([maper[x] for x in ser])

    plt.title(f'{name} histogram')
    if o_range is not None:
        ser = ser[(ser > o_range[0]) & (ser < o_range[1])]
    plt.hist(ser, range=o_range, bins=bins)

    percent_less_min = 1 - (ser.value_counts() > MIN_ITEMS).sum() / nuniq
    plt.axvspan((ser.value_counts() > MIN_ITEMS).sum(), nuniq, alpha=0.3, color='red',
                label=f"{round(percent_less_min, 2)} of {name} has less than {MIN_ITEMS} {y}"
                )
    plt.ylabel(y)
    plt.xlabel(name)
    plt.legend()

def reduce_sparsity(df, min_shows_per_user, min_user_per_show, user_col=USER_COL, item_col=ITEM_COL, return_cold=False, add_default_user=False):
    good_users = df[user_col].value_counts()[df[user_col].value_counts() > min_shows_per_user].index

    cold_users = list(df[user_col][~df[user_col].isin(good_users)].unique())
    if add_default_user:
        cold_users = cold_users + ["default"]

    df = df[df[user_col].isin(good_users)]

    good_shows = df[item_col].value_counts()[df[item_col].value_counts() > min_user_per_show].index
    df = df[df[item_col].isin(good_shows)].reset_index(drop=1)

    if return_cold:
        return df, pd.Series(cold_users)
    return df

# df_ = reduce_sparsity(df_sparse, min_shows_per_user=20, min_user_per_show=20)

def plot_metrics(metrics):
    m = pd.DataFrame(metrics)
    pal = sns.color_palette('Set2', len(m))
    sns.set_style("darkgrid")

    figure, axes = plt.subplots(2, 2, sharex=True, figsize=(10,5))
    figure.suptitle('Metrics comparison')

    axes = chain(*axes.tolist())
    for m_name in m.columns:
        g = sns.barplot(x=m[m_name].index, y=m[m_name], palette=pal, ax=next(axes))
        g.set(title=m_name)
        g.set(xlabel=' ', ylabel=' ')
        for row, t in enumerate(m[m_name]):
            if m_name == "coverage":
                v = f"{round(t, 3) * 100}%"
            elif m_name == "diversity":
                v = int(t)
            else:
                v = round(t, 3)
            g.text(row, t, v, color='black', ha='center')
