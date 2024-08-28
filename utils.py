import pandas as pd
import os
from latency_config import (
    ATColumns,
    AT_TABLE_PKEYS,
    TRADES_PKEYS,
    QUOTES_PKEYS,
    SPREAD_DEFINITIONS,
    EXCEL_SPREAD_DATE_FROMAT,
    QuotesColumns,
    SynchMode,
    TradesColumns,
)
import datetime as dt


def round_millisecond(timestamp: pd.Timestamp):
    return (timestamp).floor("ms")  # + pd.offsets.Micro(500)


def _get_previous_or_next_row(row, df, primary_key, offset):
    reindexed_df = df.copy()
    reindexed_df[ATColumns.DTIME] = df.index
    row[ATColumns.DTIME] = row.name
    reindexed_df.index = reindexed_df[primary_key]
    df_shifted = reindexed_df.shift(offset)
    index_tuple = tuple(row[primary_key])
    mask = df_shifted.index == index_tuple
    if offset == 1:
        mask[0] = False
    else:
        mask[-1] = False
    if mask.any():
        target_row = df_shifted[mask].iloc[0]
        target_row.name = target_row[ATColumns.DTIME]
        return target_row
    else:
        return None


def get_previous_row(row, df, primary_key):
    return _get_previous_or_next_row(row, df, primary_key, 1)


def get_next_row(row, df, primary_key):
    return _get_previous_or_next_row(row, df, primary_key, -1)


def substr_to_sql(substr):
    if not (substr.endswith("%") or substr.startswith("%")):
        substr = "%" + substr + "%"
    return substr


def convert_to_datetime(df: pd.Series):
    try:
        df = pd.to_datetime(df, format="ISO8601", utc=True, exact=False)
    except ValueError:
        try:
            df = pd.to_datetime(df, format="mixed", utc=True)
        except ValueError:
            df = pd.to_datetime(
                df, format="%Y-%m-%d %H:%M:%S.%f", utc=True, exact=False
            )
    return df


def join_audit_trails(at_list: list[pd.DataFrame]):
    try:
        df = pd.concat(at_list)
    except AttributeError as e:
        # So pandas does not recognice some datetime dtypes and concat fails because of it
        for at_df in at_list:
            if at_df.empty:
                continue
            for time_col in [
                ATColumns.EXCHANGE_TIME,
                ATColumns.ORIGINAL_TIME,
                ATColumns.MATURITY_DATE,
                ATColumns.CLEARING_DATE,
            ]:
                at_df[time_col] = convert_to_datetime(at_df[time_col])
        df = pd.concat(at_list)
    if df.empty:
        return df
    df.sort_values(
        by=[ATColumns.DTIME, ATColumns.DF_ROW_ID], axis=0, inplace=True, kind="stable"
    )
    # print(df)
    df.reset_index(inplace=True)
    df.drop_duplicates(subset=AT_TABLE_PKEYS, keep="last", inplace=True)
    df.set_index("dtime", inplace=True)
    return df


def update_reassembled_data(reassembled_data, instr, t2t_object, mode):
    pkey = QUOTES_PKEYS if mode == SynchMode.QUOTES else TRADES_PKEYS
    if instr in reassembled_data and not t2t_object.df.empty:
        reassembled_data[instr] = pd.concat(
            [
                reassembled_data[instr],
                t2t_object.df.sort_values(
                    by=pkey,
                    axis=0,
                    inplace=False,
                    kind="stable",
                ),
            ]
        )
    elif not t2t_object.df.empty:
        reassembled_data[instr] = t2t_object.df.sort_values(
            by=pkey,
            axis=0,
            inplace=False,
            kind="stable",
        )


def join_t2t(data, mode: str):
    """
    Concatenates the dataframes loaded by `loader.py\\request_interval`

    Parameters
    ----------
    data : list of bd.LoadDict objects
        List of data from consecutive time intervals
    mode : str
        Either "trades" or "quotes" or "t2t"

    Returns
    -------
    dict[str:pd.DataFrame]
        For each instrument, dtype pair a dataframe with data.
    """
    reassembled_data = {}
    if mode in [SynchMode.TRADES, SynchMode.QUOTES]:
        for batch in data:
            for instr in batch:
                update_reassembled_data(reassembled_data, instr, batch[instr], mode)

    elif mode == "t2t":
        for batch in data:
            for instr in batch:
                instr_trades = instr[:-3] + SynchMode.TRADES
                instr_quotes = instr[:-3] + SynchMode.QUOTES
                update_reassembled_data(
                    reassembled_data,
                    instr_trades,
                    batch[instr].trades,
                    SynchMode.TRADES,
                )
                update_reassembled_data(
                    reassembled_data,
                    instr_quotes,
                    batch[instr].quotes,
                    SynchMode.QUOTES,
                )

    for instr, df in reassembled_data.items():
        if not ATColumns.DTIME in df:
            df[ATColumns.DTIME] = df.index
        if instr.endswith(SynchMode.TRADES):
            df.drop_duplicates(subset=TRADES_PKEYS, keep="last", inplace=True)
        else:
            df.drop_duplicates(subset=QUOTES_PKEYS, keep="last", inplace=True)
    return reassembled_data


def datetime_to_excel_format(date: dt.datetime):
    return date.strftime(EXCEL_SPREAD_DATE_FROMAT)


def get_ase_weights(ase_name: str, orig_email: str, date: dt.datetime):
    """Search the excels containing spread defintion for the spread weights of `ase_name`

    Parameters
    ----------
    ase_name : str
        Spread name
    orig_email : str
        Originator email as appears in the AuditTrail
    date : str
        Date of the trades

    Returns
    -------
    (list[int] | None)
        A list of weights if the spread definition is found, otherwise `None`
    """
    date = datetime_to_excel_format(date)
    plausible_files = [
        filename
        for filename in os.listdir(SPREAD_DEFINITIONS)
        if filename.startswith(orig_email)
        and filename.endswith(".csv")
        and filename.removeprefix(orig_email)[:8] <= date
    ]
    sorted_by_date = sorted(plausible_files, reverse=True)
    # sorting a few filenames is better for the runtime than accessing a lot of old files
    for filename in sorted_by_date:
        df = pd.read_csv(os.path.join(SPREAD_DEFINITIONS, filename), sep=";")
        if ase_name in df["ArfimaName"].values:
            return [
                int(float(el))
                for el in df[df["ArfimaName"] == ase_name]
                .iloc[0]
                .loc["WeightsSpread"]
                .split(",")
            ]
    return None


def compute_stats_correctly(df: pd.DataFrame, p: list[int]):
    # Okay so pandas decided to interpolate data to compute percentiles
    describe = df.describe([]).drop("50%")
    percentiles = df.quantile(p, interpolation="higher")
    percentiles.index = [f"{int(p*100)}%" for p in percentiles.index]
    statistics_df = pd.concat([describe, percentiles])

    new_col_order = ["count", "mean", "std", "min"] + list(percentiles.index) + ["max"]
    return statistics_df[new_col_order]


def clean_interproducts(orders_df: pd.DataFrame):
    inter_product_orders = orders_df[
        orders_df[ATColumns.INSTRUMENT].str.contains("Inter-Product")
    ]
    order_ids = inter_product_orders[ATColumns.TT_ORDER_ID].unique()
    parent_ids = inter_product_orders[ATColumns.TT_PARENT_ID].unique()
    filtered_df = orders_df[~orders_df[ATColumns.TT_ORDER_ID].isin(order_ids)]
    filtered_df = filtered_df[~filtered_df[ATColumns.TT_PARENT_ID].isin(parent_ids)]
    return filtered_df
