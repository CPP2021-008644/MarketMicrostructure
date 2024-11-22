# import libraries
from decimal import Decimal
import pandas as pd
import pandas as pd
import numpy as np
import datetime as dt
from numbers import Number


def _range_of_data_where_price(
    start_time,
    df: pd.DataFrame,
    price_col: str,
    does_not_decrease: bool = False,
    does_not_increase: bool = False,
    adjacent_interval: bool = True,
    end_time: dt.datetime = None,
    price_threshold=None,
):
    """
    Generic function to return intervals of data for a given dataframe.
    Useful to compute liquidity, duration, etc, with various modes.
    Assumes that there is a quote event at start_time or after in price_col column.
    Five modes:
        - Normal mode, (when you input only the mandatory params). From start time, it
            returns the timeseries until the first change in price
                It only returns adjacent data. i.e [9,9,8,9] will return [9,9]
        - When adjacent_interval is True, it will only return a subinterval of the original dataframe
            containing the first row:
            - does_not_decrease: returns the timeseries until the first change below the initial price (or price_threshold).
                    It only returns adjacent data. i.e [9,9,8,9] will return [9,9]
            - does_not_increase: returns the timeseries until the first change above the initial price  (or price_threshold).
                    It only returns adjacent data. i.e [9,9,10,9] will return [9,9]

        - When adjacent_interval is False, it will return a subsequence of the original dataframe
            containing the first row:
            - does_not_decrease: returns all the data above the price_threshold (or initial price)
                    Example: [9,9,8,9,10] with 9 as threshold will return [9,9,9,10]
            - does_not_increase: returns all the data below the price_threshold  (or initial price)
                    Example: [9,9,10,9,8] with 9 as threshold will return [9,9,9,8]

    """
    df = df[df.index >= start_time]
    if end_time is not None:
        df = df[df.index <= end_time]
    if price_threshold is None:
        price = df[price_col].iloc[0]
    else:
        price = price_threshold
    if not adjacent_interval:
        if does_not_decrease:
            return df[df[price_col] >= price]
        elif does_not_increase:
            return df[df[price_col] <= price]
        else:
            return df[df[price_col] == price]
    elif does_not_decrease:
        worse_df = df[df[price_col] < price]
    elif does_not_increase:
        worse_df = df[df[price_col] > price]
    else:
        worse_df = df[df[price_col] != price]
    if worse_df.empty:
        return df
    df = df[df.index < worse_df.index[0]]
    return df


def filter_dates(
    df: pd.DataFrame, start, end, start_strict=False, end_strict=False
) -> pd.DataFrame:
    """
    For a given dataframe with a datetime index, this function returns
    an interval of data. It has options to specify if the start and end points are open
    or closed
    """
    if df.empty:
        return df
    start_condition = df.index >= start
    if start_strict:
        start_condition = df.index > start
    end_condition = df.index <= end
    if end_strict:
        end_condition = df.index < end
    return df.loc[(start_condition) & (end_condition)]


def decimal_quantile(serie: pd.Series, q):
    return serie.apply(pd.to_numeric, downcast="float").quantile(q)


eps = 10 ** (-10)


def reference_price_plus_cross_bid_asks(
    bid_prices: pd.DataFrame,
    ask_prices: pd.DataFrame,
    from_date: dt.datetime,
    to_date: dt.datetime,
    spread_threshold_percentile: Number,
    params: dict = {},
):
    """
    Assumes that bid_prices and ask_prices are synchronized.
    to_date should be the point in time when you want to measure the reference price
    WARNING: returns None values if reference price could not be computed
    """
    bid_prices = filter_dates(bid_prices, from_date, to_date)
    ask_prices = filter_dates(ask_prices, from_date, to_date)
    if params.get("PreventTakingPriceWithoutBidAsk", True) and bid_prices.isna().all():
        raise ValueError("Reference Price error: inexistent BID")
    if params.get("PreventTakingPriceWithoutBidAsk", True) and ask_prices.isna().all():
        raise ValueError("Reference Price Vaerror: inexistent ASK")

    bid_ask_spread = ask_prices - bid_prices
    spread_quantile = (
        decimal_quantile(bid_ask_spread, spread_threshold_percentile) + eps
    )

    reference_price_bid = np.nanmedian(bid_prices[bid_ask_spread <= spread_quantile])
    reference_price_ask = np.nanmedian(ask_prices[bid_ask_spread <= spread_quantile])
    if params.get("PreventTakingPriceWithoutBidAsk", True) and (
        pd.isna(reference_price_bid) or pd.isna(reference_price_ask)
    ):
        raise ValueError("Reference Price error: bid ask too wide")
    reference_mid_price: Decimal = Decimal(0.5) * (
        reference_price_bid + reference_price_ask
    )

    if reference_mid_price.is_nan():
        if np.min(ask_prices) < np.max(bid_prices):
            length = ask_prices.shape[0]
            L = length
            while np.min(ask_prices.iloc[length - L :]) <= np.max(
                bid_prices.iloc[length - L :]
            ):
                L = round(L * 3 / 4)
            reference_mid_price = Decimal(0.5) * (
                np.min(ask_prices.iloc[length - L :])
                + np.max(bid_prices.iloc[length - L :])
            )
        else:
            reference_mid_price = Decimal(0.5) * (
                np.min(ask_prices) + np.max(bid_prices)
            )  # maybe this should be (bid+ask)/2
        reference_price_bid = bid_prices[bid_prices.notna()].iloc[-1]
        reference_price_ask = ask_prices[ask_prices.notna()].iloc[-1]

    return reference_price_bid, reference_price_ask, reference_mid_price


def strip_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values in a DataFrame."""
    df = df.rename(columns=lambda x: x.strip())
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def _convert_datetime(col):
    try:
        return pd.to_datetime(col, utc=True)
    except:
        return pd.to_datetime(col, format="%d %b %y, %H:%M", errors="coerce", utc=True)


def datetime_conversion(df):
    df["from_date"] = _convert_datetime(df["from_date"])
    if "to_date" in df.columns:
        df["to_date"] = _convert_datetime(df["to_date"])
    return df


def _revert_spread_names(spread_event_pairs):
    """
    Converts spread name, removing slashes and underscores
    """
    if spread_event_pairs is None:
        return spread_event_pairs
    for i, pair in enumerate(spread_event_pairs):
        spread_name = pair[0]
        spread_name = spread_name  # .replace("_", "").replace("-", "")
        spread_event_pairs[i] = (spread_name, pair[1])
    return spread_event_pairs


def load_dict_to_normal_name(load_dict_name: str):
    return load_dict_name.rsplit("_", 1)[0]


def get_legs_data(spread, legs):
    """Extracts specified leg data from a MarketData dictionary."""
    return {
        load_dict_to_normal_name(key): data.sort_values(
            by=["dtime", "quote_id"], kind="stable"
        )
        for key, data in spread.legs_data.items()
    }
