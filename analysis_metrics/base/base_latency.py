import numpy as np
import pandas as pd
from latency_config import (
    ATColumns,
    ETypes,
    SynchMode,
    QuotesColumns,
    TradesColumns,
    QUOTE_ORDERS,
    ORDER_TYPE_TO_SIDE,
    AT_TABLE_PKEYS,
)
import datetime as dt
from datetime import timezone
from audit_trail_manipulation import ATManipulation
import utils


class BaseLatency:
    @staticmethod
    def ns_to_datetime(ns):
        return np.datetime64(ns, "ns")

    @staticmethod
    def filter_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
        return df.loc[(df.index >= start) & (df.index <= end)]

    @staticmethod
    def closer_row(df: pd.DataFrame, target_index) -> pd.DataFrame:
        if df is None or df.empty:
            return None
        target_index = target_index.replace(tzinfo=timezone.utc)
        closest_index = np.argmin(abs(df.index - target_index))
        closest_row = df.iloc[closest_index]
        return closest_row

    @classmethod
    def ns_col_to_datetime(cls, df: pd.DataFrame, col: str):
        df[col] = df[col].apply(cls.ns_to_datetime)
        return df

    # TODO: look which typehints to change from Dataframe to Series
    @staticmethod
    def to_millisec(df: pd.Series) -> pd.Series:
        return df.dt.total_seconds() * 1000

    @staticmethod
    def to_microsec(df: pd.Series):
        return df.dt.total_seconds() * 1000000

    @staticmethod
    def remove_tzone(df: pd.DataFrame, col: str):
        df[col] = df[col].values
        return df

    @staticmethod
    def get_previous_row_exchtime(row, df, primary_key):
        prev_row = utils.get_previous_row(row, df, primary_key)
        if prev_row is None:
            return None
        return prev_row[ATColumns.EXCHANGE_TIME]

    @staticmethod
    def get_next_row_exchtime(row, df, primary_key):
        next_row = utils.get_next_row(row, df, primary_key)
        if next_row is None:
            return None
        return next_row[ATColumns.EXCHANGE_TIME]

    @staticmethod
    def get_float_price(df, col: str):
        """
        Turn column values into floats
        """
        df[col] = df[col].apply(pd.to_numeric)
        return df

    @staticmethod
    def select_ticker_from_t2t(instrument_name: str, t2t_data: pd.DataFrame, mode: str):
        bd_name = instrument_name + "_t2t"
        if bd_name in t2t_data:
            instrument_t2t = getattr(t2t_data[bd_name], mode)
        elif instrument_name + f"_{mode}" in t2t_data:
            bd_name = instrument_name + f"_{mode}"
            instrument_t2t = t2t_data[instrument_name + f"_{mode}"]
        else:
            print(f"No T2T data found for {instrument_name}")
            return None
        print(f"We are loading {len(instrument_t2t.index)} ticks ")
        return instrument_t2t

    @classmethod
    def prepare_loaded(cls, df: pd.DataFrame):
        """
        Changes the formats of columns with time information

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that will have its time information converted

        Returns
        -------
        pd.DataFrame
            Modified DataFrame
        """
        df.index = utils.convert_to_datetime(df.index)
        for time_col in [
            ATColumns.EXCHANGE_TIME,
            ATColumns.ORIGINAL_TIME,
            ATColumns.CLEARING_DATE,
        ]:
            df[time_col] = utils.convert_to_datetime(df[time_col])

        df = cls.ns_col_to_datetime(df, ATColumns.TIME_SENT)
        df = cls.remove_tzone(df, ATColumns.EXCHANGE_TIME)
        df = cls.remove_tzone(df, ATColumns.ORIGINAL_TIME)
        df = cls.get_float_price(df, ATColumns.ORDER_PRICE)
        return df

    @staticmethod
    def get_not_ase_rows(df: pd.DataFrame):
        return df[~df[ATColumns.PRODUCT].isna()]

    @classmethod
    def _reorder_cols(cls, exch_latencies: pd.DataFrame, latency_type: str):
        cols = exch_latencies.columns.tolist()
        cols.remove(latency_type)
        cols = [latency_type] + cols
        return exch_latencies[cols]

    @classmethod
    def exchange_latency(
        cls, df: pd.DataFrame, exch_latency_valid_orders: list, latency_type: str
    ):
        # outrights dataframe
        # New, Cancel y Replace de outrights : exchlatency +/- 1ms
        df = cls.get_not_ase_rows(df)
        exch_latencies = df[
            df[ATColumns.EXEC_TYPE].isin(exch_latency_valid_orders)
        ].copy()
        exch_latencies[latency_type] = cls.to_millisec(
            exch_latencies[ATColumns.TIME_SENT]
            - exch_latencies[ATColumns.EXCHANGE_TIME]
        )
        exch_latencies = cls._reorder_cols(exch_latencies, latency_type=latency_type)
        return exch_latencies

    @classmethod
    def _wrapper_latency_by_id(cls, group_df: pd.DataFrame, latency_type: str):
        news = group_df[group_df[ATColumns.EXEC_TYPE] == ETypes.NEW]
        wrapper_news = news[news[ATColumns.ORDER_ROUTE] == "Algo SE"]
        real_news = news[news[ATColumns.ORDER_ROUTE] == "Direct"]

        def wrapper_latency(row):
            real_time = row[ATColumns.ORIGINAL_TIME]
            matching_wrapper_news = wrapper_news[
                wrapper_news[ATColumns.ORIGINAL_TIME] <= real_time
            ]
            previous_real_news = real_news[
                real_news[ATColumns.EXCHANGE_TIME] < row[ATColumns.EXCHANGE_TIME]
            ]
            if not previous_real_news.empty:
                previous_real_new = previous_real_news.iloc[-1]
                matching_wrapper_news = matching_wrapper_news[
                    matching_wrapper_news[ATColumns.ORIGINAL_TIME]
                    > previous_real_new[ATColumns.ORIGINAL_TIME]
                ]
            if matching_wrapper_news.empty:
                return None
            matching_wrapper_new = matching_wrapper_news.iloc[-1]
            return (
                real_time - matching_wrapper_new[ATColumns.ORIGINAL_TIME]
            ).total_seconds() * 1000

        wrapper_latencies = real_news.apply(
            lambda row: wrapper_latency(row), axis=1, result_type="reduce"
        ).dropna()
        wrapper_latencies.name = latency_type
        return wrapper_latencies

    @classmethod
    def wrapper_latency(cls, df: pd.DataFrame, latency_type: str):
        """Function to measure latencies created by the speed bump in some exchanges (EUREX)"""
        df = cls.get_not_ase_rows(df)
        # coalesce
        df["merged_id"] = df[ATColumns.TT_PARENT_ID].combine_first(
            df[ATColumns.TT_ORDER_ID]
        )

        wrapper_latencies = df.groupby("merged_id").apply(
            lambda group_df: cls._wrapper_latency_by_id(group_df, latency_type)
        )
        if wrapper_latencies.empty:
            wrapper_latencies[latency_type] = None

        wrapper_latencies = wrapper_latencies.reset_index(level=[0]).dropna(
            subset=latency_type
        )
        return wrapper_latencies

    @classmethod
    def server_to_exchange_latency(
        cls, df: pd.DataFrame, latency_type: str, valid_orders: list = ["NEW"]
    ):
        df = cls.get_not_ase_rows(df)
        server_to_exchange_latencies = df[
            df[ATColumns.EXEC_TYPE].isin(valid_orders)
        ].copy()
        # First filter to discard corrupt data
        server_to_exchange_latencies = server_to_exchange_latencies[
            server_to_exchange_latencies[ATColumns.EXCHANGE_TIME]
            >= server_to_exchange_latencies[ATColumns.ORIGINAL_TIME]
        ]
        server_to_exchange_latencies[latency_type] = cls.to_millisec(
            server_to_exchange_latencies[ATColumns.EXCHANGE_TIME]
            - server_to_exchange_latencies[ATColumns.ORIGINAL_TIME]
        )
        server_to_exchange_latencies = cls._reorder_cols(
            server_to_exchange_latencies, latency_type=latency_type
        )
        return server_to_exchange_latencies

    @staticmethod
    def filter_instruments(df: pd.DataFrame, substr: str = None, instr_list=[]):
        if substr:
            if substr.endswith("%") and substr.startswith("%"):
                return df[df[ATColumns.INSTRUMENT].str.contains(substr[1:-1])]
            elif substr.endswith("%"):
                return df[df[ATColumns.INSTRUMENT].str.startswith(substr[:-1])]
            elif substr.startswith("%"):
                return df[df[ATColumns.INSTRUMENT].str.endswith(substr[:-1])]
            return df[df[ATColumns.INSTRUMENT].str.contains(substr)]
        else:
            return df[df[ATColumns.INSTRUMENT].isin(instr_list)]

    @staticmethod
    def opposite_side(side: str):
        if side.lower() == "buy":
            return "sell"
        elif side.lower() == "sell":
            return "buy"
        return side

    @classmethod
    def get_aggressor_from_at(cls, audit_df: pd.DataFrame):
        return audit_df.apply(
            lambda row: (
                row[ATColumns.BUY_SELL].lower()
                if row[ATColumns.PASSIVE_AGRESSIVE] == "A"
                else cls.opposite_side(row[ATColumns.BUY_SELL])
            ),
            axis=1,
        )

    @classmethod
    def t2t_get_complete_dtime(cls, t2t_df: pd.DataFrame) -> pd.DataFrame:
        # given a t2t dataframe get an index with exact times in nanoseconds
        return t2t_df.index.floor("s") + pd.to_timedelta(t2t_df["nanos"])

    @classmethod
    def compute_offsets(cls, audit_row, matched_tick, exact: bool, type):
        result = {}
        if matched_tick is None:
            return {
                f"exch_synch_offset_{type}": None,
                "exch_time_audit_trail": None,
                "dtime_audit_trail": None,
                "row_id_audit_trail": None,
                "dtime_quote": None,
                "id_quote": None,
                "exact": None,
            }
        t2t_dtime = matched_tick.name.replace(tzinfo=None)
        result[f"exch_synch_offset_{type}"] = (
            audit_row[ATColumns.EXCHANGE_TIME] - t2t_dtime
        )
        result["exch_time_audit_trail"] = audit_row[ATColumns.EXCHANGE_TIME]
        result["dtime_audit_trail"] = (
            audit_row[ATColumns.DTIME]
            if ATColumns.DTIME in audit_row.index.values
            else audit_row.name
        )
        result["row_id_audit_trail"] = audit_row[ATColumns.DF_ROW_ID]
        result["dtime_quote"] = matched_tick.name
        result["id_quote"] = (
            matched_tick[QuotesColumns.QUOTE_ID]
            if type == SynchMode.QUOTES
            else matched_tick[TradesColumns.EXCHANGE_TRADE_ID]
        )
        result["exact"] = exact
        return result

    @classmethod
    def synchronize_replace(
        cls, row: pd.Series, original_row: pd.Series, filtered_ticks: pd.DataFrame
    ):
        """
        Find replace or restate orders in t2t data

        Parameters
        ----------
        row : pd.Series
            AT row
        original_row : pd.Series
            AT row of the previous order that is getting replaced
        filtered_ticks : pd.DataFrame
            t2t data

        Returns
        -------
        dict
            dict mapping types of offsets to values
        """
        order_price, order_size, side = row[
            [ATColumns.ORDER_PRICE, ATColumns.ORDER_SIZE, ATColumns.BUY_SELL]
        ]
        size_variation = order_size - original_row[ATColumns.ORDER_SIZE]
        price_variation = order_price - original_row[ATColumns.ORDER_PRICE]
        order_prefix = ORDER_TYPE_TO_SIDE[side]
        mask_same_price = filtered_ticks[f"{order_prefix}_price"] == order_price
        targets = filtered_ticks[mask_same_price]
        # compute increase/decrease of order number/size/price with respect to the previous tick
        order_diffs = targets[f"{order_prefix}_orders"].diff()
        size_diffs = targets[f"{order_prefix}_size"].diff()
        diffs_in_price = targets[f"{order_prefix}_price"].diff()
        # the number of orders stays the same when there is a replace
        mask_orders_same = order_diffs == 0
        if price_variation == 0:  # only order size changes
            mask_size_diff = size_diffs == size_variation
            mask_final = mask_size_diff & mask_orders_same
            matching_orders = targets[mask_final]
        elif size_variation == 0:  # only order price changes
            mask_price_diff = diffs_in_price == price_variation
            mask_final = mask_price_diff & mask_orders_same
            matching_orders = targets[mask_final]
        elif price_variation != 0 and size_variation != 0:
            # both order price and size changed, cannot find our order exactly
            matching_orders = None
        if (matching_orders is not None) and (not matching_orders.empty):
            tick = cls.closer_row(matching_orders, row[ATColumns.EXCHANGE_TIME])
            return cls.compute_offsets(row, tick, exact=True, type=SynchMode.QUOTES)
        # if there is order aggregation there is no way we can find our replace order
        # you cannot find your volume (could be increased by other new orders or decreased by cancels)
        # or your order number (orders can be both added or canceled)
        tick = cls.closer_row(targets, row[ATColumns.EXCHANGE_TIME])
        return cls.compute_offsets(row, tick, exact=False, type=SynchMode.QUOTES)

    @classmethod
    def synchronize_new_or_cancel(cls, row, filtered_ticks, is_cancel):
        """
        Method to find a new order or a cancel order in the t2t data. We treat a cancel (flag `is_cancel == True`) as a new with negative order size.

        Parameters
        ----------
        row : pd.Series
            AT row
        filtered_ticks : pd.DataFrame
            t2t data
        is_cancel : bool
            If is a cancel order

        Returns
        -------
        dict
            dict mapping types of offsets to values
        """
        order_price, order_size, side = row[
            [ATColumns.ORDER_PRICE, ATColumns.ORDER_SIZE, ATColumns.BUY_SELL]
        ]
        if is_cancel:
            order_size = -order_size
        order_prefix = ORDER_TYPE_TO_SIDE[side]
        mask_same_price = filtered_ticks[f"{order_prefix}_price"] == order_price
        targets = filtered_ticks[mask_same_price]
        # it's a new trade so we should see both an increase in the volume in our side and an increase by one in the number of orders
        # for cancels it's the same but volume will decrease
        size_diffs = targets[f"{order_prefix}_size"].diff()
        order_diffs = targets[f"{order_prefix}_orders"].diff()
        mask_size_diff = size_diffs == order_size
        mask_order_diff = (order_diffs == -1) if is_cancel else (order_diffs == 1)
        mask_final = mask_size_diff & mask_order_diff
        matching_orders = targets[mask_final]
        if (matching_orders is not None) and (not matching_orders.empty):
            matched_tick = cls.closer_row(matching_orders, row[ATColumns.EXCHANGE_TIME])
            return cls.compute_offsets(
                row, matched_tick, exact=True, type=SynchMode.QUOTES
            )
        # if we don't find the exact trade, we pick the closest order with the same price
        # there is no better way: if there is order aggregation you cannot find your volume (could be increased by other new orders or decreased by cancels)
        # or your order number (orders can be both added or canceled)
        matched_tick = cls.closer_row(targets, row[ATColumns.EXCHANGE_TIME])
        return cls.compute_offsets(
            row, matched_tick, exact=False, type=SynchMode.QUOTES
        )

    @classmethod
    def match_single_quote_with_t2t(
        cls,
        audit_quote: pd.Series,
        t2t_quotes,
        max_synch_offset: dt.timedelta,
        at_quotes: pd.DataFrame,
        prev_exchtime,
        next_exchtime,
    ):
        if (
            audit_quote[ATColumns.FILL_TYPE] == "INDIVIDUAL_LEG_OF_A_MULTI_LEG_SECURITY"
        ):  # Synthetic products are difficult/impossible to match
            # TODO: download data for the synthetic product to perform synchronization
            return cls.compute_offsets(audit_quote, None, False, type=SynchMode.TRADES)
        exch_time_at = audit_quote[ATColumns.EXCHANGE_TIME]
        t2t_quotes.index = cls.t2t_get_complete_dtime(t2t_quotes)
        if prev_exchtime is not None and next_exchtime is not None:
            # the way we filter quotes is chosen to avoid multiple audit trail entries to be matched to the same t2t quote.
            # If you imagine the AT exchange times on a line, an AT quote can be matched to some tick whose time is between
            # the midpoint of itself and the previous AT quote, and the midpoint of itself and the next AT quote. This creates disjoint intervals.
            # also, we only consider ticks that are within max_synch_offset of the timestamp in the audit trail
            ticks_within_offset = t2t_quotes[
                (
                    t2t_quotes.index.tz_localize(None)
                    > max(
                        exch_time_at - max_synch_offset,
                        exch_time_at - 0.5 * (exch_time_at - prev_exchtime),
                    )
                )
                & (
                    t2t_quotes.index.tz_localize(None)
                    < min(
                        exch_time_at + max_synch_offset,
                        exch_time_at + 0.5 * (next_exchtime - exch_time_at),
                    )
                )
            ]
        else:
            ticks_within_offset = t2t_quotes[
                (t2t_quotes.index.tz_localize(None) > exch_time_at - max_synch_offset)
                & (t2t_quotes.index.tz_localize(None) < exch_time_at + max_synch_offset)
            ]
        ticks_within_offset = ticks_within_offset.astype(
            {QuotesColumns.BID_PRICE: np.float64, QuotesColumns.ASK_PRICE: np.float64}
        )
        # we always return a result, we have a row that tells us if the matching was exact or not
        if audit_quote[ATColumns.EXECUTION_TYPE] in [ETypes.NEW, ETypes.CANCELED]:
            tick = cls.synchronize_new_or_cancel(
                audit_quote,
                ticks_within_offset,
                is_cancel=(audit_quote[ATColumns.EXECUTION_TYPE] == ETypes.CANCELED),
            )
        else:  # quote is a replace/restate
            original = ATManipulation.find_previous_replace(audit_quote, at_quotes)
            tick = cls.synchronize_replace(audit_quote, original, ticks_within_offset)
        return tick

    @classmethod
    def _same_millisecond_matching(cls, at_trades, t2t_trades):
        """
        Function matching AT trades that have the same `ATColumns.EXCHANGE_TIME` and have the same `ATColumns.TT_ORDER_ID`
        with t2t trades.

        Parameters
        ----------
        at_trades : pd.DataFrame
            AT rows corresponding to trades that have the same `ATColumns.EXCHANGE_TIME` and have the same `ATColumns.TT_ORDER_ID`
        t2t_trades : pd.DataFrame
            t2t data rows corresponding to the trades

        Returns
        -------
        pd.DataFrame
            The rows are: for each of the `at_trades` either a row with the matching information or one with `None` values.
        2024-01-11 13:30:01.064
        """
        j = 0
        n = len(t2t_trades)
        matched_trades = []
        curr_tick = t2t_trades.iloc[j]
        curr_qty = curr_tick[TradesColumns.TRADE_SIZE]
        for (
            _,
            trade_at,
        ) in at_trades.iterrows():  # iterate over the trades in `at_trades`
            while (
                j < n
            ):  # iterate over the trades in `t2t_trades`, with the possibility of "staying" at the same trade if the
                # trade in the AT doesn't exhaust the quantity in the t2t trade
                if (
                    trade_at[ATColumns.FILL_SIZE] == curr_qty
                    and trade_at[TradesColumns.AGGRESSOR]
                    == curr_tick[TradesColumns.AGGRESSOR]
                    and trade_at[ATColumns.ORDER_PRICE]
                    == curr_tick[TradesColumns.TRADE_PRICE]
                ):  # quantity, aggressor and price match exactly between AT and t2t
                    matched_trades.append(
                        cls.compute_offsets(
                            trade_at,
                            curr_tick,
                            exact=True,
                            type=SynchMode.TRADES,
                        )
                    )  # add row to the result with the matching information
                    j += 1  # move to next t2t trade
                    if (
                        j >= n
                    ):  # break out of the while loop to avoid getting an error in the next line
                        break
                    curr_tick = t2t_trades.iloc[j]
                    curr_qty = curr_tick[TradesColumns.TRADE_SIZE]
                    break
                elif (
                    trade_at[ATColumns.FILL_SIZE] <= curr_qty
                    and curr_tick[TradesColumns.AGGRESSOR]
                    in [trade_at[TradesColumns.AGGRESSOR], "unknown"]
                    and trade_at[ATColumns.ORDER_PRICE]
                    == curr_tick[TradesColumns.TRADE_PRICE]
                ):  # match is not exact: t2t quantity is greater or equal, the aggressor can be unknown (but price still has to be the same)
                    matched_trades.append(
                        cls.compute_offsets(
                            trade_at,
                            t2t_trades.iloc[j],
                            exact=False,
                            type=SynchMode.TRADES,
                        )
                    )
                    curr_qty -= float(trade_at[ATColumns.FILL_SIZE])
                    break
                else:  # go to the next tick
                    j += 1
                    if j >= n:
                        break
                    curr_tick = t2t_trades.iloc[j]
                    curr_qty = curr_tick[TradesColumns.TRADE_SIZE]
        if not matched_trades:
            return
        return pd.DataFrame(matched_trades)

    @classmethod
    def match_trade_group_with_t2t(
        cls,
        group_trades: pd.DataFrame,
        trades_t2t_original,
        max_synch_offset: dt.timedelta,
        has_micro=False,
    ):
        """
        This function limits the t2t data a group of trades in the audit-trail is matched with.

        Parameters
        ----------
        group_trades : pd.DataFrame
            Trades from audit-trail with same exchange time
        trades_t2t_original : pd.DataFrame
            All the t2t data
        max_synch_offset : dt.timedelta
            How many milliseconds away can we match a trade from the audit-trail to the t2t trades.
        has_micro : bool, optional
            If the t2t dtime (i.e. the exchange time) has microsecond information, by default False

        Returns
        -------
        pd.DataFrame
            Matched trades
        """
        # Things that should match: instrument, order_price, fill_size, buy_sell (depends on agressor)
        # Assumes that dates are in the same timezone
        if (
            group_trades[ATColumns.FILL_TYPE]
            == "INDIVIDUAL_LEG_OF_A_MULTI_LEG_SECURITY"
        ).any():  # Synthetic products are difficult/impossible to match: return None row
            return
        at_exchtime = group_trades.name[0]  # these trades all have same exchange time
        group_trades = group_trades.assign(**{ATColumns.EXCHANGE_TIME: at_exchtime})
        trades_t2t_original.index = cls.t2t_get_complete_dtime(trades_t2t_original)
        if has_micro:
            # if there is microsecond information in the t2t data, we know that "exchange_time" is obtained by truncating the
            # dtime of the t2t data. This means that it is sufficient to look at the interval of 1 millisecond at the right of
            # the trade.
            ticks_within_offset = trades_t2t_original[
                (trades_t2t_original.index.tz_localize(None) >= at_exchtime)
                & (
                    trades_t2t_original.index.tz_localize(None)
                    <= at_exchtime + dt.timedelta(milliseconds=1)
                )
            ]
        else:
            ticks_within_offset = trades_t2t_original[
                (
                    trades_t2t_original.index.tz_localize(None)
                    > at_exchtime - max_synch_offset
                )
                & (
                    trades_t2t_original.index.tz_localize(None)
                    < at_exchtime + max_synch_offset
                )
            ]

        trades_t2t: pd.DataFrame = ticks_within_offset.astype(
            {"trade_price": np.float64, "trade_size": np.float64}
        )

        return cls._same_millisecond_matching(group_trades, trades_t2t)

    @classmethod
    def match_at_trades_single_instrument_with_t2t(
        cls,
        instrument_name: str,
        instrument_trades_df: pd.DataFrame,
        t2t_load_dict,
        max_synch_offset=dt.timedelta(milliseconds=1),
    ):
        instrument_trades_df["aggressor"] = cls.get_aggressor_from_at(
            instrument_trades_df
        )
        trades_t2t = cls.select_ticker_from_t2t(
            instrument_name, t2t_load_dict, mode=SynchMode.TRADES
        )
        has_micro = (
            (trades_t2t.index.microsecond % 1000) != 0
        ).any()  # if the microseconds don't end with 3 zeros then it has microsecond information
        matched_single_instrument = (
            instrument_trades_df.reset_index()
            .groupby([ATColumns.EXCHANGE_TIME, ATColumns.BUY_SELL])
            .apply(
                lambda trade_at: cls.match_trade_group_with_t2t(
                    trade_at,
                    trades_t2t,
                    max_synch_offset,
                    has_micro,
                )
            )
        )
        if matched_single_instrument.empty:
            raise Exception(
                "We couldn't match any trade. This error often presents itself when using the `audit_trail` branch in beautiful data, which sometimes doesn't load all the necessary rows."
            )
        offsets = matched_single_instrument.dropna()
        offsets["exch_synch_offset_trades"] = offsets[
            ["exch_synch_offset_trades"]
        ].apply(cls.to_millisec)
        return offsets

    @classmethod
    def match_quotes_single_instrument(
        cls,
        instrument_name: str,
        instrument_quotes: pd.DataFrame,
        t2t_load_dict,
        max_synch_offset: dt.timedelta,
    ):
        """
        Matches the audit trail quotes relative to an instrument to the respective t2t quotes.


        Parameters
        ----------
        instrument_name : str
            Name of the instrument
        instrument_quotes : pd.DataFrame
            AT entries relative to the instrument
        t2t_load_dict : dict
            Dictionary returned by bd, that receives a string specifying product and type of ticks
            and returns a dataframe.
        max_synch_offset : dt.timedelta
            MAximum difference in milliseconds between audit trail quotes and t2t quotes

        Returns
        -------
        pd.DataFrame
            Matched quotes with offsets in milliseconds.
        """
        instrument_t2t = cls.select_ticker_from_t2t(
            instrument_name, t2t_load_dict, mode=SynchMode.QUOTES
        )
        quotes_matched = instrument_quotes.apply(
            lambda at_quote: cls.match_single_quote_with_t2t(
                at_quote,
                instrument_t2t,
                max_synch_offset,
                instrument_quotes,
                cls.get_previous_row_exchtime(
                    at_quote, instrument_quotes, AT_TABLE_PKEYS
                ),
                cls.get_next_row_exchtime(at_quote, instrument_quotes, AT_TABLE_PKEYS),
            ),
            axis=1,
            result_type="expand",
        )
        offsets = quotes_matched.dropna()
        offsets[["exch_synch_offset_quotes"]] = offsets[
            ["exch_synch_offset_quotes"]
        ].apply(cls.to_millisec)
        return offsets

    @classmethod
    def compute_matching_stats(cls, original_df, matched_df, type):
        # print how many of the quotes we managed to match, even approximately
        total_ticks = len(
            original_df[
                original_df[ATColumns.FILL_TYPE]
                != "INDIVIDUAL_LEG_OF_A_MULTI_LEG_SECURITY"
            ]
        )
        matched_ticks = len(matched_df)
        exactly_matched_ticks = (
            len(matched_df[matched_df["exact"] == True]) if not matched_df.empty else 0
        )
        t2t_info = {}
        t2t_info[f"total {type}"] = total_ticks
        t2t_info[f"matched {type}"] = matched_ticks
        t2t_info[f"exactly matched {type}"] = exactly_matched_ticks
        t2t_info[f"proportion of matched {type}"] = (
            matched_ticks / total_ticks if total_ticks > 0 else 0
        )
        t2t_info[f"proportion of exactly matched {type}"] = (
            (exactly_matched_ticks / total_ticks) if total_ticks > 0 else 0
        )
        return pd.DataFrame(list(t2t_info.items()), columns=["Metric", "Value"])

    @classmethod
    def match_at_trades_with_t2t(cls, not_ase_df: pd.DataFrame, t2t_load_dict):
        trades = not_ase_df[not_ase_df[ATColumns.EXECUTION_TYPE] == ETypes.TRADE]
        matched_offsets = trades.groupby(ATColumns.INSTRUMENT).apply(
            lambda group_df: cls.match_at_trades_single_instrument_with_t2t(
                group_df.name, group_df, t2t_load_dict
            )
        )
        # print how many of the trades we managed to match even approximately
        trades_info = cls.compute_matching_stats(
            trades, matched_offsets, SynchMode.TRADES
        )
        return matched_offsets, trades_info

    @classmethod
    def match_at_quotes_with_t2t(
        cls,
        audit_rows_tickers: pd.DataFrame,
        t2t_load_dict,
        max_synch_offset: dt.timedelta,
    ):
        quotes = audit_rows_tickers[
            audit_rows_tickers[ATColumns.EXEC_TYPE].isin(QUOTE_ORDERS)
        ]
        quotes_matching_result = quotes.groupby(ATColumns.INSTRUMENT).apply(
            lambda group_df: cls.match_quotes_single_instrument(
                group_df.name, group_df, t2t_load_dict, max_synch_offset
            )
        )
        quotes_info = cls.compute_matching_stats(
            quotes, quotes_matching_result, SynchMode.QUOTES
        )
        return quotes_matching_result, quotes_info

    @classmethod
    def audit_trail_t2t_synchronization(
        cls,
        audit_df: pd.DataFrame,
        t2t_load_dict,
        tickers: list[str],
        max_synch_offset: dt.timedelta,
        mode,
    ):
        """
        This is the method that should be callled by other modules to perform trades and quotes synchronization between audit trail and t2t data.

        Parameters
        ----------
        audit_df : pd.DataFrame
            Audit trail data
        t2t_load_dict : dict[str: pd.DataFrame]
            Takes as an argument instrument name followed by suffixes `"_trades"` or `"_quotes"` and gives the t2t data
        tickers : list[str]
            Tickers of the instruments we synchronize
        max_synch_offset : dt.timedelta
            Time window outside of which you give up on matching an audit trail row with a tick.
        mode : str
            "trades", "quotes" or "both" (which are the modes in `SynchMode` in `config.py`)

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            Results of the synchronization
            1st: trades offsets
            2nd: quotes offsets
        """
        # Assumes that dates are in the same timezone
        not_ase_df = cls.get_not_ase_rows(audit_df)
        audit_rows_tickers = not_ase_df[not_ase_df[ATColumns.INSTRUMENT].isin(tickers)]
        trades_offsets, trades_info, quotes_offsets, quotes_info = (
            None,
            None,
            None,
            None,
        )
        if mode == SynchMode.TRADES or mode == SynchMode.BOTH:
            trades_offsets, trades_info = cls.match_at_trades_with_t2t(
                audit_rows_tickers, t2t_load_dict
            )
        if mode == SynchMode.QUOTES or mode == SynchMode.BOTH:
            quotes_offsets, quotes_info = cls.match_at_quotes_with_t2t(
                audit_rows_tickers, t2t_load_dict, max_synch_offset
            )
        return trades_offsets, trades_info, quotes_offsets, quotes_info
