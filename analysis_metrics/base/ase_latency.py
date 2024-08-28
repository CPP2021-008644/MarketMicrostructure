import logging
from latency_config import ATColumns, ETypes
from utils import get_ase_weights
from analysis_metrics.base.base_latency import BaseLatency
from math import gcd
import pandas as pd
import numpy as np


LOG = logging.getLogger(__name__)


class ASELatency(BaseLatency):
    @staticmethod
    def child_rows(df: pd.DataFrame, tt_parent_id: str):
        return df[df[ATColumns.TT_PARENT_ID] == tt_parent_id]

    @classmethod
    def get_ase_rows(cls, df: pd.DataFrame):
        return df[df[ATColumns.PRODUCT].isna()]

    @classmethod
    def _get_ase_name_from_order_id(cls, df, order_id):
        try:
            return df[df[ATColumns.TT_ORDER_ID] == order_id][ATColumns.INSTRUMENT].iloc[
                0
            ]
        except IndexError:
            return ""

    @classmethod
    def _latency_include_useful_information(
        cls,
        latency,
        latency_name: str,
        ase_name: str,
        active_leg_orders: pd.DataFrame,
        hedge_leg_orders: pd.DataFrame,
    ):
        df = pd.DataFrame(
            data={
                latency_name: latency,
                ATColumns.INSTRUMENT: ase_name,
                "t0_exchange_transaction_id": active_leg_orders[
                    ATColumns.EXCH_TRANS_ID
                ].values,
                "t1_exchange_transaction_id": hedge_leg_orders[
                    ATColumns.EXCH_TRANS_ID
                ].values,
            }
        )
        return df

    @classmethod
    def requote_latency(
        cls,
        liquid_leg_quotes: pd.DataFrame,
        illiquid_leg_quotes: pd.DataFrame,
        threshold=None,
    ):
        """
        Computes the time it takes to our logic to send a new quote after a market update. The market updates are our own quotes.
        We are assuming we receive the acknowledge roughly at the same time we receive the market update, so we use the field `ATColumns.TIME_SENT`
        of our own quotes.

        Parameters
        ----------
        liquid_leg_quotes : pd.DataFrame
            Quotes that we send after reacting to the market
        illiquid_leg_quotes : pd.DataFrame
            Quotes we react to
        threshold : int
            Offsets larger than this will be dropped and won't be used, neither for statistics nor for plotting etc.
        Returns
        -------
        pd.DataFrame
            Dataframe with "requote_latency", and timestamps of the illiquid and liquid legs.
        """
        # mask_new = liquid_leg_quotes[ATColumns.EXECUTION_TYPE] == ETypes.NEW
        requote_time = liquid_leg_quotes[ATColumns.EXCHANGE_TIME].copy()
        # requote_time.loc[mask_new] = liquid_leg_quotes.loc[
        #     mask_new, ATColumns.ORIGINAL_TIME
        # ]
        requote_time = requote_time.apply(lambda timestamp: timestamp.floor("ms"))
        received_time = illiquid_leg_quotes[ATColumns.TIME_SENT].apply(
            lambda timestamp: timestamp.floor("ms")
        )
        result = pd.DataFrame(
            data={
                "requote_latency": cls.to_millisec(
                    requote_time - received_time.values
                ).values,
                "illiquid_dtime": illiquid_leg_quotes.index,
                "illiquid_row_id": illiquid_leg_quotes[ATColumns.DF_ROW_ID].values,
                "liquid_dtime": liquid_leg_quotes.index,
                "liquid_row_id": liquid_leg_quotes[ATColumns.DF_ROW_ID].values,
            },
            index=liquid_leg_quotes.index,
        )
        if threshold is not None:
            return result[result["requote_latency"] <= threshold]
        return result

    # Example where _hedge_latency_single_leg and _hedge_latency_with_ack_time_single_leg differ by 15ms:
    # b03cd0e6-35e7-4779-a8d5-db07a9747d13 2023-04-12
    @classmethod
    def _hedge_latency_single_leg(
        cls, latency_name: str, ase_name: str, active_leg_fills, hedge_leg_orders
    ):
        # This makes an approximation of the mapping from fills of the quoted leg to new orders in the hedge leg
        # The correct way is to take into account legs weights
        milli_latency = cls.to_millisec(
            hedge_leg_orders[ATColumns.ORIGINAL_TIME]
            - active_leg_fills[ATColumns.TIME_SENT].values
        )  # .apply(lambda lat: max(lat, 0))
        return cls._latency_include_useful_information(
            milli_latency, latency_name, ase_name, active_leg_fills, hedge_leg_orders
        )

        # NOTE: timesent is nanoseconds while exchange time is milliseconds

    @classmethod
    def _hedge_latency_with_ack_time_single_leg(
        cls, latency_name: str, ase_name: str, active_leg_fills, hedge_leg_orders
    ):
        # This makes an approximation of the mapping from fills of the quoted leg to new orders in the hedge leg
        # The correct way is to take into account legs weights
        milli_latency = cls.to_millisec(
            hedge_leg_orders[ATColumns.TIME_SENT]
            - active_leg_fills[ATColumns.TIME_SENT].values
        )
        return cls._latency_include_useful_information(
            milli_latency, latency_name, ase_name, active_leg_fills, hedge_leg_orders
        )

        # NOTE: timesent is nanoseconds while exchange time is milliseconds

    @classmethod
    def _combine_leg_latencies(
        cls, leg_latencies: list[pd.DataFrame], latency_name: str
    ):
        """
        Concatenates the dataframes in `leg_latencies` into a single dataframe
        Parameters
        ----------
        leg_latencies : list[pd.DataFrame]
            Contains the list of dataframes corresponding to th edifferent hedge legs of a spread,
            as produced by `hedge_latency_single`.
        latency_name : str
            Type of latency measures the dataframes contain.

        Returns
        -------
        pd.DataFrame
            The combined dataframes
        """
        if len(leg_latencies) == 1:
            return leg_latencies[0]
        latencies_df = pd.concat(
            [leg[latency_name].reset_index(drop=True) for leg in leg_latencies],
            axis=1,
            ignore_index=True,
        )
        indexes = [pd.Series(leg.index) for leg in leg_latencies]
        dtimes_df = pd.concat(indexes, axis=1)
        idxmax = latencies_df.idxmax(axis=1)
        dtimes_df["idxmax"] = idxmax
        combined_index = dtimes_df.apply(lambda row: row.iloc[row["idxmax"]], axis=1)

        combined = []
        for i, leg_df in enumerate(leg_latencies):
            leg_df = leg_df.reset_index(drop=True)
            leg_df["leg_index"] = i
            leg_df["idxmax"] = idxmax
            combined.append(leg_df)
        combined = pd.concat(combined, axis=0)
        combined = combined[combined["leg_index"] == combined["idxmax"]]
        combined = combined.set_axis(combined_index).drop(
            columns=["idxmax", "leg_index"]
        )
        return pd.DataFrame(combined)

    @classmethod
    def _matching_approximate(
        cls,
        hedge_orders: list[pd.DataFrame],
        active_leg_fills: pd.DataFrame,
        latency_fn,
        latency_name: str,
        parent_id: str,
        weights: list[int],
    ):
        """Computes an approximation of the hedge latency when the number of fills for the active leg is different than the number of fills for the passive leg.

        Parameters
        ----------
        hedge_orders : list[pd.DataFrame]
            DataFrames with fills for each hedge leg
        active_leg_fills : pd.DataFrame with fills for the active leg
            DataFrame
        latency_fn : (str, pd.DataFrame, pd.DataFrame) -> pd.DataFrame
            Function that take latency description, active legs dataframe and hedge legs dataframe to compute latency
        latency_name : str
            Type of latency we're computing
        parent_id : str
            Order ID of the parent spread

        Returns
        -------
        list[pd.DataFrame]
            List with latency dataframes for each hedge leg
        """
        hedge_legs_critical = [
            [] for _ in hedge_orders
        ]  # for each hedge leg, indices we use for latency computation
        active_legs_critical = (
            []
        )  # for the active leg, indices we use for latency computation
        active_legs_traded = 0
        gcd_weights = gcd(*weights)
        for row_idx, (_, active_row) in enumerate(active_leg_fills.iterrows()):
            active_legs_traded += int(active_row[ATColumns.FILL_SIZE])
            # I assume that as soon as the number of traded active legs crosses the active leg weight divided by the gcd of the weights (threshold), an hedge order is sent.
            if active_legs_traded >= abs(weights[0]) // gcd_weights:
                active_legs_traded %= abs(weights[0]) // gcd_weights
                active_legs_critical.append(row_idx)
                current_time = active_row[ATColumns.TIME_SENT]
                # iterate over hedge legs
                for i, hedge_df in enumerate(hedge_orders):
                    condition = hedge_df[ATColumns.TIME_SENT] >= current_time
                    idx = np.argmax(condition.values)
                    hedge_legs_critical[i].append(idx)

        active_leg_critical_rows = active_leg_fills.iloc[active_legs_critical]
        hedge_legs_critical_rows = []
        for i, indices in enumerate(hedge_legs_critical):
            leg_orders = hedge_orders[i].iloc[indices]
            hedge_legs_critical_rows.append(leg_orders)

        if cls._distant_orders(hedge_legs_critical_rows, active_leg_critical_rows):
            return None
        result = []
        # iterate over hedge legs and compute the latency function for each one
        for i, indices in enumerate(hedge_legs_critical):
            leg_orders = hedge_orders[i].iloc[indices]
            result.append(
                latency_fn(latency_name, active_leg_critical_rows, leg_orders)
            )
        return result

    @classmethod
    def _handle_unmatched_orders(
        cls,
        active_leg_fills,
        df_group,
        tt_parent_id,
        hedge_legs_new_orders,
        latency_fn,
        latency_name,
        legs,
    ):
        originator = active_leg_fills.iloc[0].loc[ATColumns.EMAIL]
        date = active_leg_fills.iloc[0].loc[ATColumns.EXCHANGE_TIME]
        if not df_group[df_group[ATColumns.TT_ORDER_ID] == tt_parent_id].empty:
            spread_id = (
                df_group[df_group[ATColumns.TT_ORDER_ID] == tt_parent_id]
                .iloc[0]
                .loc[ATColumns.INSTRUMENT]
            )
            weights = get_ase_weights(spread_id, originator, date)
            if weights is not None:
                leg_latencies = cls._matching_approximate(
                    hedge_legs_new_orders,
                    active_leg_fills,
                    latency_fn,
                    latency_name,
                    tt_parent_id,
                    weights,
                )
                return leg_latencies
            else:
                LOG.warn(
                    f"WARN: {tt_parent_id=} has unmatching active fills and hedge orders. Legs={legs}"
                )
                return None
        else:
            LOG.warn(
                f"WARN: {tt_parent_id=} has unmatching active fills and hedge orders. Legs={legs}"
            )
            return None

    @classmethod
    def _distant_orders(cls, hedge_legs, active_leg_fills, threshold):
        differences = [
            cls.to_millisec(
                pd.Series(
                    hedge_new_orders[ATColumns.TIME_SENT]
                    - active_leg_fills[ATColumns.TIME_SENT].values
                )
            )
            for hedge_new_orders in hedge_legs
        ]
        return any(any(diff < threshold) for diff in differences)

    @classmethod
    def hedge_latency_single(  # TODO: fix for exchange-traded calendar/flies
        cls,
        tt_parent_id: str,
        ase_name: str,
        latency_name: str,
        df_group: pd.DataFrame,
        latency_fn,
    ):  # Already grouped by tt_parent_id
        """
        Performs aggregation of DataFrame rows that have the same "tt_parent_id".
        Parameters
        ----------
        tt_parent_id : str
            Order id of the spread for which we want to compute the latencies.
        spread_id : str
            Name of the parent instrument
        latency_name : str
            Type of latency measure we are computing.
        df_group : pd.DataFrame
            Spread legs
        latency_fn : (str, pd.DataFrame, pd.DataFrame) -> pd.DataFrame
            Function that take latency description, active legs dataframe and hedge legs dataframe to compute latency


        Returns
        -------
        pd.DataFrame
            DataFrame with matched new orders and fills
        """
        # separate orders in TRADES and NEWS
        legs = df_group[ATColumns.INSTRUMENT].unique()
        fills = df_group[df_group[ATColumns.EXECUTION_TYPE] == ETypes.TRADE]
        new_orders = df_group[df_group[ATColumns.EXECUTION_TYPE] == ETypes.NEW]
        if fills.empty:  # Exclude groups without trades
            return None
        # assume the first traded instrument is the active leg
        active_leg = fills[ATColumns.INSTRUMENT].iloc[0]
        active_leg_fills = fills[fills[ATColumns.INSTRUMENT] == active_leg]
        # all the new orders that don't involve the active leg are hedge orders
        hedge_new_orders = new_orders[new_orders[ATColumns.INSTRUMENT] != active_leg]
        # for each hedge instrument we have a dataframe with the new orders of that instrument
        hedge_legs_new_orders = [
            group
            for _, group in hedge_new_orders.groupby(
                ATColumns.INSTRUMENT
            )  # the version with [ATColumns.INSTRUMENT] gives a warning in pandas
        ]
        if hedge_new_orders.empty:  # Exclude groups without hedge orders
            LOG.warn(f"{tt_parent_id=} does not have any hedge orders in period")
            return None
        # all the dataframes in the hedge_legs_new_orders list must have the same size as the dataframe with the active leg fills, i.e. the active leg
        # and each hedge leg must have been traded with the same number of new orders
        if any(
            len(hedge_new_orders) != len(active_leg_fills)
            for hedge_new_orders in hedge_legs_new_orders
        ):
            leg_latencies = cls._handle_unmatched_orders(
                active_leg_fills,
                df_group,
                tt_parent_id,
                hedge_legs_new_orders,
                latency_fn,
                latency_name,
                legs,
            )
            if leg_latencies == None:
                LOG.warn(
                    f"WARN: {tt_parent_id=} has unmatching active fills and hedge orders. Legs={legs}"
                )
                return None
        else:
            # if the times of the orders are too far apart we are doing a mismatch
            if cls._distant_orders(
                hedge_legs_new_orders, active_leg_fills, threshold=-10
            ):
                return None
            # creates a list with dataframes each one containing a measurement of latency (by the latency function)
            leg_latencies = [
                latency_fn(latency_name, ase_name, active_leg_fills, leg_orders)
                for leg_orders in hedge_legs_new_orders
            ]
        # concatenates the dataframes (on the index axis). In theory you can get a difference latency for each hedge leg
        return cls._combine_leg_latencies(leg_latencies, latency_name=latency_name)

    @classmethod
    def _hedge_latency(cls, df: pd.DataFrame, latency_fn, latency_name):
        """
        Parameters
        ----------
        df : pd.DataFrame
            AuditTrail data
        latency_fn : (str, pd.DataFrame, pd.DataFrame) -> pd.DataFrame
            Function measuring the latency (depends of course o the kind of latency being measured)
        latency_name : str
            The kind of latency being measured

        Returns
        -------
        pd.DataFrame
            Hedge latencies
        """
        hedge_latencies = df.groupby([ATColumns.TT_PARENT_ID]).apply(
            lambda df_group: cls.hedge_latency_single(
                tt_parent_id=df_group.name,
                ase_name=cls._get_ase_name_from_order_id(df, df_group.name),
                latency_name=latency_name,
                df_group=df_group,
                latency_fn=latency_fn,
            )
        )
        if hedge_latencies.empty:
            hedge_latencies[latency_name] = None

        hedge_latencies = hedge_latencies.reset_index(level=[0]).dropna(
            subset=latency_name
        )
        return hedge_latencies

    @classmethod
    def get_first_ase_child_order_of_period_with_trades(
        cls, df: pd.DataFrame, freq="1d"
    ) -> pd.DataFrame:
        """
        Gets the tt_order_id of the first trade of each autospreader in a given period.
        Then, it returns the children orders of that parent order.

        Parameters
        ----------
        df : str
            dataframe with audit trail data (both ase orders and children orders)
        freq: str
            The frequence to aggregate by. Default is one day

        Returns
        -------
        DataFrame
            The children orders.
        """

        ase_orders = cls.get_ase_rows(df)
        ase_trade_orders = ase_orders[ase_orders[ATColumns.EXEC_TYPE] == ETypes.TRADE]
        first_ase_trade_orders = ase_trade_orders.groupby(
            [pd.Grouper(freq=freq, label="left"), ATColumns.INSTRUMENT],
            observed=True,
        ).first()
        order_ids = first_ase_trade_orders[ATColumns.TT_ORDER_ID].unique()
        return df[df[ATColumns.TT_PARENT_ID].isin(order_ids)]

    @classmethod
    def hedge_latency(
        cls, df: pd.DataFrame, only_first_ase_order_of_day: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assumes that we quote the first leg and the other ones are hedged. Retrieves the hedge latency and the hedge latency including acks for the provided AuditTrail data in `df`.

        Parameters
        ----------
        df : pd.DataFrame
            AuditTrail data
        only_first_ase_order_of_day : bool, optional
            Whether to compute the latency only for the first autospreader trade of one day, by default False

        Returns
        -------
        tuple[ pd.DataFrame, pd.DataFrame ]
            Hedge latencies and hedge latencies with ack.
        """
        if only_first_ase_order_of_day:
            df = cls.get_first_ase_child_order_of_period_with_trades(df, "1d")
            if df.empty:
                LOG.warn(
                    "WARNING :: No autospreader orders in period (try and change filters, like exchange name)"
                )
                return None, None
        hedge_latencies = cls._hedge_latency(
            df, cls._hedge_latency_single_leg, "hedge_latency"
        )

        hedge_latencies_with_ack = cls._hedge_latency(
            df, cls._hedge_latency_with_ack_time_single_leg, "hedge_latency_with_ack"
        )
        return hedge_latencies, hedge_latencies_with_ack

    @classmethod
    def payup_latency_single_leg_order(
        cls,
        tt_order_id: str,
        latency_name: str,
        df_group: pd.DataFrame,
        valid_payup_orders=["REPLACED", "RESTATED"],
    ):  # Already grouped by tt_parent_id
        """
        Performs payup latency for single leg
        ----------
        tt_order_id : str
            Order id of the spread for which we want to compute the latencies.
        spread_id : str
            Name of the parent instrument
        latency_name : str
            Type of latency measure we are computing.
        df_group : pd.DataFrame
            Spread legs

        Returns
        -------
        pd.DataFrame
            DataFrame with matched new orders and fills
        """
        df_reindexed = df_group.reset_index()
        shifted = df_reindexed.shift(1)

        def row_wise(row, last_row):
            if not row[ATColumns.EXECUTION_TYPE] in valid_payup_orders:
                return None
            if not last_row[ATColumns.EXECUTION_TYPE] in valid_payup_orders + ["NEW"]:
                return None
            return (
                row[ATColumns.EXCHANGE_TIME] - last_row[ATColumns.EXCHANGE_TIME]
            ).total_seconds() * 1000

        payup_latencies = df_reindexed.iloc[1:].apply(
            lambda row: row_wise(row, shifted.loc[row.name]),
            axis=1,
        )
        payup_latencies.index = df_group.iloc[1:].index
        return pd.DataFrame(payup_latencies.dropna()).rename(
            columns={"0": latency_name, 0: latency_name}
        )

    @classmethod
    def payup_latency_single_parent_id(
        cls,
        tt_parent_id: str,
        latency_name: str,
        df_group: pd.DataFrame,
    ):  # Already grouped by tt_parent_id
        """
        Performs aggregation of DataFrame rows that have the same "tt_parent_id".
        Parameters
        ----------
        tt_parent_id : str
            Order id of the spread for which we want to compute the latencies.
        spread_id : str
            Name of the parent instrument
        latency_name : str
            Type of latency measure we are computing.
        df_group : pd.DataFrame
            Spread legs

        Returns
        -------
        pd.DataFrame
            DataFrame with matched new orders and fills
        """
        # separate orders in TRADES and NEWS
        fills = df_group[df_group[ATColumns.EXECUTION_TYPE] == ETypes.TRADE]
        if fills.empty:  # Exclude groups without trades
            return None
        # assume the first traded instrument is the active leg
        active_leg = fills[ATColumns.INSTRUMENT].iloc[0]
        hedging_legs = [
            leg for leg in df_group[ATColumns.INSTRUMENT].unique() if leg != active_leg
        ]
        hedging_leg_events = df_group[df_group[ATColumns.INSTRUMENT].isin(hedging_legs)]
        payup_latencies = hedging_leg_events.groupby([ATColumns.TT_ORDER_ID]).apply(
            lambda group_df: cls.payup_latency_single_leg_order(
                group_df.name, latency_name, group_df
            )
        )
        if payup_latencies.empty:
            return None
        return payup_latencies

    @classmethod
    def payup_latency(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assumes that we quote the first leg and the other ones are hedged.
        Retrieves the payup latency as the distance between payups in the hedging legs

        Parameters
        ----------
        df : pd.DataFrame
            AuditTrail data

        Returns
        -------
        tuple[ pd.DataFrame, pd.DataFrame ]
            Hedge latencies and hedge latencies with ack.
        """
        payup_latencies = df.groupby([ATColumns.TT_PARENT_ID]).apply(
            lambda df_group: cls.payup_latency_single_parent_id(
                tt_parent_id=df_group.name,
                latency_name="payup_latency",
                df_group=df_group,
            )
        )
        if payup_latencies.empty:
            payup_latencies["payup_latency"] = None

        payup_latencies = payup_latencies.reset_index(level=[0]).dropna(
            subset="payup_latency"
        )
        return payup_latencies
