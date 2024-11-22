# Import libraries
from decimal import Decimal
from numbers import Number
import numpy as np
import pandas as pd

from analysis.util_funcs import (
    load_dict_to_normal_name,
    reference_price_plus_cross_bid_asks,
    filter_dates,
    _range_of_data_where_price,
)
from math import ceil
from beautifulData.functions.trading import fading_price
import datetime as dt
from analysis.util_funcs import get_legs_data
from beautifulData.spreads import Spread
import logging

LOG = logging.getLogger(__name__)


class Opportunity:
    """
    Statistically analyses ONE fading opportunity during ONE event.

    Attributes:
        params (dict): Configuration parameters for the analysis.
            Required keys:
                - "input" (dict): Input configuration parameters.
                    Required keys:
                        - "start_event" (str): Start time of the analysis event.
                        - "end_event" (str): End time of the analysis event.
                        - "fading_starting_point" (int): Starting point for the fading price.
                        - "fraction_of_event_for_new_position" (float): Fraction of event where you can enter a position.
                - "output" (dict): Output configuration parameters.
                - "opportunity" (dict): Opportunity-specific parameters.
                - "spread_threshold"
        event (DataFrame): Event-related data.
        spread (Spread): Spread-related data.

    Main method:
        find_opportunities(): Calculates fading opportunity metrics.

    Returns:
        results (pd.Series): Calculated trading metrics.
    """

    # Initialize the Opportunity object with provided parameters

    opportunity_type = "base"

    def __init__(self, params, event, spread_def, spread: Spread):
        self.params = params
        self.spread = spread
        self.event = event
        self.spread_name = spread.arfima_name
        self.spread_def = spread_def
        self.spread_data = spread.quotes
        self.weights_spread = {
            leg: w for leg, w in zip(spread.legs, spread.weights_spread)
        }
        self.weights_price = {
            leg: w for leg, w in zip(spread.legs, spread.weights_price)
        }
        self._last_timestamp = self.spread_data.index[-1]
        self.legs_name = self.spread.legs
        self.legs_data = get_legs_data(self.spread, self.legs_name)
        if hasattr(self.spread, "trades_data") and self.spread.trades_data is not None:
            self.legs_trades = {
                load_dict_to_normal_name(key): data.df
                for key, data in spread.trades_data.items()
            }
        self.strategies_results = {}
        self.metrics_to_include = [
            "max_dislocation_nc",
            "time_to_max_dislocation_ms",
            "retracement_after_dislocation_nc",
            "retracement_ratio",
            "reference_to_final_price_abs_nc",
            "min_bid_ask_spread",
        ]

    def _debug_plot(self):
        to_plot = pd.concat(
            [
                self.fadingprice,
                self.spread_data[["bid_price", "ask_price"]],
            ],
            axis=1,
        )
        # to_plot["mid_ref_price"] = self.mid_referenceprice
        to_plot.plot().show()

    def _find_opportunities(self):
        """Function that each subclass has to define in order to compute its particular metrics"""
        pass

    # Main method to calculate various statistics and populate attributes
    def find_opportunities(self):
        params = self.params["opportunity"]

        # Calculate the time limit index for new position
        self.timelimit = self._timelimit()

        # Calculate the reference price based on bid and ask prices
        (
            self.bid_referenceprice,
            self.ask_referenceprice,
            self.mid_referenceprice,
        ) = self._reference_price(self.spread_data)

        # Calculate the fading price based on bid and ask prices
        self.fadingprice, reference_point = self._fading_price(
            self.spread_data, self.mid_referenceprice
        )

        (
            self.bid_finalprice,
            self.ask_finalprice,
            self.mid_finalprice,
        ) = self._final_referenceprice()

        (
            self.max_dislocation_time,
            self.max_dislocation_nc,
            self.max_dislocation_price,
            self.max_dislocation_side,
        ) = self._max_dislocation(reference_point, self.fadingprice)

        self.time_to_max_dislocation_s = self._time_to_entry(self.max_dislocation_time)

        self.reference_to_final_price_abs_nc = self._drift()
        (
            self.retracement_after_dislocation_time,
            self.retracement_after_dislocation_nc,
            self.retracement_after_dislocation_price,
        ) = self._max_retracement_after_max_dislocation(
            "buy" if self.max_dislocation_nc < 0 else "sell"
        )

        (
            self.max_bid_ask_spread,
            self.mean_bid_ask_spread,
            self.min_bid_ask_spread,
        ) = self._bid_ask_spread(self.spread_data)
        self.retracement_ratio = self._retracement_ratio()
        self._find_opportunities()
        self.max_dislocation_nc = abs(self.max_dislocation_nc)
        self.leading_leg_idx, self.leading_leg = self._determine_leading_leg()
        if self.leading_leg_idx is not None and self.leading_leg is not None:
            # Create a self.leg attribute for each leg name
            for i, leg in enumerate(self.legs_name):
                setattr(self, f"leg_{i+1}_is_leading", 0)
                self.metrics_to_include.append(f"leg_{i+1}_is_leading")

            # Update the attribute with the leading leg to 1.
            setattr(self, f"leg_{self.leading_leg_idx}_is_leading", 1)
        else:
            for i, leg in enumerate(self.legs_name):
                setattr(self, f"leg_{i+1}_is_leading", None)
                self.metrics_to_include.append(f"leg_{i+1}_is_leading")

        self.stats_data = self._get_results_series()
        return self.stats_data

    def _reference_price(self, instrument_data):
        """Calls aux function to compute reference price"""
        params = self.params["opportunity"]
        reference_time_from = (
            self.event["from_date"]
            - params["reference_price_time_before"]
            - params["reference_price_window"]
        )
        reference_time_to = (
            self.event["from_date"] - params["reference_price_time_before"]
        )
        return reference_price_plus_cross_bid_asks(
            bid_prices=instrument_data["bid_price"],
            ask_prices=instrument_data["ask_price"],
            from_date=reference_time_from,
            to_date=reference_time_to,
            spread_threshold_percentile=Decimal(
                params["reference_price_spread_threshold_percentile"]
            ),
        )

    def _fading_price(self, instrument_data, mid_referenceprice):
        """Calls aux function to compute fading price"""
        fading_price_starting_point = self.params["opportunity"][
            "fading_price_starting_point"
        ]
        reference_point = mid_referenceprice
        if fading_price_starting_point == "bid":
            reference_point = instrument_data["bid_price"].dropna().iloc[0]
        elif fading_price_starting_point == "ask":
            reference_point = instrument_data["ask_price"].dropna().iloc[0]
        elif fading_price_starting_point == "mid_price":
            reference_point = (
                ((instrument_data["bid_price"] + instrument_data["ask_price"]) / 2)
                .dropna()
                .iloc[0]
            )
        return (
            fading_price(
                bid_price=instrument_data["bid_price"],
                ask_price=instrument_data["ask_price"],
                start_point=reference_point,
            ),
            reference_point,
        )

    def _bid_ask_spread(self, instrument_data):
        filter_from = (
            self.event["from_date"]
            - self.params["opportunity"]["reference_price_time_before"]
        )
        filter_to = self.timelimit
        bid = filter_dates(instrument_data["bid_price"], filter_from, filter_to)
        ask = filter_dates(instrument_data["ask_price"], filter_from, filter_to)
        bid_ask_spread = ask - bid
        return bid_ask_spread.max(), bid_ask_spread.mean(), bid_ask_spread.min()

    def _timelimit(self):
        """
        Establishes the time limit for new trades
        """
        timelimit_duration = self.params["opportunity"][
            "window_of_event_for_new_positions"
        ]
        return timelimit_duration + self.event["from_date"]

    # Corresponds to path_statistics 1.
    def _final_referenceprice(self):
        """Calculates the final reference price of a spread"""
        params = self.params["opportunity"]
        reference_time_from = self._last_timestamp - params["final_price_window"]
        reference_time_to = self._last_timestamp
        # To discuss
        return reference_price_plus_cross_bid_asks(
            bid_prices=self.spread_data["bid_price"],
            ask_prices=self.spread_data["ask_price"],
            from_date=reference_time_from,
            to_date=reference_time_to,
            spread_threshold_percentile=params[
                "reference_price_spread_threshold_percentile"
            ],
        )

    def _drift(self):
        return abs(self.mid_finalprice - self.mid_referenceprice)

    def _entry_exit_metrics(
        self,
        entry_time: dt.datetime,
        exit_time: dt.datetime,
        entry_side="buy",
        only_entry_liquidity=False,
        entry_price_threshold=None,
        exit_price_threshold=None,
        adjacent_interval=False,
    ):
        """Computes:
        - Dislocation nc
        - Worst retracement_nc
        - worst_retracement_time
        - Liquidez/Volumen
        - Duración de la oportunidad
        - Holding time
        - pnl
        - Worst pnl after entry
        - Time to entry
        """

        entry_prices, exit_prices = (
            self.spread_data["ask_price"],
            self.spread_data["bid_price"],
        )
        if entry_side == "sell":
            entry_prices, exit_prices = exit_prices, entry_prices
        entry_price = entry_prices.loc[entry_time:].iloc[0]
        exit_price = exit_prices.loc[exit_time:].iloc[0]
        worst_retracement_time, worst_retracement_nc = self._worst_open_retracement(
            entry_time, entry_price, exit_time, self.fadingprice, entry_side
        )
        if entry_price_threshold is not None:
            entry_price = entry_price_threshold
        if exit_price_threshold is not None:
            exit_price = exit_price_threshold
        exit_entry_nc = exit_price - entry_price
        if entry_side == "sell":
            exit_entry_nc = -exit_entry_nc
        liquidity = self._liquidity(
            entry_time,
            exit_time,
            entry_side,
            "max",
            only_entry=only_entry_liquidity,
            entry_price_threshold=entry_price_threshold,
            exit_price_threshold=exit_price_threshold,
            adjacent_interval=adjacent_interval,
        )
        entry_duration = self._entry_duration(
            entry_time,
            exit_time,
            entry_side,
            price_threshold=entry_price_threshold,
        )

        holding_time = self._holding_mseconds(entry_time, exit_time)
        pnl = self._pnl(exit_entry_nc, liquidity)
        worst_open_pnl = self._pnl(worst_retracement_nc, liquidity)
        time_to_entry = self._time_to_entry(entry_time)
        return (
            exit_entry_nc,
            worst_retracement_nc,
            worst_retracement_time,
            liquidity,
            entry_duration,
            holding_time,
            pnl,
            worst_open_pnl,
            time_to_entry,
        )

    # Corresponds to path_statistics 2.
    def _max_dislocation(self, reference_point, fadingprice):
        """Calculates the maximum dislocation in the fading price before timelimit has been reached"""
        fadingprice_net_change = (
            filter_dates(fadingprice, self.event["from_date"], self.timelimit)
            - reference_point
        )
        max_dislocation_time = abs(
            fadingprice_net_change.apply(pd.to_numeric, downcast="float")
        ).idxmax()
        max_dislocation_nc = fadingprice_net_change.loc[max_dislocation_time]
        max_dislocation_price = fadingprice.loc[max_dislocation_time]

        if len(fadingprice_net_change.unique()) <= 1:
            return max_dislocation_time, 0, None, None

        # max_dislocation side needed for the plots. Could also be a useful metric.
        if max_dislocation_nc >= 0:
            side = "bid"
        elif max_dislocation_nc < 0:
            side = "ask"
        else:
            side = None
        return (max_dislocation_time, max_dislocation_nc, max_dislocation_price, side)

    def _opposite_dislocation(
        self,
        reference_point,
        fadingprice,
        max_dislocation_time,
        max_dislocation_side,
        delta,
    ):
        """Calculates the maximum dislocation in the oppposite side of the max dislocation, before the max dislocation"""
        filter_from = (
            self.event["from_date"]
            - self.params["opportunity"]["reference_price_time_before"]
        )
        fadingprice_net_change = (
            filter_dates(
                fadingprice,
                filter_from,
                min(self.timelimit, max_dislocation_time),
            )
            - reference_point
        ).apply(pd.to_numeric, downcast="float")
        if max_dislocation_side == "bid":  # positive max dislocation
            fadingprice_net_change_filtered = fadingprice_net_change[
                fadingprice_net_change < -delta
            ] + float(delta)
        elif max_dislocation_side == "ask":  # negative max dislocation
            fadingprice_net_change_filtered = fadingprice_net_change[
                fadingprice_net_change > delta
            ] - float(delta)
        dislocation_time = (
            abs(fadingprice_net_change_filtered).idxmax()
            if not fadingprice_net_change_filtered.empty
            else None
        )
        dislocation_nc = (
            fadingprice_net_change_filtered.loc[dislocation_time]
            if dislocation_time is not None
            else 0
        )
        dislocation_duration = (
            (
                (fadingprice_net_change.loc[dislocation_time:] <= 0).idxmax()
                if dislocation_nc > 0
                else (fadingprice_net_change.loc[dislocation_time:] >= 0).idxmax()
            )
            - dislocation_time
            if dislocation_time is not None
            else dt.timedelta(seconds=0)
        )

        if len(fadingprice_net_change_filtered.unique()) <= 1:
            return dislocation_time, 0, dt.timedelta(seconds=0)

        return (dislocation_time, abs(dislocation_nc), dislocation_duration)

    def _max_retracement_after_max_dislocation(self, entry_side: str, best=True):
        """Maximum retracement from the maximum dislocation. In price"""
        retracement = (
            self.fadingprice[self.fadingprice.index > self.max_dislocation_time]
            - self.fadingprice.loc[self.max_dislocation_time]
        )
        if entry_side == "sell":
            retracement = -retracement
        float_retr = retracement.apply(pd.to_numeric, downcast="float")
        if float_retr.empty:
            return None, None, None
        if best:
            max_drift_time = float_retr.idxmax()
        else:
            max_drift_time = float_retr.idxmin()
        max_drift = retracement.loc[max_drift_time]
        max_drift_price = self.fadingprice.loc[max_drift_time]
        return max_drift_time, max_drift, max_drift_price

    def _worst_open_retracement(
        self,
        entry_time: dt.datetime,
        entry_price: Number,
        exit_time: dt.datetime,
        exit_prices: pd.Series,
        entry_side: str,
    ):
        """Worst retracement. Assumes in the worst case scenario, you are exiting at exit_prices"""
        exit_prices = filter_dates(exit_prices, entry_time, exit_time)
        float_exit_prices = exit_prices.apply(pd.to_numeric, downcast="float")
        if entry_side == "buy":
            # We are exiting by selling at the bid. Lower is worse
            worst_open_retracement_time = float_exit_prices.idxmin()
        else:
            worst_open_retracement_time = float_exit_prices.idxmax()
        worst_open_retracement_nc = abs(
            entry_price - exit_prices.loc[worst_open_retracement_time]
        )
        return worst_open_retracement_time, worst_open_retracement_nc

    def _holding_mseconds(self, entry_time: dt.datetime, exit_time: dt.datetime):
        return (exit_time - entry_time).total_seconds() * 1000

    # Corresponds to path_statistics 6 (to change min and max).
    def _pnl(self, exit_entry_nc, liquidity=1):
        """Maximum possible P&L, from the fading opportunity"""
        pnl = exit_entry_nc * self.spread_def["DV01"] * liquidity
        return pnl

    def _time_to_entry(self, entry_time):
        """Computes the time elapsed from start of event to timelimit"""
        time_to_entry = entry_time - self.event["from_date"]
        return time_to_entry.total_seconds()

    def _relative_max_dislocation_aposteriori(self):
        """Calculates the relative maximum dislocation after fading has occured."""
        return np.abs(
            self.fadingprice.loc[self.max_dislocation_time] / self.mid_finalprice
        )

    def _retracement_ratio(self):
        if self.max_dislocation_nc is None or self.retracement_after_dislocation_nc is None:
            return None
        if self.max_dislocation_nc == 0:
            return 0
        return abs(self.retracement_after_dislocation_nc / self.max_dislocation_nc)

    def _liquidity(
        self,
        entry_time,
        exit_time,
        side,
        max_or_min="max",
        only_entry=False,
        entry_price_threshold=None,
        exit_price_threshold=None,
        adjacent_interval=False,
    ):
        """
        Outputs max (or min) volume taking into account that you enter the market at entry_time
        and exit at exit_time. Entry side should be in ["buy", "sell]
        """
        bid = self.spread_data[["bid_price", "bid_size"]].copy()
        bid = bid.rename(columns={"bid_price": "price", "bid_size": "size"})
        ask = self.spread_data[["ask_price", "ask_size"]].copy()
        ask = ask.rename(columns={"ask_price": "price", "ask_size": "size"})

        if side == "sell":
            bid, ask = ask, bid

        does_not_decrease = entry_price_threshold is not None and side == "sell"
        does_not_increase = entry_price_threshold is not None and side == "buy"
        ask_for_entry = _range_of_data_where_price(
            entry_time,
            ask,
            "price",
            does_not_decrease=does_not_decrease,
            does_not_increase=does_not_increase,
            price_threshold=entry_price_threshold,
            adjacent_interval=adjacent_interval,
        )
        does_not_decrease = exit_price_threshold is not None and side == "buy"
        does_not_increase = exit_price_threshold is not None and side == "sell"
        bid_for_exit = _range_of_data_where_price(
            exit_time,
            bid,
            "price",
            does_not_decrease=does_not_decrease,
            does_not_increase=does_not_increase,
            price_threshold=exit_price_threshold,
            adjacent_interval=adjacent_interval,
        )
        if max_or_min == "max":
            entry_liquidity = max(ask_for_entry["size"])
            exit_liquidity = max(bid_for_exit["size"])
        elif max_or_min == "min":
            entry_liquidity = min(ask_for_entry["size"])
            exit_liquidity = min(bid_for_exit["size"])
        if only_entry:
            return entry_liquidity
        return min(entry_liquidity, exit_liquidity)

    def _entry_duration(
        self,
        entry_time,
        exit_time,
        entry_side,
        price_threshold=None,
    ):
        """
        Calculates the duration of a general opportunity. If include_spread_partials is True,
        it considers that the level does not change even when there are only partial sizes
        """
        price_col = "bid_price" if entry_side == "sell" else "ask_price"
        # Get range of constant data and compute its time length
        does_not_decrease = price_threshold is not None and entry_side == "sell"
        does_not_increase = price_threshold is not None and entry_side == "buy"
        constant_data = _range_of_data_where_price(
            entry_time,
            self.spread_data,
            price_col,
            does_not_decrease=does_not_decrease,
            does_not_increase=does_not_increase,
            price_threshold=price_threshold,
            adjacent_interval=True,
        )
        # If we have every quote, duration is the time passed until next quote where price changes
        if self.params["input"].get("include_spread_partials", False):
            next_quote_idx = (
                self.spread_data[price_col].index.get_loc(constant_data.index[-1]) + 1
            )
            duration = self.spread_data[price_col].index[next_quote_idx] - entry_time
        # Compute its length if it has any.
        else:
            duration = constant_data.index[-1] - constant_data.index[0]

        return duration.total_seconds() * 1000

    def _determine_leading_leg(self):
        """
        Determines the leading leg by analyzing which leg has contributed more to the max dislocation.
        The output consists of: first, the index of the leading leg; second, the name of the leading leg;
        and third, the weighted max dislocation.

        We store as an attribute the index of the leading_leg.
        """

        leading_leg = None
        leading_leg_index = None
        max_weighted_dislocation = 0
        side_mapping = {"bid": "ask", "ask": "bid"}
        params = self.params["opportunity"]
        if self.max_dislocation_side is None or self.max_dislocation_nc == 0:
            return -1, None

        # Iterate through each leg of the spread.
        for idx, (leg_name, leg_df) in enumerate(self.legs_data.items()):
            side = self.max_dislocation_side

            # If weights are negative, we consider the alternate side where dislocation occurs.
            weight = self.weights_price[leg_name]
            if weight < 0:
                side = side_mapping.get(side, side)

            # Calculate the reference price as the median of prices within the specified interval.
            reference_time_from = (
                self.event["from_date"]
                - params["reference_price_time_before"]
                - params["reference_price_window"]
            )
            reference_time_to = (
                self.event["from_date"] - params["reference_price_time_before"]
            )
            interval_for_reference_price = leg_df[
                (leg_df.index >= reference_time_from)
                & (leg_df.index <= reference_time_to)
            ][f"{side}_price"].dropna()
            try:
                reference_price = np.median(interval_for_reference_price)
            except:
                pass

            # Calculate the side's price at the dislocation time.
            prices_at_dislocation = leg_df[f"{side}_price"][
                leg_df.index <= self.max_dislocation_time
            ]
            leg_at_dislocation = prices_at_dislocation[-1]

            # Calculate the weighted dislocation.
            weighted_dislocation = np.abs(
                weight * (reference_price - leg_at_dislocation)
            )

            # Update values accordingly.
            if weighted_dislocation > max_weighted_dislocation:
                max_weighted_dislocation = weighted_dislocation
                leading_leg = leg_name
                leading_leg_index = idx + 1

        return leading_leg_index, leading_leg

    def _get_results_series(self):
        results = {}
        for metric in self.metrics_to_include:
            if hasattr(self, metric):
                results[metric] = getattr(self, metric)
            else:
                LOG.error(
                    f"Error: metric {metric} does not exist for {self.spread_name} during {self.event['EventName']}"
                )
        for strategy_name, strategy in self.strategies_results.items():
            results.update(strategy.export_to_dict())
        results = pd.Series(results)
        return results
