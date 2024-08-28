from decimal import Decimal
from .metrics import Opportunity
from .metric import Metric
from analysis.util_funcs import filter_dates
from beautifulData.spreads import Spread
import datetime as dt
import numpy as np
from .lead_lag.lead_lag import LeadLag
import pandas as pd
from .hasbrouck import InfoShares


class LeadLagOpportunity(Opportunity):
    """
    Opportunity to manage metrics related to the fading project:
        - Best fading opportunity
        - Particular metrics related to net change strategies
            i.e enter the market above/below a specified net change
    """

    opportunity_type = "lead_lag"

    def __init__(self, params, event, spread_def, spread: Spread):
        super().__init__(params, event, spread_def, spread)
        self.metrics_to_include = [
            "max_dislocation_nc",
            "time_to_max_dislocation_s",
            "retracement_ratio",
            "reference_to_final_price_abs_nc",
            # "min_bid_ask_spread",
        ]

    def _mid(self, df):
        return Decimal(0.5) * (df["bid_price"] + df["ask_price"])

    def _find_opportunities(self):
        """Computes the lead lag
        - Max dislocation during initial minutes of the event
        - Retracement after that max dislocation
        """
        self.legs_start_event = {}
        self.max_dislocation_nc = abs(self.max_dislocation_nc)
        use_prod_names = len(set([leg[:-3] for leg in self.legs_name])) == len(
            self.legs_name
        )
        legs_statistics = {}
        for leg_name, leg_data in self.legs_data.items():
            leg_trades = self.legs_trades[leg_name]
            legs_statistics[leg_name] = self._leg_statistics(
                leg_name, leg_data, leg_trades
            )
            for stat_name, stat in legs_statistics[leg_name].items():
                prefix = leg_name[:-3] if use_prod_names else leg_name
                setattr(self, prefix + "_" + stat_name, stat)
                self.metrics_to_include.append(prefix + "_" + stat_name)
        if len(self.legs_name) == 2:
            leg1_quotes = self.legs_data[self.legs_name[0]]
            leg2_quotes = self.legs_data[self.legs_name[1]]
            self.HY_lead_lag, self.HY_llr, self.HY_llr2 = self._Hayashi_Yoshida(
                self.legs_trades[self.legs_name[0]]["trade_price"].copy(),
                self.legs_trades[self.legs_name[1]]["trade_price"].copy(),
            )
            dislocation1 = legs_statistics[self.legs_name[0]]["max_dislocation_nc"]
            dislocation2 = legs_statistics[self.legs_name[1]]["max_dislocation_nc"]
            beta = None if dislocation2 == 0 else dislocation1 / dislocation2
            beta_metric_name = f"beta_{self.legs_name[0][:-3] if use_prod_names else self.legs_name[0]}_{self.legs_name[1][:-3] if use_prod_names else self.legs_name[1]}"
            setattr(self, beta_metric_name, beta)
            time_between_max_dislocations_s = (
                legs_statistics[self.legs_name[1]]["time_to_max_dislocation_s"]
                - legs_statistics[self.legs_name[0]]["time_to_max_dislocation_s"]
            )
            time_between_metric_name = f"time_between_max_dislocations_s"
            setattr(self, time_between_metric_name, time_between_max_dislocations_s)
            start_metrics = [
                m for m in self.metrics_to_include if m.endswith("_arfima_start_event")
            ]
            event_start_leg1 = getattr(
                self,
                [m for m in start_metrics if m.startswith(self.legs_name[0][:2])][0],
            )

            event_start_leg2 = getattr(
                self,
                [m for m in start_metrics if m.startswith(self.legs_name[1][:2])][0],
            )
            self.arfima_leadlag_ms = (
                event_start_leg2 - event_start_leg1
            ).total_seconds() * 1000
            hasbrouck_leg1, hasbrouck_leg2 = self._compute_Hasbrouck_statistic(
                leg1_quotes, leg2_quotes
            )
            hasbrouck_name_leg1 = "hasbrouck_" + (
                self.legs_name[0][:-3] if use_prod_names else self.legs_name[0]
            )
            hasbrouck_name_leg2 = "hasbrouck_" + (
                self.legs_name[1][:-3] if use_prod_names else self.legs_name[1]
            )
            setattr(self, hasbrouck_name_leg1, hasbrouck_leg1)
            setattr(self, hasbrouck_name_leg2, hasbrouck_leg2)
            self.metrics_to_include.extend(
                [
                    "HY_lead_lag",
                    "HY_llr",
                    "HY_llr2",
                    "arfima_leadlag_ms",
                    beta_metric_name,
                    time_between_metric_name,
                    hasbrouck_name_leg1,
                    hasbrouck_name_leg2,
                ]
            )

    def _compute_Hasbrouck_statistic(self, leg1_quotes, leg2_quotes):
        # We want to create a datafame with the bid price column of the leg1_trades, bid column of leg_2.
        leg1_quotes_bid = leg1_quotes["bid_price"].to_frame()
        leg2_quotes_bid = leg2_quotes["bid_price"].to_frame()
        HY_window_after_event = self.params["opportunity"].get(
            "leadlag_window_after_event", dt.timedelta(minutes=1)
        )
        leg1_quotes_bid = filter_dates(
            leg1_quotes_bid,
            self.event["from_date"],
            self.event["from_date"] + HY_window_after_event,
        )
        leg2_quotes_bid = filter_dates(
            leg2_quotes_bid,
            self.event["from_date"],
            self.event["from_date"] + HY_window_after_event,
        )
        merged_df = pd.merge(
            leg1_quotes_bid,
            leg2_quotes_bid,
            how="outer",
            left_index=True,
            right_index=True,
        )
        synchronized_df = merged_df.ffill()
        synchronized_df = (
            synchronized_df.bfill()
        )  # First NAN values filled with the first value of the series.
        synchronized_df = synchronized_df.reset_index(
            drop=True
        )  # Reseting index for IS.
        # We change the type of the dataframe to float32 to avoid any type errors
        synchronized_df = synchronized_df.astype("float32")
        info_shares = InfoShares(
            synchronized_df, k_ar_diff=1
        )  # We assume there exists cointegration relationship.
        results_1 = info_shares.fit()
        # synchronized_df_2 = synchronized_df[["bid_price_y", "bid_price_x"]]
        # info_shares_2 = InfoShares(synchronized_df_2, k_ar_diff=1)
        # results_2 = info_shares_2.fit()

        return np.asarray(results_1.infoShares)[0]  # , results_2.infoShares

    def _leg_statistics(self, leg_name, leg_data, leg_trades) -> dt.datetime:
        (
            bid_referenceprice,
            ask_referenceprice,
            mid_referenceprice,
        ) = self._reference_price(leg_data)
        delta = ask_referenceprice - mid_referenceprice
        refprice = mid_referenceprice
        fadingprice, reference_point = self._fading_price(
            leg_data,
            ((leg_data["bid_price"] + leg_data["ask_price"]) / 2).dropna().iloc[0],
        )
        fadingprice_for_max_dislocation = fadingprice[
            fadingprice.index >= self.event["from_date"] + dt.timedelta(seconds=5)
        ]
        (
            max_dislocation_time,
            max_dislocation_nc,
            max_dislocation_price,
            max_dislocation_side,
        ) = self._max_dislocation(mid_referenceprice, fadingprice_for_max_dislocation)

        max_dislocation_nc = abs(max_dislocation_nc)
        max_dislocation_time_s = self._time_to_entry(max_dislocation_time)
        opposite_dislocation_time, opposite_dislocation_nc, dislocation_duration = (
            self._opposite_dislocation(
                mid_referenceprice,
                fadingprice,
                max_dislocation_time,
                max_dislocation_side,
                delta,
            )
        )
        betrayal_nc = opposite_dislocation_nc
        betrayal_ratio = (
            opposite_dislocation_nc / float(max_dislocation_nc)
            if max_dislocation_nc > 0
            else None
        )
        is_there_betrayal = (
            dislocation_duration < dt.timedelta(seconds=1) and betrayal_ratio > 0.05
            if betrayal_ratio is not None
            else None
        )
        liquidity_metrics = self._liquidity_metrics(leg_data, leg_trades)

        legs_start_event = pd.to_datetime(self._arfima_start_of_event(leg_trades))
        return {
            # "reference_price": mid_referenceprice,
            # "fadingprice": fadingprice,
            "betrayal_ratio": betrayal_ratio,
            "betrayal_nc": betrayal_nc,
            "is_there_betrayal": is_there_betrayal,
            "time_to_max_dislocation_s": max_dislocation_time_s,
            "max_dislocation_nc": max_dislocation_nc,
            "arfima_start_event": legs_start_event,
            **liquidity_metrics,
        }

    def _arfima_start_of_event(self, leg_trades):
        if leg_trades is None or leg_trades.empty:
            return None
        prod = leg_trades["instrument"].iloc[0][:-3]
        filter_from = self.event["from_date"]
        filter_to = self.event["from_date"] + self.params["opportunity"].get(
            "leadlag_window_after_event",
            self.params["opportunity"]["window_of_event_for_new_positions"],
        )
        p = self.params["opportunity"].get("leadlag_volume_percentile", 0.80)
        if isinstance(p, dict):
            p = {con: perc for perc, contrs in p.items() for con in contrs}
            p = p.get(prod, 0.8)
        leg_trades = filter_dates(leg_trades, filter_from, filter_to)
        resampled = leg_trades["trade_size"].resample("1ms").sum()
        leg_trades = resampled[resampled > 0]
        resampled_vol_median = leg_trades.apply(float).quantile(p)

        # cum_traded_vol = leg_trades.apply(float).cumsum()
        # vol_percentile = float(leg_trades.sum()) * p
        cut_point = leg_trades[leg_trades >= resampled_vol_median]
        cut_point = cut_point.index[0] if not cut_point.empty else None
        # p_time = cum_traded_vol[cum_traded_vol <= vol_percentile].index[-1]
        return cut_point

    def _liquidity_metrics(self, leg_data, leg_trades):
        if leg_trades.empty or leg_data.empty:
            return {
                "mean_duration_between_trades": None,
                "median_duration_between_trades": None,
                "mean_bid_ask_spread": None,
                "trades_wipeout_level_perc": None,
            }
        filter_from = self.event["from_date"]
        filter_to = self.timelimit
        leg_data = filter_dates(leg_data, filter_from, filter_to)
        leg_trades = filter_dates(leg_trades, filter_from, filter_to)
        try:
            duration_between_trades = self._avg_duration_between_trades(leg_trades)
        except:
            pass
        mean_duration_between_trades = duration_between_trades.mean()
        median_duration_between_trades = duration_between_trades.median()
        (
            max_bid_ask_spread,
            mean_bid_ask_spread,
            min_bid_ask_spread,
        ) = self._bid_ask_spread(leg_data)
        trades_wipeout_level_perc = self._trades_wipe_out_level(leg_data, leg_trades)
        metrics = {
            "mean_duration_between_trades": mean_duration_between_trades,
            "median_duration_between_trades": median_duration_between_trades,
            "mean_bid_ask_spread": mean_bid_ask_spread,
            "trades_wipeout_level_perc": trades_wipeout_level_perc,
        }
        return {name + "|liquidity": metric for name, metric in metrics.items()}

    def _avg_duration_between_trades(self, instrument_trades: pd.DataFrame):
        return (
            (1000 * instrument_trades.index.diff().total_seconds()).dropna().to_series()
        )

    def _trade_wipe_out_level(self, leg_data: pd.DataFrame, trade):
        dtime = trade.name
        quote_idx = leg_data.index.searchsorted(dtime, side="left")

        if quote_idx == 0 or len(leg_data) == quote_idx:
            return False

        prev_quote = leg_data.iloc[quote_idx - 1]
        next_quote = leg_data.iloc[quote_idx]

        aggress = trade["aggressor"]
        if trade["aggressor"] not in ["buy", "sell"]:
            aggress = (
                "sell" if trade["trade_price"] <= prev_quote["bid_price"] else "buy"
            )
        if aggress == "buy":
            return prev_quote["ask_price"] < next_quote["ask_price"]
        else:
            return prev_quote["bid_price"] > next_quote["bid_price"]

    def _trades_wipe_out_level(self, leg_data: pd.DataFrame, leg_trades: pd.DataFrame):
        """Assumes trades and quotes are in synch (currently, this does not hold for Sonia)"""
        if leg_trades.empty or leg_data.empty:
            return 0

        num_trades = leg_trades.apply(
            lambda row: self._trade_wipe_out_level(leg_data, row), axis=1
        )
        return sum(num_trades) / len(leg_trades)

    def _Hayashi_Yoshida(self, prices_X, prices_Y, max_lag_s=2):
        HY_window_after_event = self.params["opportunity"].get(
            "leadlag_window_after_event", dt.timedelta(minutes=2)
        )
        prices_X = filter_dates(
            prices_X,
            self.event["from_date"],
            self.event["from_date"] + HY_window_after_event,
        )
        prices_Y = filter_dates(
            prices_Y,
            self.event["from_date"],
            self.event["from_date"] + HY_window_after_event,
        )
        prices_X.index = prices_X.index.map(lambda x: x.round("1ms"))
        prices_Y.index = prices_Y.index.map(lambda x: x.round("1ms"))
        ll = LeadLag(ts1=prices_X, ts2=prices_Y, max_lag=max_lag_s)
        print("Running inference...")
        ll.run_inference(5)
        return ll.lead_lag * 1000, ll.llr, ll.llr2
