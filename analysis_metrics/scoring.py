from decimal import Decimal
import pandas as pd
import copy


class ScoringMetric:
    def __init__(
        self,
        raw_opportunity_data: dict,
        by_spread: pd.DataFrame,
        params,
        spreads_def: pd.DataFrame,
        mode="by_spread_event",
    ):
        self.raw_opportunity_data = raw_opportunity_data
        self.by_spread = by_spread
        self.params = params
        self.spreads_def = spreads_def
        self.mode = mode

    def score(self):
        """
        Calculate and return a scoring metric based on the mode of operation.

        If the mode is set to "by_spread," the scoring is performed on a subset of the
        'by_spread' DataFrame, considering only columns with the level value "mean." The
        resulting DataFrame is sorted based on the 'final_score' column in descending order.

        If the mode is not "by_spread," the method processes the raw opportunity data,
        calculates scores, and generates a DataFrame with mean scores for each event. The
        DataFrame is sorted based on the 'final_score' column in descending order, and
        additional columns such as 'mean_' prefix and 'min_final_score' are added.

        Returns:
            pandas.DataFrame: A DataFrame containing the calculated scoring metric.

        Note:
            The specific scoring logic is implemented in the '_score' method, and behavior
            may vary based on the attributes of the class instance, such as 'mode',
            'by_spread', and 'raw_opportunity_data'.
        """
        if self.mode == "by_spread":
            only_mean = self.by_spread.loc[
                :, self.by_spread.columns.get_level_values(1) == "mean"
            ].droplevel(1, axis=1)
            scoring_metric_df = only_mean.apply(self._score, axis=1)
            scoring_metric_df = scoring_metric_df.sort_values(
                by="final_score", na_position="first", ascending=False
            )
        else:
            raw_metrics_with_name = copy.deepcopy(self.raw_opportunity_data)
            for spread, data in raw_metrics_with_name.items():
                for event, data2 in data.items():
                    data2["spread_name"] = spread
            df = pd.DataFrame(raw_metrics_with_name)
            raw_scores = df.stack().apply(self._score)
            scoring_metric_df = raw_scores.groupby(level=[1]).mean()
            scoring_metric_df = scoring_metric_df.sort_values(
                by="final_score", na_position="first", ascending=False
            )
            scoring_metric_df.columns = "mean_" + scoring_metric_df.columns
            min_scores = raw_scores.groupby(level=[1]).min()
            scoring_metric_df["min_final_score"] = min_scores["final_score"]
            cols = scoring_metric_df.columns.tolist()
            cols = cols[0:1] + cols[-1:] + cols[1:-1]
            scoring_metric_df = scoring_metric_df[cols]

        return scoring_metric_df


class ManualScorer(ScoringMetric):
    """The idea of the metric is to capture:
    - Reference price to final price net change. Closer to zero is better. Absolute value is penalized.
    - Max dislocation net change. Higher is better
    - Retracement ratio: higher than 0.8 is good
    - Duration: higher is better
    - Liquidity: higher than 1 is ideal"""

    def score_ref_price_to_final_price(self, spread):
        """
        Calculate a score for the reference price to final price metric of the provided spread.

        Parameters:
            spread (pd.Series): The spread data for which the score is calculated.

        Returns:
            float: A score between 0 and 1 based on the deviation of 'reference_to_final_price_abs_nc'
            from specified thresholds.

        Notes:
            The scoring is based on the absolute value of 'reference_to_final_price_abs_nc'.
            The score is determined by comparing this value to predefined thresholds. If the
            value is below the lower threshold, the score is 1. If it is above the upper threshold,
            the score is 0. For values between the thresholds, the score is calculated on a linear scale.

        See Also:
            - spreads_def
        """
        ref_price_to_final_price = abs(spread["reference_to_final_price_abs_nc"])
        col = "score_thresholds_refprice_to_finalprice"
        curr_spread_def = self.spreads_def[
            self.spreads_def["ArfimaName"] == spread["spread_name"]
        ].iloc[0]
        thresholds = (0.25, 2)
        if col in self.spreads_def.columns:
            thresholds = [float(x) for x in curr_spread_def[col].split(",")]
        if ref_price_to_final_price < thresholds[0]:
            return 1
        elif ref_price_to_final_price > thresholds[1]:
            return 0
        else:
            return 1 + (ref_price_to_final_price - thresholds[0]) * (
                -1 / (thresholds[1] - thresholds[0])
            )

    def score_max_dislocation_nc(self, spread):
        """
        Calculate a score for the maximum dislocation metric of the provided spread.

        Parameters:
            spread (pd.Series): The spread data for which the score is calculated.

        Returns:
            float: A score between 0 and 1 based on the deviation of 'max_dislocation_nc'
            from half of the bid-ask spread.

        Notes:
            The scoring is based on the absolute value of 'max_dislocation_nc'. The score is 0
            if 'max_dislocation_nc' is less than half of the bid-ask spread, 1 if it is greater
            than 3 times the bid-ask spread, and calculated on a linear scale for values in between.

        See Also:
            - bid_ask_spread
        """
        half_bid_ask = spread["min_bid_ask_spread"] / 2
        if pd.isnull(spread["max_dislocation_nc"]):
            return 0
        dislocation_nc = abs(spread["max_dislocation_nc"])
        if dislocation_nc < half_bid_ask:
            return 0
        elif dislocation_nc > 3 * half_bid_ask:
            return 1
        else:
            return (dislocation_nc - half_bid_ask) * (1 / (2 * half_bid_ask))

    def score_retracement_ratio(self, spread):
        """
        Calculate a score for the retracement ratio metric of the provided spread.

        Parameters:
            spread (pd.Series): The spread data for which the score is calculated.

        Returns:
            float: A score between 0 and 1 based on the 'retracement_ratio' value.

        Notes:
            The scoring is based on the 'retracement_ratio'. The score is 0 if the ratio is
            below 0.25, 1 if it is above 0.8, and calculated on a linear scale for values in between.

        """
        retracement_ratio = spread["retracement_ratio"]
        if pd.isnull(retracement_ratio) or retracement_ratio < 0.25:
            return 0
        elif retracement_ratio > 0.8:
            return 1
        else:
            return (retracement_ratio - 0.25) * (1 / 0.55)

    def score_duration(self, spread):
        """
        Calculate a score for the duration metric of the provided spread.

        Parameters:
            spread (pd.Series): The spread data for which the score is calculated.

        Returns:
            float: A score between 0 and 1 based on the 'entry_max_dislocation_duration_ms' value.

        Notes:
            The scoring is based on the absolute value of 'entry_max_dislocation_duration_ms'.
            The score is 0 if the duration is less than 5 milliseconds, 1 if it is greater
            than 40 milliseconds, and calculated on a linear scale for values in between.

        """
        duration = spread["entry_max_dislocation_duration_ms"]
        if pd.isnull(duration):
            return 0
        duration = abs(duration)
        if duration < 5:
            return 0
        elif duration > 40:
            return 1
        else:
            return (duration - 5) * (1 / 35)

    def execute_with_try_catch(self, func, spread):
        """
        Execute the provided function with try-catch error handling.

        Parameters:
            func (callable): The function to be executed.
            spread (pd.Series): The data to be passed to the function.

        Returns:
            float: The result of the function execution or 0 if an exception occurs.

        """
        try:
            return func(spread)
        except:
            return 0

    def _score(self, spread):
        """
        Calculate individual scores for specific metrics based on the provided spread.

        Parameters:
            spread (pd.Series): The spread data for which scores are calculated.

        Returns:
            pd.Series: A Series containing individual scores for each metric and a final
            overall score.

        Weights:
            - 'ref_price_to_final_price': 30%
            - 'max_dislocation_nc': 30%
            - 'retracement_ratio': 30%
            - 'duration': 10%

        Notes:
            The method uses the 'execute_with_try_catch' function to handle exceptions
            during score calculations. The weights assigned to each metric are defined in
            the 'weights' dictionary. The method returns a Series containing individual
            scores for each metric and an overall final score.

        See Also:
            - score_ref_price_to_final_price
            - score_max_dislocation_nc
            - score_retracement_ratio
            - score_duration
            - execute_with_try_catch
        """
        weights = {
            "ref_price_to_final_price": 0.3,
            "max_dislocation_nc": 0.3,
            "retracement_ratio": 0.3,
            "duration": 0.1,
        }
        scores = {}
        scores["ref_price_to_final_price"] = self.execute_with_try_catch(
            self.score_ref_price_to_final_price, spread
        )
        scores["max_dislocation_nc"] = self.execute_with_try_catch(
            self.score_max_dislocation_nc, spread
        )
        scores["retracement_ratio"] = self.execute_with_try_catch(
            self.score_retracement_ratio, spread
        )
        scores["duration"] = self.execute_with_try_catch(self.score_duration, spread)
        final_scores = {}
        final_scores["final_score"] = sum(
            Decimal(weights[k]) * Decimal(score) for k, score in scores.items()
        )
        for k, w in weights.items():
            final_scores[k + f"({int(w*100)}%)"] = scores.pop(k)
        return pd.Series(final_scores)
