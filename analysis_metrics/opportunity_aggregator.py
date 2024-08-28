from loader.input_loader import (
    t2t_loader,
    complete_generic_spread,
    elligible_spreads_for_event,
)
from analysis.opportunity import Opportunity
from analysis.fading_opportunity import FadingOpportunity
from analysis.leadlag_opportunity import LeadLagOpportunity
import pandas as pd
import logging
import traceback
from tqdm import tqdm
from output.plots import short_plot, detailed_plot


class OpportunityAggregator:
    """
    This opportunity aggregator is in charge of calling the corresponding loading, opportunity (metrics) and plotting functions
    Then, it aggregates the results by spread, event or both
    """

    def __init__(self, params: dict, events: pd.DataFrame, spreads_def: pd.DataFrame):
        self.params = params
        self.events = events
        self.spreads_def = spreads_def
        self.data = {}
        self.opportunity_raw_results = {}
        self.opportunity_df = None
        self.result_across_events = None
        self.result_across_spreads = None
        self._initialize_tqdm()

    def _initialize_tqdm(self):
        num_events = len(self.events)
        num_eligible_spreads = len(
            elligible_spreads_for_event(
                self.params, self.events.iloc[0], self.spreads_def
            )
        )
        self._tqdm = tqdm(total=num_events * num_eligible_spreads)

    def _process_event_spread_pair(self, spread_def: pd.Series, event: pd.Series):
        """
        Processes one spread,event pair: calls corresponding loader, computes metrics by calling
        the opportunity object, and finally plots the results
        """
        spread_def = complete_generic_spread(self.params, spread_def, event)
        spread_name = spread_def["ArfimaName"]
        event_name = event["EventName"]
        LOG.info(f"\n### Analyzing {spread_name} in event {event_name} ###\n")
        if not spread_name in self.opportunity_raw_results:
            self.opportunity_raw_results[spread_name] = {}
        if event_name in self.opportunity_raw_results[spread_name]:
            return
        try:
            # Load Market Data from t_0 - event_time_before to t_T
            t_0 = event["from_date"] - self.params["input"]["event_time_before"]
            t_1 = event["from_date"] + self.params["input"]["event_time_after"]
            if (
                self.params["input"].get("use_events_csv_time_after", False)
                and "to_date" in event.index
            ):
                t_1 = event["to_date"]
            MarketData = t2t_loader(
                self.params,
                t_0,
                t_1,
                spread_def.to_frame().transpose(),
            )
            spread = MarketData[MarketData.keys()[0]]
            # Create an opportunity instance from the event-spread pair.
            opportunity = LeadLagOpportunity(self.params, event, spread_def, spread)

            # Calculate all statistics for the opportunity instance and store them.
            self.opportunity_raw_results[spread_name][
                event_name
            ] = opportunity.find_opportunities()
            if (
                self.params["output"]["show_detailed_spread_event_pair"]
                and (opportunity.spread_name, opportunity.event["EventName"])
                in self.params["output"]["spread_event_pair"]
            ):
                detailed_plot(opportunity, self.params)

            short_plot(opportunity, self.params)
            self._tqdm.update(1)
        except Exception as e:
            self._tqdm.update(1)
            LOG.error(f"{traceback.format_exc()}, skipping")
            return

    def generate_opportunity_df(self) -> pd.DataFrame:
        """Loops through all the spread,event pairs to call _process_event_spread_pair"""
        self.events.apply(
            lambda event: elligible_spreads_for_event(
                self.params, event, self.spreads_def
            ).apply(
                lambda spread_def: self._process_event_spread_pair(spread_def, event),
                axis=1,
            ),
            axis=1,
        )
        self._tqdm.close()

        self.opportunity_df = pd.concat(
            {
                spread: pd.DataFrame(events).transpose()
                for spread, events in self.opportunity_raw_results.items()
            }
        )
        time_cols = [
            col
            for col in self.opportunity_df.columns
            if self.opportunity_df[col].first_valid_index() is not None
            and isinstance(
                self.opportunity_df.loc[
                    self.opportunity_df[col].first_valid_index(), col
                ],
                pd.Timestamp,
            )
        ]
        self.opportunity_df = self.opportunity_df.drop(columns=time_cols).astype(
            "float64"
        )

    def _aggregate(self, grouped_df: pd.DataFrame):
        """After generate_opportunity_df has been callen and the metrics are computed, this
        function aggregates the results"""
        aggreg = self.params["opportunity"]["aggregation"]["default"]
        percentiles = [int(perc.split("%")[0]) / 100 for perc in aggreg if "%" in perc]
        aggregation = grouped_df.describe([])
        aggregation = self._compute_percentiles_correctly(
            aggregation, grouped_df, percentiles
        )
        specific_agg = self.params["opportunity"]["aggregation"]["specific"]
        for field, cols in specific_agg.items():
            if not field in aggregation:
                continue
            aggregation.iloc[
                :,
                (aggregation.columns.get_level_values(0) == field)
                & (~aggregation.columns.get_level_values(1).isin(cols)),
            ] = None
        try:
            return aggregation.loc[:, (slice(None), aggreg)]
        except:
            return aggregation.T[aggregation.T.index.isin(aggreg, level=1)].T

    def _compute_percentiles_correctly(
        self, aggregation: pd.DataFrame, grouped_df: pd.DataFrame, percentiles
    ):
        # Okay so pandas decided to interpolate data to compute percentiles
        aggregation = aggregation.stack(level=0).drop(columns=["50%"])
        percentiles = (
            grouped_df.quantile(percentiles, interpolation="higher")
            .unstack()
            .stack(level=0)
        )
        percentiles.columns = [f"{int(p*100)}%" for p in percentiles.columns]

        statistics_df = pd.concat(
            [aggregation, percentiles],
            axis=1,
        )

        new_col_order = (
            ["count", "mean", "std", "min"] + list(percentiles.columns) + ["max"]
        )
        statistics_df = statistics_df[new_col_order]

        return statistics_df.stack(level=0).unstack(level=1).unstack(level=1)

    def statistics_across_events(self):
        """Aggregates by spread"""
        LOG.info("Performing analysis across events")
        if self.opportunity_df is None:
            self.generate_opportunity_df()
        if not self.result_across_events:
            self.result_across_events = self._aggregate(
                self.opportunity_df.groupby(level=[0])
            )
        return self.result_across_events

    def statistics_across_spreads(self):
        """Aggregates by event"""
        LOG.info("Performing analysis across events")
        if self.opportunity_df is None:
            self.generate_opportunity_df()
        if self.result_across_spreads is None:
            self.result_across_spreads = self._aggregate(
                self.opportunity_df.groupby(level=[1])
            )
        return self.result_across_spreads
