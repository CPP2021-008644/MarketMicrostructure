from latency_config import ATColumns, ETypes
import pandas as pd


class ATManipulation:
    "In our code we perform many audit trail manipulation operations. Let's group those functions in this class so it's easier to reuse them."

    @staticmethod
    def find_new_for_replace(at_row, at):
        """
        Given an audit trail row for a replace/restate order finds the corresponding new order in the audit trail

        Parameters
        ----------
        at_row : pd.Sereis
            AT row with a replace/restate
        at : pd.DataFrame
            The AT data
        """
        order_id = at_row[ATColumns.TT_ORDER_ID]
        return at[
            (at[ATColumns.TT_ORDER_ID] == order_id)
            & (at[ATColumns.EXEC_TYPE] == ETypes.NEW)
        ].iloc[0]

    @staticmethod
    def find_previous_replace(at_row, at):
        """
        Given an audit trail row for a replace/restate order finds the immediately previous replace, restated or new

        Parameters
        ----------
        at_row : pd.Sereis
            AT row with a replace/restate
        at : pd.DataFrame
            The AT data
        """
        order_id = at_row[ATColumns.TT_ORDER_ID]
        return at[
            (at[ATColumns.TT_ORDER_ID] == order_id)
            & (
                (at[ATColumns.EXEC_TYPE] == ETypes.NEW)
                | (at[ATColumns.EXEC_TYPE] == ETypes.REPLACED)
                | (at[ATColumns.EXEC_TYPE] == ETypes.RESTATED)
            )
            & (at.index < at_row.name)
        ].iloc[-1]
