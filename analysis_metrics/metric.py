import datetime as dt


class Metric:
    """Data class that contains the results of a strategy execution"""

    do_not_include_when_exporting = [
        "name",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "entry_side",
    ]

    def __init__(
        self,
        name: str,
        is_there_entry: bool = False,
        entry_time: dt.datetime = None,
        entry_price=None,
        exit_time: dt.datetime = None,
        exit_price=None,
        entry_side: str = None,
        **kwargs,
    ):
        self.name = name
        self.is_there_entry = is_there_entry
        if entry_side is not None:
            self.entry_side = entry_side.lower()
            if entry_side.lower() not in ["buy", "sell"]:
                raise ValueError(f"Entry side {entry_side} not in [buy, sell]")
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.exit_time = exit_time
        self.exit_price = exit_price
        for attr, val in kwargs.items():
            setattr(self, attr, val)  # Optional attributes

    def export_to_dict(self):
        """
        Exports all attributes of this class to a dict. It includes the name of the strategy with a separator,
        such that it is well identified in the output
        """
        return {
            f"{key}|{self.name}": value
            for key, value in self.__dict__.items()
            if key not in self.do_not_include_when_exporting
        }
