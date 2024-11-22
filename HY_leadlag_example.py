from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import datetime as dt
from analysis_metrics.lead_lag.lead_lag import LeadLag

"""
Example of usage of the Hayashi Yoshida lead-lag estimation parameter. It outputs the LLR.
"""


def main():
    min_data_precision = 0.001  # 1ms.
    start = datetime.utcnow().replace(microsecond=0)
    timestamp = [start - timedelta(seconds=i) * min_data_precision for i in range(1000)]
    values = np.cumsum(np.random.uniform(low=-1, high=1, size=len(timestamp)))
    ts = pd.Series(data=values, index=timestamp)

    data_lag = dt.timedelta(milliseconds=100)
    ts_lag = ts.copy() + np.random.normal(0, 1, ts.shape)
    ts_lag.index += data_lag
    ts_lag = ts_lag.dropna()
    ts_lag = ts_lag.drop(ts_lag.sample(frac=0.5).index)

    ll = LeadLag(ts1=ts, ts2=ts_lag, max_lag=4)
    print("Running inference...")
    ll.run_inference(3)
    print(f"Estimated lag is {ll.lead_lag} seconds. True lag was {data_lag} seconds.")
    print(
        f"Positive lag means ts1 is leading. LLR: {ll.llr:.2f} (cf. paper for the definition of LLR)."
    )


if __name__ == "__main__":
    main()
