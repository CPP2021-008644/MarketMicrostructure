## Estimation of the lead-lag parameter from non-synchronous data [[original paper](https://arxiv.org/abs/1303.4871)]

*Works on Linux, MacOS and Windows (Microsoft Visual C++ 14.0 or greater is required for Windows).*

> Abstract: We propose a simple continuous time model for modeling the lead-lag effect between two financial
assets. A two-dimensional process (Xt, Yt) reproduces a lead-lag effect if, for some time shift
ϑ ∈ R, the process (Xt, Yt+ϑ) is a semi-martingale with respect to a certain filtration. The
value of the time shift ϑ is the lead-lag parameter. Depending on the underlying filtration,
the standard no-arbitrage case is obtained for ϑ = 0. We study the problem of estimating the
unknown parameter ϑ ∈ R, given randomly sampled non-synchronous data from (Xt) and (Yt).
By applying a certain contrast optimization based on a modified version of the Hayashi–Yoshida
covariation estimator, we obtain a consistent estimator of the lead-lag parameter, together with
an explicit rate of convergence governed by the sparsity of the sampling design. The complexity is
**O(n.LOG(n))**.

### API

Calculate the time lag (delay) in seconds between two time series in Python using the `lead_lag` module.

```python
lead_lag.lag(ts1: pd.Series, ts2: pd.Series, max_lag: Union[float, int]) -> Optional[float]
```

#### Arguments
- `ts1`: This is a Pandas Series containing the first time series data.
- `ts2`: This is another Pandas Series containing the second time series data.
- `max_lag`: defines a time interval within which the optimal lag is sought: `[-max_lag, max_lag]`.

It's important to note that the timestamps in the two input series, need not be synchronized. This means that the data points in these series don't have to occur at the exact same times. The module can handle non-synchronous data. However, for efficiency reasons, the smallest achievable lag is set at 100 microseconds.

#### Returns
The signed estimated `lag`, expressed in seconds. 
- If the calculated lag is positive, it implies that `ts1` leads `ts2`.
- If the calculated lag is negative, it implies that `ts2` leads `ts1`.
-  If any issues arise during the calculation, the function returns `None`.

There is also a `lead_lag.LeadLag` object, which offers more features. Refer to the examples to learn how to use it.


