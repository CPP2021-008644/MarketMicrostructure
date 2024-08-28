<img src="https://afs-services.com/wp-content/uploads/2025/09/MM_banner.png" alt="RV Banner" style="width: 100%; height: auto; display: block;">

## 
This repository focuses on analyzing market microstructure, particularly within the context of Green and Digital Finance. The goal is to explore financial models, and apply theoretical models and metrics to real market data.

- The most important file is **T2T_microstructure.pdf**, which is the document that describes the research project, microstructure problems, metrics and results.

## Project Overview

This project aims to provide a comprehensive analysis of the dynamics underlying financial
markets, focusing on key aspects of market microstructure. The analysis thesis is included in the document named 
**T2T_microstructure.pdf**
The first chapter of this document focuses on execution, latency, and the operational intricacies of trading systems. It provides an in-depth look at the practical aspects of trading, including the
measurement of various latencies encountered in the execution pipeline, the synchronization of tick-by-tick data with audit trails, and the implications of these factors for market
efficiency and trading performance.
The second and third parts delve into specific metrics that are instrumental in analyzing and comparing different markets, with a particular emphasis on identifying where
price discovery occurs. In Chapter 2, we explore the lead-lag relationships and price discovery metrics, including the Hayashi-Yoshida correlation, which helps in understanding
the temporal dynamics between different assets and how quickly various markets incorporate new information. Chapter 3 extends this analysis by introducing Hasbrouck’s Information Share, a metric designed to quantify the contribution of different markets to price
discovery, offering deeper insights into how information is reflected in asset prices across
multiple trading venues.
Together, these chapters present a multi-faceted view of market microstructure, combining theoretical insights with practical considerations to offer a robust framework for
analyzing the complex interactions that drive financial markets.

## Repository Structure

### Lead-Lag Analysis

The `lead_lag/lead_lag` directory is focused on exploring lead-lag relationships in financial data.

- **`lead_lag_impl.py`**: Core implementation for lead-lag analysis.
- **`conditioned_lead_lag.ipynb`**: Jupyter notebook for Lead/lag response functions conditioned on a movement threshold.
- **Data Files**: Includes various CSV files containing trade and spread data relevant to the analysis, such as **`FFV24.csv`, `SFRU24.csv`** and **`spread_data.csv`**

### Hasbrouck Measure

- **`hasbrouck.py`**: Implements Hasbrouck's measure to analyze market price discovery.

### Metrics Implementation

These scripts help evaluate different metrics, implementing them based on the analyzed data.

- **`metrics.py`**: Framework for  trading metrics. This is the parent class that all metrics child classes should inherit from.
- **`leadlag_metrics.py`**: Specialized script for metrics within lead-lag context.

### Latency computation

These scripts help evaluate different market latencies and implement them based on the analyzed data.
The `analysis_metrics/base` directory contains all the implementation.

- **`server_exch.ipynb`**: The time that an order takes to go from our trading machine to the exchange
- **`hedge_latency.ipynb`**: The time taken by our trading machine to send a hedge order after we receive a fill on the active leg of an autospreader

### Utilities

- **`util_funcs.py`**: A collection of utility functions used throughout the project.
- **`HY_leadlag_example.py`**: An example script showcasing lead-lag analysis.
- **`Hasbrouck_example.ipynb`**: An example script showcasing the Hasbrouck metric.

### Documentation and Dependencies

- **`README.md`**: This documentation file.
- **`requirements.txt`**: A list of Python dependencies needed to run the scripts in this repository. 

## Acknowledgements

This work has been supported by the Government of Spain (Ministerio de Ciencia e Innovación) and the European Union through Project CPP2021-008644 / AEI / 10.13039/501100011033 / Unión Europea Next GenerationEU / PRTR. Visit our website for more information [Green and Digital Finance – Next GenerationEU](https://afs-services.com/proyectos-nextgen-eu/).

<p align="center">
  <img
    src="https://afs-services.com/wp-content/uploads/2025/06/logomciaienetgeneration-1232x264.png"
    alt="Logo MCIAI NetGeneration"
    height="100"
  >
</p>