# forecastiso

![logo](./docs/images/logo.svg)

Experimental but well-tested, forecastISO is a times series forecasting project focused on day-ahead system-wide (i.e. at the level of ISOs and RTOs) electricity load forecasting. This is a fundamental task for markets and system operators, as well as an interesting domain for time series analysis due to the complexity and multiscale seasonality. The goal is to make it easy for practitioners and students to develop new methods with reproducible workflows and reliable baselines.

Features:

- Baseline methods and models built-in
- Tools for reproducible analyses
- Evaluation utilities for model comparison
- Framework for incorporating custom models with custom metrics

## For Students and Educators 📚

Another, but not secondary, goal of this project is to provide a set of tools for those new to time series forecasting or machine learning in general. With many built-in utilities and a low barrier to entry, this package is a unique educational offering. It is also the author's opinion that getting more people thinking about energy consumption and the systems that serve us all will benefit everyone.

## Brief Background

## User Guide

Here are the essential commands to get started building and running the project.

| Command         | Purpose                |
| --------------- | ---------------------- |
| `make install`  | Install dependencies   |
| `make test`     | Run unit tests         |
| `make pipeline` | Validate full pipeline |

Note that `make pipeline` may be long running to modify the config as needed.

### Apple Silicon

To get xgboost to install properly, you may need to install libomp - there's more information in the [docs](https://xgboost.readthedocs.io/en/latest/install.html).

## Currently Supported ISOs

| ISO   | Link to Data                                             |
| ----- | -------------------------------------------------------- |
| CAISO | https://www.caiso.com/library/historical-ems-hourly-load |

Support for others is in progress.
