# forecastiso

![logo](./logo.svg)

Time series forecasting tools applied to ISO electricity load data.

Provides reliable and reproducible baseline results and a framework for evaluating custom models.

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
