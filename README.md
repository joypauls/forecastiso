# forecastiso

Time series forecasting tools applied to ISO electricity demand data

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

## Get the Data

| ISO   | Link                                                     |
| ----- | -------------------------------------------------------- |
| CAISO | https://www.caiso.com/library/historical-ems-hourly-load |

## Modeling Notes

### Baselines

I experimented with linear regression and ridge as baselines, but found that collinearity and matrix instability made them unreliable without aggressive feature pruning or regularization. Instead, I focused on naive and rolling statistical baselines, which are both interpretable and robust, and compared those directly to a non-linear gradient boosting model.
