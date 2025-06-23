# forecastiso

Time series forecasting tools applied to ISO electricity demand data

## Development

| Command        | Purpose              |
| -------------- | -------------------- |
| `make install` | Install dependencies |
| `make test`    | Run unit tests       |

### Apple Silicon

Using xgboost can cause problems on Apple silicon.

## Modeling Notes

### Baselines

I experimented with linear regression and ridge as baselines, but found that collinearity and matrix instability made them unreliable without aggressive feature pruning or regularization. Instead, I focused on naive and rolling statistical baselines, which are both interpretable and robust, and compared those directly to a non-linear gradient boosting model.
