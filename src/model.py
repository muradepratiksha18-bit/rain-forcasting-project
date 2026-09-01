"""Multi-task LSTM: predicts how much rain, and whether it rains at all."""
import torch
import torch.nn as nn


class RainfallLSTM(nn.Module):
    """Shared LSTM encoder with two heads.

    Daily rainfall is zero-inflated: ~2/3 of days are dry, and a regression head
    on its own learns to hedge towards zero. Training a rain/no-rain classifier
    alongside it gives the encoder a cleaner gradient signal for "is a wet spell
    starting", and at inference the two heads combine into a calibrated forecast.
    """

    def __init__(self, n_features, hidden=96, layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.head_clf = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(self.norm(out[:, -1, :]))   # last timestep summarises the window
        # reg is in log1p(mm) space; clf is a raw logit.
        return self.head_reg(h).squeeze(-1), self.head_clf(h).squeeze(-1)


class MultiTaskLoss(nn.Module):
    """Intensity-weighted Huber on log-rainfall + BCE on rain occurrence.

    Two imbalances have to be corrected or the model just predicts "dry":

    - `pos_weight` rebalances the classifier against the dry-day majority.
    - `intensity_weight` up-weights the regression loss on heavy days. Without
      it the handful of cloudburst days is drowned out by ~1100 easy dry days,
      and the model systematically under-forecasts exactly the events that
      matter. Weight grows with log1p(mm), so a 100 mm day counts several times
      a 1 mm day without letting one outlier dominate the batch.

    Huber rather than MSE keeps a single 250 mm day from destabilising training;
    `delta` sets where it switches from quadratic to linear, in log-mm units.
    """

    def __init__(self, pos_weight=1.0, clf_weight=0.5, intensity_weight=0.6, delta=2.0):
        super().__init__()
        self.reg = nn.HuberLoss(delta=delta, reduction="none")
        self.clf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.clf_weight = clf_weight
        self.intensity_weight = intensity_weight

    def forward(self, pred_reg, pred_clf, y_reg, y_clf):
        w = 1.0 + self.intensity_weight * y_reg   # y_reg is already log1p(mm)
        lr = (self.reg(pred_reg, y_reg) * w).sum() / w.sum()
        lc = self.clf(pred_clf, y_clf)
        return lr + self.clf_weight * lc, lr.item(), lc.item()
