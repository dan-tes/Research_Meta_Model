"""seq2seq-прогноз кривой val_loss для SmartEarlyStopping (вариант A).

Модель: GRU-энкодер по нормированному префиксу кривой -> линейная голова,
которая сразу выдаёт HORIZON будущих значений (без авторегрессии, чтобы ошибка
не накапливалась).

Нормировка — поэлементное деление на val_loss[0]: кривая всегда стартует с 1.0,
а прогноз/цель живут в относительной шкале, одинаковой для MNIST-CE и Wine-MSE.
"""
import numpy as np
import torch
import torch.nn as nn

# --- параметры представления (должны совпадать при обучении и инференсе) ---
MIN_PREFIX = 5       # минимум эпох на входе, иначе прогноз не считаем
HORIZON = 10         # на сколько эпох вперёд предсказываем
MAX_LEN = 40         # длиннее префикс обрезаем слева (берём последние MAX_LEN)
N_FEATURES = 2       # [v_norm, delta]

DEFAULT_PATH = "models/curve_forecaster.pt"


# ------------------------------------------------------------------ модель
class CurveForecaster(nn.Module):
    def __init__(self, hidden=64, horizon=HORIZON, n_features=N_FEATURES, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)              # h: (1, B, hidden)
        return self.head(h[-1])              # (B, horizon) — нормированные будущие значения


# ------------------------------------------------------------------ фичи
def _features_from_norm(v_norm):
    """v_norm: 1-D np.array нормированной кривой -> (T, N_FEATURES)."""
    v_norm = np.asarray(v_norm, dtype=np.float32)
    delta = np.diff(v_norm, prepend=v_norm[:1])
    return np.stack([v_norm, delta], axis=1)


def prefix_to_input(val_loss):
    """Полная история val_loss (list/array сырых значений) -> (x_tensor[1,T,F], length).

    Возвращает None, если истории меньше MIN_PREFIX.
    """
    v = np.asarray(val_loss, dtype=np.float64)
    if len(v) < MIN_PREFIX:
        return None
    v0 = max(float(v[0]), 1e-8)
    v_norm = (v / v0).astype(np.float32)[-MAX_LEN:]
    feats = _features_from_norm(v_norm)
    x = torch.from_numpy(feats).unsqueeze(0)          # (1, T, F)
    return x, torch.tensor([feats.shape[0]]), v0


def make_windows(curve, min_prefix=MIN_PREFIX, horizon=HORIZON, max_len=MAX_LEN):
    """Одна сырая кривая -> список примеров (x_feats, y_future_norm, y_mask)."""
    v = np.asarray(curve, dtype=np.float64)
    if len(v) < min_prefix + 1:
        return []
    v0 = max(float(v[0]), 1e-8)
    v_norm = (v / v0).astype(np.float32)

    out = []
    for k in range(min_prefix, len(v_norm)):
        x = _features_from_norm(v_norm[max(0, k - max_len):k])
        fut = v_norm[k:k + horizon]
        y = np.zeros(horizon, dtype=np.float32)
        m = np.zeros(horizon, dtype=np.float32)
        y[:len(fut)] = fut
        m[:len(fut)] = 1.0
        out.append((x, y, m))
    return out


def collate(batch):
    xs, ys, ms = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    T = int(lengths.max())
    xb = torch.zeros(len(xs), T, N_FEATURES, dtype=torch.float32)
    for i, x in enumerate(xs):
        xb[i, :len(x)] = torch.from_numpy(x)
    return xb, lengths, torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms))


# ------------------------------------------------------------------ загрузка
def load_forecaster(path=DEFAULT_PATH, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = CurveForecaster(
        hidden=cfg.get("hidden", 64),
        horizon=cfg.get("horizon", HORIZON),
        n_features=cfg.get("n_features", N_FEATURES),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model
