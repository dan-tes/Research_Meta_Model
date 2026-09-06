import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, \
    HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchvision import datasets, transforms
import ast
import warnings

warnings.filterwarnings('ignore')

# ========================
# ГЛОБАЛЬНЫЙ КОМПРОМИСС СКОРОСТЬ / КАЧЕСТВО
# ========================
# Минимальное относительное улучшение val_loss (к текущему лучшему), которое
# должна приносить одна дополнительная эпоха, чтобы обучение продолжалось.
# Большое значение = сильный штраф за лишние эпохи = ранняя остановка.
EPOCH_PENALTY = 0.03

# ========================
# DEVICE SETUP (CUDA)
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def gpu_selftest():
    """Небольшая проверка GPU. Вызывается вручную, а не при импорте модуля,
    чтобы импорт не тратил ~1 ГБ видеопамяти и время на matmul."""
    if not torch.cuda.is_available():
        print("CUDA недоступна")
        return

    print("torch version:", torch.__version__)
    print("device count:", torch.cuda.device_count())
    print("device name:", torch.cuda.get_device_name(0))
    print("torch cuda version:", torch.version.cuda)

    x = torch.randn(10000, 10000, device="cuda")
    y = torch.randn(10000, 10000, device="cuda")
    z = x @ y
    torch.cuda.synchronize()

    print("result device:", z.device)
    print("allocated MB:", torch.cuda.memory_allocated() / 1024 ** 2)

# ========================
# FEATURE ENGINEERING (как было)
# ========================

FEATURE_GROUPS = {

    # A. Текущее состояние
    "state": [
        "loss_last_1",
        "loss_last_2",
        "distance_norm",
        "epochs_since_best",
    ],

    # B. Глобальный тренд
    "globalTrend": [
        "global_slope",
        "global_curvature",
    ],

    # C. Локальный тренд
    "localTrend": [
        "recent_slope",
        "recent_improvement",
    ],

    # D. Ускорение
    "acceleration": [
        "acc_norm",
    ],

    # # E. Шум
    # "noise": [
    #     "std_recent",
    #     "sign_changes",
    # ],
}

def build_feature_set(*groups):
    """
    build_feature_set("state", "local_trend")

    -> список признаков для обучения
    """
    features = []

    for group in groups:
        features.extend(FEATURE_GROUPS[group])

    return features


feature_order = build_feature_set("state")

# ========================
# ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ========================
def load_loss_data(csv_path="data/final.csv"):
    """Загрузка данных о loss"""
    df = pd.read_csv(csv_path, sep=';')

    # Преобразование строк в списки
    df['val_loss'] = df['val_loss'].map(ast.literal_eval)
    df['train_loss'] = df['train_loss'].map(ast.literal_eval)

    # Flatten вложенные списки
    df['val_loss'] = df['val_loss'].map(lambda l: [v[0] if isinstance(v, list) else v for v in l])
    df['train_loss'] = df['train_loss'].map(lambda l: [v[0] if isinstance(v, list) else v for v in l])

    # Фильтрация
    df = df[df['val_loss'].apply(lambda x: len(x) > 4)]
    df = df[df['shift_type'].isin(['none', 'noise'])]

    return df


def build_loss_features(df):
    """Создание признаков на основе loss"""
    # Loss features
    df["loss_start"] = df["val_loss"].apply(lambda x: x[0])
    df["loss_last_1"] = df["val_loss"].apply(lambda x: x[-2] / x[0])
    df["loss_last_2"] = df["val_loss"].apply(lambda x: x[-3] / x[0])
    df['acc_norm'] = df["val_loss"].apply(
        lambda c: ((c[-2] - c[-3]) - (c[-3] - c[-4])) / (c[-3] + 1e-8)
    )
    df["loss_end"] = df["val_loss"].apply(lambda x: x[-1])
    df['distance_norm'] = df['val_loss'].apply(
        lambda x: (x[-2] - min(x[:-1])) / (x[0] - min(x[:-1]) + 1e-8)
    )

    # Global features
    def build_global_features(val_loss):
        val = np.array(val_loss)
        n = len(val)
        x = np.arange(n)

        slope = np.polyfit(x, val, 1)[0] if n > 1 else 0
        curvature = np.polyfit(x, val, 2)[0] if n > 2 else 0
        best_idx = np.argmin(val)

        return pd.Series({
            "global_slope": slope,
            "global_curvature": curvature,
            "epochs_since_best": n - best_idx,
        })

    global_df = df["val_loss"].apply(build_global_features)
    df = pd.concat([df, global_df], axis=1)

    # Recent features
    def build_recent_features(val_loss, window=5):
        val = np.array(val_loss)
        n = len(val)

        if n < 2:
            return pd.Series({
                "recent_slope": 0,
                "recent_improvement": 0
            })

        val_recent = val[-window:] if n >= window else val
        x = np.arange(len(val_recent))

        slope = np.polyfit(x, val_recent, 1)[0] if len(val_recent) > 1 else 0
        improvement = val_recent[0] - val_recent[-1]

        return pd.Series({
            "recent_slope": slope,
            "recent_improvement": improvement
        })

    recent_df = df["val_loss"].apply(build_recent_features)
    df = pd.concat([df, recent_df], axis=1)

    return df


# ========================
# PyTorch МОДЕЛИ
# ========================

class MLPRegressor_PyTorch(nn.Module):
    """MLP для регрессии"""

    def __init__(self, input_dim, hidden_dim=64):
        super(MLPRegressor_PyTorch, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class MLPForMNIST(nn.Module):
    """MLP для MNIST классификации"""

    def __init__(self):
        super(MLPForMNIST, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)


class CNNForCIFAR(nn.Module):
    """Небольшая свёрточная сеть для CIFAR-10 (вход: N x 3 x 32 x 32)"""

    def __init__(self):
        super(CNNForCIFAR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 64 x 8 x 8

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class SimpleMLPRegression(nn.Module):
    """Простая MLP для регрессии на Wine dataset (один непрерывный выход)"""

    def __init__(self, input_dim=11):
        super(SimpleMLPRegression, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ========================
# EARLY STOPPING CALLBACKS
# ========================

class SmartEarlyStoppingMultiStep:
    """Early Stopping, который прогнозирует ближайший кусок кривой val_loss и
    останавливает обучение, как только ожидаемое улучшение перестаёт окупать
    эпохи, которые оно стоит.

    Компромисс задаётся параметром ``epoch_penalty`` — это минимальное
    относительное (к текущему лучшему val_loss) улучшение, которое должна
    приносить одна дополнительная эпоха, чтобы обучение продолжалось.
    Чем больше ``epoch_penalty`` — тем раньше происходит остановка.
    """

    def __init__(self, model_meta=None, train_size=None, future_steps=5,
                 min_epochs=10, patience=2, epoch_penalty=EPOCH_PENALTY,
                 forecast_window=6, restore_best_weights=True, use_meta=False,
                 meta_path="models/meta_forecaster.pkl"):
        self.model_meta = model_meta
        self.train_size = train_size
        self.future_steps = future_steps
        self.min_epochs = min_epochs
        self.patience = patience
        self.epoch_penalty = epoch_penalty
        self.forecast_window = forecast_window
        self.restore_best_weights = restore_best_weights
        # Вариант C: табличный GBM-прогнозист из meta_forecaster.py, обученный на
        # кривых ИЗ ЭТОГО ЖЕ пайплайна (gen_curves.py) и предсказывающий НАПРЯМУЮ
        # ещё доступное относительное улучшение за горизонт (а не одношаговую
        # дельту с авторегрессией, как сломанная модель на final.csv). Включается
        # явно через use_meta=True; при отсутствии .pkl тихо откатываемся на тренд.
        self.meta = None
        self.meta_horizon = None
        if use_meta:
            try:
                import meta_forecaster as MF
                self.meta = MF.load_meta(meta_path)
                self.meta_horizon = self.meta["config"].get("horizon", 10)
            except Exception:
                self.meta = None
        self.use_meta = self.meta is not None

        self.val_loss = []
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, epoch, val_loss):
        """Вызывается в конце каждой эпохи. Возвращает True, если пора стоп."""
        val_loss = float(val_loss)
        self.val_loss.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if len(self.val_loss) < self.min_epochs:
            return False

        if self.use_meta:
            rel_gain = self._forecast_meta_gain()                # прямой прогноз GBM
            budget = self.epoch_penalty * self.meta_horizon      # горизонт GBM (10)
        else:
            preds = self._forecast()
            projected = min([self.val_loss[-1], *preds])
            denom = max(self.best_val_loss, 1e-8)
            rel_gain = (self.best_val_loss - projected) / denom  # ещё доступное улучшение
            budget = self.epoch_penalty * self.future_steps      # сколько мы готовы "заплатить"

        if rel_gain < budget:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    # ------------------------------------------------------------------
    # Прогноз будущих значений
    # ------------------------------------------------------------------
    def _forecast(self):
        return self._forecast_trend()

    def _forecast_trend(self):
        """Устойчивая линейная экстраполяция по короткому недавнему окну.

        Линейный тренд на последних эпохах показывает, продолжает ли кривая
        снижаться, но без "разбегания" квадратичной экстраполяции (которая на
        падающей кривой уводит прогноз в минус и мешает остановке).
        """
        y = np.asarray(self.val_loss, dtype=float)
        w = int(min(self.forecast_window, len(y)))
        yr = y[-w:]
        slope = np.polyfit(np.arange(w), yr, 1)[0] if w > 1 else 0.0

        steps = np.arange(1, self.future_steps + 1)
        preds = yr[-1] + slope * steps
        preds = np.clip(preds, 0.0, None)
        if slope >= 0:                       # тренд вверх/плато -> дальше улучшений нет
            preds[:] = yr[-1]
        return preds.tolist()

    def _forecast_meta_gain(self):
        """Вариант C: табличный GBM напрямую оценивает ещё доступное
        относительное улучшение val_loss за meta_horizon эпох по признакам
        текущего префикса. При сбое — откат на линейный тренд."""
        try:
            import meta_forecaster as MF
            feats = MF.features_from_prefix(np.asarray(self.val_loss, dtype=np.float64))
            row = pd.DataFrame([feats])[self.meta["features"]].astype(np.float64)
            return float(np.clip(self.meta["relgain"].predict(row)[0], 0.0, 1.0))
        except Exception:
            preds = self._forecast_trend()
            projected = min([self.val_loss[-1], *preds])
            return (self.best_val_loss - projected) / max(self.best_val_loss, 1e-8)

    @staticmethod
    def _meta_features(val_np):
        if len(val_np) < 5:
            val_np = np.pad(val_np, (5 - len(val_np), 0), mode="edge")

        loss0 = max(float(val_np[0]), 1e-8)
        x_full = np.arange(len(val_np))
        best_idx = int(np.argmin(val_np))
        recent = val_np[-5:]

        return {
            "loss_last_1": val_np[-1] / loss0,
            "loss_last_2": val_np[-2] / loss0,
            "distance_norm": (val_np[-1] - val_np.min()) / (loss0 - val_np.min() + 1e-8),
            "epochs_since_best": (len(val_np) - best_idx) / len(val_np),
            "global_slope": np.polyfit(x_full, val_np, 1)[0] / loss0,
            "global_curvature": np.polyfit(x_full, val_np, 2)[0] / loss0,
            "recent_slope": np.polyfit(np.arange(len(recent)), recent, 1)[0] / loss0,
            "recent_improvement": (recent[0] - recent[-1]) / loss0,
            "acc_norm": (val_np[-1] - 2 * val_np[-2] + val_np[-3]) / loss0,
        }

class SimpleEarlyStopping:
    """Стандартный Early Stopping"""

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


class ParametricEarlyStopping:
    """Параметрический Early Stopping: подгоняет экспоненту a*exp(-b*t)+c к
    кривой val_loss и останавливается, когда прогноз обещает меньше улучшения,
    чем ``epoch_penalty`` (относительно лучшего val_loss) на эпоху."""

    def __init__(self, min_epochs=10, patience=2, epoch_penalty=EPOCH_PENALTY, future_steps=5):
        self.min_epochs = min_epochs
        self.patience = patience
        self.epoch_penalty = epoch_penalty
        self.future_steps = future_steps
        self.val_loss = []
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, epoch, val_loss):
        val_loss = float(val_loss)
        self.val_loss.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if len(self.val_loss) < self.min_epochs:
            return False

        try:
            from scipy.optimize import curve_fit

            y = np.asarray(self.val_loss)
            x = np.arange(len(y))

            def exp_func(t, a, b, c):
                return a * np.exp(-b * t) + c

            popt, _ = curve_fit(exp_func, x, y, bounds=(0, [10, 10, 10]), maxfev=5000)

            future_epochs = np.arange(len(y), len(y) + self.future_steps)
            preds = exp_func(future_epochs, *popt)

            projected = float(min(preds.min(), y[-1]))
            rel_gain = (self.best_val_loss - projected) / max(self.best_val_loss, 1e-8)

            if rel_gain < self.epoch_penalty * self.future_steps:
                self.counter += 1
            else:
                self.counter = 0

            if self.counter >= self.patience:
                self.should_stop = True
                return True
        except Exception:
            pass

        return False


# ========================
# ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ
# ========================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    """Валидация"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            total_loss += loss.item() * batch_X.size(0)

    return total_loss / len(val_loader.dataset)


def train_epoch_classifier(model, train_loader, criterion, optimizer, device):
    """Обучение классификатора"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(train_loader.dataset), correct / total


def validate_classifier(model, val_loader, criterion, device):
    """Валидация классификатора"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(val_loader.dataset), correct / total


def evaluate_classifier(model, test_loader, device):
    """Оценка точности классификатора"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total


def run_training_classifier(X_train, y_train, X_test, y_test, train_size,
                            callback_name, model_builder, batch_size=16, epochs=200):
    """Обучение классификатора с выбранным callback"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Создание DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train[:train_size], dtype=torch.float32),
        torch.tensor(y_train[:train_size], dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # Split для валидации
    train_size_split = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size_split
    train_dataset, val_dataset = random_split(train_dataset, [train_size_split, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Модель
    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Выбор callback
    if callback_name == "early":
        callback = SimpleEarlyStopping(patience=5)
    elif callback_name == "smart":
        callback = SmartEarlyStoppingMultiStep(None, train_size)  # Без meta-модели
    elif callback_name == "parametric":
        callback = ParametricEarlyStopping()

    best_val_loss = np.inf
    epochs_trained = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_classifier(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_classifier(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        epochs_trained = epoch + 1

        # Callback step
        if callback_name == "early":
            should_stop = callback.step(val_loss)
        else:
            should_stop = callback.step(epoch, val_loss)

        if should_stop:
            break

    # Тестирование
    test_acc = evaluate_classifier(model, test_loader, device)

    return {
        "epochs": epochs_trained,
        "test_acc": test_acc,
        "best_val_loss": best_val_loss
    }


# ========================
# БЕНЧМАРКИНГ
# ========================

def benchmark_sklearn_models(X_train, X_test, y_train, y_test):
    """Бенчмаркинг sklearn моделей"""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(),
        "HistGradientBoosting": HistGradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
        "MLP": MLPRegressor(max_iter=500)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "model_obj": model
        })

    df = pd.DataFrame(results).sort_values("RMSE", ascending=True)

    # лучший результат
    best_model = df.iloc[0]["model_obj"]
    print(f'Лучшая модель {df.iloc[0]}')
    return best_model

from sklearn.base import clone

def evaluate_feature_set(
        X,
        y,
        feature_list,
):
    print(feature_list, X.columns)
    X_sub = X[feature_list]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y,
        test_size=0.2,
        random_state=42
    )
    best_model = benchmark_sklearn_models(
        X_train,
        X_test,
        y_train,
        y_test
    )
    model = clone(best_model)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }

if __name__ == '__main__':

    from itertools import combinations

    FEATURE_SETS = {}

    group_names = list(FEATURE_GROUPS.keys())

    for r in range(1, len(group_names) + 1):
        for combo in combinations(group_names, r):

            features = []
            for group in combo:
                features.extend(FEATURE_GROUPS[group])

            # убрать возможные дубликаты, сохранив порядок
            features = list(dict.fromkeys(features))

            if len(combo) == len(group_names):
                name = "all"
            else:
                name = "_".join(combo)

            FEATURE_SETS[name] = features
    results = []

    df = load_loss_data()
    df = build_loss_features(df)

    X = df[feature_order]

    y = df["val_loss"].apply(
        lambda x:
        (x[-1] - x[-2]) /
        (x[-2] + 1e-8)
    )
    df = load_loss_data()
    df = build_loss_features(df)

    # X = df
    y = y.loc[X.index]

    results = []
    MODELS = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(),
        "HistGradientBoosting": HistGradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
        "MLP": MLPRegressor(max_iter=500)
    }
    for model_name, model in MODELS.items():

        for feature_name, features in FEATURE_SETS.items():
            X = df[features]


            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            results.append({
                "model": model_name,
                "feature_set": feature_name,
                "R2": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            })

    results = pd.DataFrame(results)
    results = results.sort_values("RMSE", ascending=False)
    print(results)
    pivot = results.pivot(
        index="model",
        columns="feature_set",
        values="RMSE"
    ).sort_values("all", ascending=False)

    plt.figure(figsize=(14, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis_r"
    )
    plt.title("RMSE for feature groups and regression models")
    plt.tight_layout()
    plt.show()