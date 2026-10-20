import numpy as np


def create_spiral_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype=int)

    c = 0
    while c < classes:
        ix_start = c * samples
        ix_end = (c + 1) * samples

        r = np.linspace(0.0, 1.0, samples)
        t = np.linspace(c * 4, (c + 1) * 4, samples) + np.random.randn(samples) * 0.2

        X[ix_start:ix_end, 0] = r * np.sin(t * 2.0 * np.pi)
        X[ix_start:ix_end, 1] = r * np.cos(t * 2.0 * np.pi)
        y[ix_start:ix_end] = c

        c = c + 1

    return X, y


def load_dry_bean(filepath):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_excel(filepath)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    class_names = le.classes_

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names
