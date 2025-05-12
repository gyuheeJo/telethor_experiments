from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def scale(X, scaler_type = 'minmax'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise Exception
    X = scaler.fit_transform(X)

    return X

def ts_preprocess(X, Y = None, n_steps = 30):

    X_processed = []

    for i in range(n_steps, len(X)+1):
        i_ = i - n_steps
        x = X[i_:i]

        X_processed.append(x)

    if Y is not None:
        return np.array(X_processed), Y[n_steps-1:] 
    return np.array(X_processed)

def balance(X, y):
    X_class_1 = X[y == 1]
    y_class_1 = y[y == 1]

    sample_size = X_class_1.shape[0]

    X_class_0 = X[y == 0]
    y_class_0 = y[y == 0]

    # Muestreo aleatorio de clase 0
    np.random.seed(42)
    indices_class_0 = np.random.choice(len(X_class_0), size=sample_size, replace=False)
    X_class_0_sample = X_class_0[indices_class_0]
    y_class_0_sample = y_class_0[indices_class_0]

    # Combinar las clases
    X_balanced = np.concatenate([X_class_1, X_class_0_sample], axis=0)
    y_balanced = np.concatenate([y_class_1, y_class_0_sample], axis=0)

    return X_balanced, y_balanced