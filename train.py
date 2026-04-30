from sklearn.linear_model import Ridge, RidgeCV

_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

def train_ridge(X_train, y_train, alpha: float, cv: bool = False):
    model = RidgeCV(alphas=_ALPHAS) if cv else Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model