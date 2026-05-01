from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV

_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

def train_ridge_regression(X_train, y_train, alpha: float, cv: bool = False):
    model = RidgeCV(alphas=_ALPHAS) if cv else Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train, alpha: float, cv: bool = False):
    model = LassoCV(alphas=_ALPHAS) if cv else Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_ridge_classifier(X_train, y_train, alpha: float, cv: bool = False):
    model = RidgeClassifierCV(alphas=_ALPHAS) if cv else RidgeClassifier(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_logistic_classifier(X_train, y_train, alpha: float = 1.0, cv: bool = False):
    if cv:
        Cs = [1.0 / a for a in _ALPHAS]
        model = LogisticRegressionCV(Cs=Cs)
    else:
        model = LogisticRegression(C=1.0 / alpha)

    model.fit(X_train, y_train)
    return model

def train(
    model_name: str,
    X_train,
    y_train,
    *,
    alpha: float = 1.0,
    C: float = 1.0,
    cv: bool = False,
):
    model_name = model_name.lower()

    if model_name in ["ridge", "ridge_regression"]:
        return train_ridge_regression(X_train, y_train, alpha=alpha, cv=cv)

    elif model_name in ["lasso", "lasso_regression"]:
        return train_lasso_regression(X_train, y_train, alpha=alpha, cv=cv)

    elif model_name in ["ridge_classifier", "ridge_clf"]:
        return train_ridge_classifier(X_train, y_train, alpha=alpha, cv=cv)

    elif model_name in ["logistic", "logistic_regression", "logreg"]:
        return train_logistic_classifier(X_train, y_train, alpha=alpha, cv=cv)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")