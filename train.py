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

def train_logistic_classifier(X_train, y_train, C: float = 1.0, cv: bool = False):
    model = LogisticRegressionCV() if cv else LogisticRegression(C=C)
    model.fit(X_train, y_train)
    return model