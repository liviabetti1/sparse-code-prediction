import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
import shap
from lime.lime_tabular import LimeTabularExplainer


def top_concepts(model: Ridge | RidgeCV, k: int = 10, concepts=None, abs_val=True):
    weights = np.abs(np.asarray(model.coef_)) if abs_val else np.asarray(model.coef_)
    labels = np.array(concepts) if concepts is not None else np.arange(weights.shape[-1])
    if weights.ndim == 2:
        return {i: labels[np.argsort(w)[::-1][:k]].tolist() for i, w in enumerate(weights)}
    return labels[np.argsort(weights)[::-1][:k]].tolist()
 

def shap_explanations(model, X):
    """Returns SHAP values array (n_samples, n_features)."""
    explainer = shap.LinearExplainer(model, X)
    return explainer.shap_values(X)


def lime_explanations(model, X_train, X_explain, n_samples: int = 1000):
    """Returns list of LIME explanation objects, one per row of X_explain."""
    explainer = LimeTabularExplainer(X_train, mode="regression")
    return [
        explainer.explain_instance(row, model.predict, num_samples=n_samples)
        for row in X_explain
    ]
