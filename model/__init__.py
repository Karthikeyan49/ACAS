_EXPORTED = ("SatelliteRiskRegressor", "SatelliteRiskClassifier")

def __getattr__(name):
    if name in _EXPORTED:
        from model import lgbm_model as _m
        return getattr(_m, name)
    raise AttributeError(...)