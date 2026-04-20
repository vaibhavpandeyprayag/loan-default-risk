import statsmodels.api as sm

def train_lr(X, y):
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit()
    return model