# import statsmodels.api as sm

# def train_lr(X, y):
#     X = sm.add_constant(X)
#     model = sm.Logit(y, X).fit()
#     return model

from sklearn.tree import DecisionTreeClassifier

def train_dt(X, y):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    return model