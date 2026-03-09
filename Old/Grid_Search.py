import pandas as pd 
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 
# Models 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import make_scorer, precision_score

def test_combinations(X_train, y_train, random_state=None):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    param_distributions = [

        # ---- Logistic Regression ----
        {
            'scaler': [StandardScaler(), RobustScaler()],
            'clf': [LogisticRegression(max_iter=2000, solver='lbfgs')],
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__class_weight': [None, 'balanced'],
        },

        # ---- Linear SVM ----
        {
            'scaler': [StandardScaler()],
            'clf': [LinearSVC(max_iter=5000)],
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__class_weight': [None, 'balanced'],
        },

        # ---- RBF SVM ----
        {
            'scaler': [StandardScaler()],
            'clf': [SVC()],
            'clf__C': [0.1, 1, 10],
            'clf__gamma': ['scale', 0.1, 0.01],
            'clf__class_weight': [None, 'balanced'],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    precision_M = make_scorer(
        precision_score,
        pos_label="M",
        zero_division=0
    )
    
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=50,
        scoring=precision_M,
        n_jobs=10,
        cv=cv,
        random_state=random_state,
        verbose=1
    )

    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_).sort_values('mean_test_score', ascending=False)
    best_model = search.best_estimator_

    return search, results, best_model
