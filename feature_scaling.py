import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression

# Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def get_best_features(df, k, score, label_name='Class'):

    bad_cols = ['Unnamed: 0']
    for bad_col in bad_cols:
        if bad_col in df.columns:
            df = df.drop(columns=[bad_col])

    # X = df.iloc[:, :-2].values
    # y = df.iloc[:, -1].values

    y = df[label_name].values
    X = df.drop(columns=[label_name]).values

    # removes all but the K highest scoring features using chi2, f_classif, mutual_info_classif
    all_statistics = {'f_cssif': SelectKBest(f_classif, k=k),
                      'mutual_info_classif': SelectKBest(mutual_info_classif, k=k),
                      # 'chi2': SelectKBest(chi2, k=k),
                      'f_regression': SelectKBest(f_regression, k=k)
                      }

    all_columns = {}
    X_df = df.drop(columns=[label_name])

    for statistics_name in all_statistics:
        stat_value = all_statistics[statistics_name].fit_transform(X, y)
        feature_idx = all_statistics[statistics_name].get_support()
        feature_names = X_df.columns[feature_idx]
        all_columns[statistics_name] = feature_names.tolist()

    # ///////////////      removes all but the SelectFromModel scoring features using.LinearSVC , LogisticRegression     ////////////
    all_models = {'LinearSVC': LinearSVC(C=0.15, penalty="l1", dual=False),
                  'lLogisticRegression': LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
                  'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=50)
                  }

    for model_name in all_models:
        model = all_models[model_name].fit(X, y)
        select_from_model = SelectFromModel(model, prefit=True)
        feature_idx = select_from_model.get_support()
        feature_names = X_df.columns[feature_idx]
        all_columns[model_name] = feature_names.tolist()

    # ///////////////     Creating a comparison matrix between variables    ////////////

    test_names = ['LinearSVC', 'lLogisticRegression', 'ExtraTreesClassifier',
                  'f_cssif', 'mutual_info_classif', 'chi2', 'f_regression']
    result_matrix_df = pd.DataFrame(columns=test_names, index=X_df.columns)

    for test_name in all_columns:
        result_matrix_df[test_name] = [1 if x in all_columns[test_name] else 0 for x in X_df.columns]
    result_matrix_df.loc[:, 'sum_score'] = result_matrix_df.sum(axis=1)

    # /*-----------------------------------------------------------------------------------------------------*/

    final_featre_names = []

    for feature_names in result_matrix_df.index.values.tolist():
        if result_matrix_df.loc[feature_names]['sum_score'] > score:
            final_featre_names.append(feature_names)


    return final_featre_names

# ///////////////     removes all but the Different methods like: PCA    ////////////
# Feature Extraction with PCA

# pca = PCA(n_components=3).fit(X,y)
# pca2 - PCA.transform(X,y)
