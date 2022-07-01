from math import log, log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry_convert
import pycountry
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


# ----- DATA PREPROCESSING -----
def preprocess_dataset(path, MSC_level=1, categorial_rows=[], drop_rows=[]):
    """
    :categorial list: list of feature names that are treated as categories
    :drop list: list of feature names to remove from the data set
    """

    df = pd.read_csv(path, index_col=0, keep_default_na=False)

    # TODO: how to deal with those???
    df["author_country"] = df["author_country"].fillna("")
    df["author_journal_classification"] = df["author_journal_classification"].fillna("")
    df["author_continent"] = df["author_continent"].fillna("")
    
    # extract MSC at desired level
    if MSC_level == 1:
        df["author_classification"] = df["author_classification"].map(lambda c: c[0:2] if pd.notnull(c) else "")
    elif MSC_level == 2:
        df["author_classification"] = df["author_classification"].map(lambda c: c[0:3] if pd.notnull(c) else "")

    # handle categorial features
    for f in categorial_rows:
        try:
            df[f] = df[f].astype("category")
        except:
            pass

    # drop not required features
    for r in drop_rows:
        try:
            df = df.drop(r, axis=1)
        except:
            pass

    return df



# ----- CLUSTERING -----
PUBLICATION_COUNT_CLUSTERING = {
    "a) 1":     lambda x: x in [1],
    "b) 2":     lambda x: x in [2],
    "c) 3":     lambda x: x in [3],
    "d) 4":     lambda x: x in [4],
    "e) 5":     lambda x: x in [5],
    "f) 6":     lambda x: x in [6],
    "g) 7-8":   lambda x: x in [7,8],
    "h) 9-10":  lambda x: x in [9,10],
    "i) 11-13": lambda x: x in range(11, 14),
    "j) 14-18": lambda x: x in range(14, 19),
    "k) 19-29": lambda x: x in range(19, 30),
    "l) 30-50": lambda x: x in range(30, 51),
    "m) >50":   lambda x: x > 50
}


EGONET_1_CLUSTERING = {
    "PDE / Numerical / Physics":  lambda x: x in ['65', '78', '74', '76', '35', '86', '80'],
    "Real & complex functions":  lambda x: x in ['42', '28', '33', '44', '26', '41'],
    "ODE / Dynamical systems / Other equations":  lambda x: x in ['34', '37', '70', '45', '39'],
    "Probability / Statistics / Operations research":  lambda x: x in ['91', '62', '90', '60'],
    "Computer science / Information theory / Control":  lambda x: x in ['94', '68', '93', '92'],
    "General / Logic / Education":  lambda x: x in ['01', '03', '00', '97'],
    "Topology / Functional analysis":  lambda x: x in ['54', '46', '47', '40'],
    "Quantum / Relativity / Astronomy":  lambda x: x in ['81', '83', '82', '85'],
    "Number theory / Algebraic geometry":  lambda x: x in ['11', '13', '14', '12'],
    "Category theory / Algebraic topology": lambda x: x in ['55', '18', '57', '19'],
    "Algebra": lambda x: x in ['16', '17', '20'],
    "Function theory": lambda x: x in ['30', '31', '32'],
    "Graphs / Linear Algebra": lambda x: x in ['05', '15'],
    "Topological groups / Harmonical analysis": lambda x: x in ['43', '22'],
    "Order / Lattices": lambda x: x in ['06', '08'],
    "Geometry": lambda x: x in ['52', '51'],
    "Differential geometry": lambda x: x in ['58', '53'],
    "Calculus of variations": lambda x: x in ['49']
}


def find_cluster(clusters, value):
    for cluster_name, clustering_fn in clusters.items():
        if clustering_fn(value):
            return cluster_name


def cluster_column(df, clusters, column, newcolumn=None):
    """
    :clusters dict: (clustername,[cluster_values])
    :column str: Name of dataframe column to cluster
    :newcolumn str: If not None, use this new column for the result. Operates inplace instead.
    """
    if newcolumn is None:
        newcolumn = column
    df[newcolumn] = df.loc[:,column].map(lambda v: find_cluster(clusters, v)).astype("category")
    return df



# ----- DATA SET SAMPLING -----
def sampled_train_test_split(data, reference_test_data, sample_col, random_state=None):
    """Create a train/test split of the `data` DataFrame.
    This split has the following properties:
    * For each unique value of `sample_col`, the created `data_test` data set
      will have the same amount of data points as the input `reference_test_data` data set
    * Merging the created `data_train` and `data_test` data sets yields the input
      `data` data set
    :data:
    :reference_test_data:
    :sample_col str: Name of the column whose value counts are used to create the split
    """

    data_train = data.copy()
    data_test = pd.DataFrame()
    for val in np.unique(reference_test_data[sample_col]):
        print(val)
        t = data[data[sample_col] == val].sample(
                len(reference_test_data[reference_test_data[sample_col] == val]), random_state=random_state)
        data_test = pd.concat([data_test, t])
        data_train = data_train.append(t)
        data_train = data_train[~data_train.index.duplicated(keep=False)]
    return (data_train, data_test)


def X_y_split(data, target_column):
    return (data.drop(target_column, axis=1), data[target_column])



# ----- MODEL EVALUATION -----
def k_fold_eval(model, k, X, y, copy=[], fit=True,random_state=None):
    """Returns a list of (training-result, testing-result) tuples. Each result is a pd.DataFrame object.
    :model:
    :k int: number of folds to evaluate
    :X: input matrix
    :y: output vector
    :copy list: of row names to copy over from the feature input DataFrame
    """
    
    k_fold = KFold(k, shuffle=True, random_state=random_state) # add shuffle and random_state
    results = []
    
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        print(k)
        X_train, X_test = X.iloc[train].reset_index(drop=True), X.iloc[test].reset_index(drop=True)
        y_train, y_test = y[train].reset_index(drop=True), y[test].reset_index(drop=True)        
        
        if fit:
            model.fit(X_train, y_train)
        predictions_train = model.predict(X_train)
        predictions_test = model.predict(X_test)
        
        train_dict = { n: X_train[n] for n in copy }
        train_dict["y"] = y_train
        train_dict["y_hat"] = predictions_train
        
        test_dict = { n: X_test[n] for n in copy }
        test_dict["y"] = y_test
        test_dict["y_hat"] = predictions_test
        
        results.append((pd.DataFrame(train_dict), pd.DataFrame(test_dict)))
        # TODO: vielleich kopennte ich auch noch die jeweiligen Feature-Importances zurueck geben, wenn mehr als ein Modell gefittet wird
    return results


def gs_k_fold_eval(model, k, X, y, copy=[], feature_names=None, random_state=None, test_only=[]):
    """Returns a list of (training-result, testing-result) tuples. Each result is a pd.DataFrame object.
    :model:
    :k int: number of folds to evaluate
    :X: input matrix
    :y: output vector
    :copy list: of row names to copy over from the feature input DataFrame
    """

    k_fold = KFold(k, shuffle=True, random_state=random_state) # add shuffle and random_state
    results_train_test = []
    if feature_names is None:
        feature_importance_results = {}
    else:
        feature_importance_results = { n : [] for n in feature_names }
    results_test_only = []

    for k, (train, test) in enumerate(k_fold.split(X, y)):
        print("Working on fold {}".format(k+1))
        X_fold_train, X_fold_test = X.iloc[train].reset_index(drop=True), X.iloc[test].reset_index(drop=True)
        y_fold_train, y_fold_test = y[train].reset_index(drop=True), y[test].reset_index(drop=True)        

        model.fit(X_fold_train, y_fold_train)

        train_fold_dict = { n: X_fold_train[n] for n in copy }
        train_fold_dict["y"] = y_fold_train
        train_fold_dict["y_hat"] = model.predict(X_fold_train)

        test_fold_dict = { n: X_fold_test[n] for n in copy }
        test_fold_dict["y"] = y_fold_test
        test_fold_dict["y_hat"] = model.predict(X_fold_test)

        results_train_test.append((pd.DataFrame(train_fold_dict), pd.DataFrame(test_fold_dict)))

        if feature_names is not None:
            try:
                fold_feature_importance = zip(feature_names, model.feature_importances_)
            except AttributeError:
                fold_feature_importance = zip(feature_names, model.coef_)
            for fname, fimportance in fold_feature_importance:
                feature_importance_results[fname].append(fimportance)

        for i, (X_test_only, y_test_only) in enumerate(test_only):
            if len(X) != len(X_test_only):
                raise RuntimeException("Size of test only dataset (index {}) does not match size of training dataset!".format(i))
            if len(y) != len(y_test_only):
                raise RuntimeException("Size of test only results (index {}) does not match size of training results!".format(i))
            X_fold_test_only, y_fold_test_only = X_test_only.iloc[test].reset_index(drop=True), y_test_only[test].reset_index(drop=True)
            test_only_fold_dict = { n : X_fold_test_only[n] for n in copy }
            test_only_fold_dict["y"] = y_fold_test_only
            test_only_fold_dict["y_hat"] = model.predict(X_fold_test_only)
            results_test_only.append(pd.DataFrame(test_only_fold_dict))

    return (results_train_test, feature_importance_results, results_test_only)



def single_train_multi_test_eval(model, X_train, y_train, copy=[], fit=True, **test_data):
    """
    :test data: dict with entries of the form test_dataset_name: (X_test_dataset_name, y_test_dataset_name)
    """
    try:
        if fit:
            model.fit(X_train, y_train)
        predictions_train = model.predict(X_train)
        predictions = { name: model.predict(X_y[0]) for name, X_y in test_data.items() }
    except Exception as e:
        if fit:
            model.fit(X_train.toarray(), y_train)
        predictions_train = model.predict(X_train.toarray())
        predictions = { name: model.predict(X_y[0].toarray()) for name, X_y in test_data.items() }
    
    train_res = pd.DataFrame({
            **{ col: X_train[col] for col in copy},
            **{"y": y_train, "y_hat": predictions_train}
    })
    
    test_res = { name: pd.DataFrame({
            **{ col: X_y[0][col].reset_index(drop=True) for col in copy},
            **{"y": X_y[1].reset_index(drop=True), "y_hat": predictions[name]}
        }) for name, X_y in test_data.items()
    }
    
    return (train_res, test_res)


def data_stats(df, select_by, stats_for):
    """
    :select_by str: select these columns distinct values as x values
    :stats_for str: generate the statistics for the values in this column as y values
    """
    stats = []
    for sel in np.unique(df[select_by]):
        col = np.asarray(df[df[select_by] == sel][stats_for].values)
        stats.append([sel, col.shape[0], np.mean(col), np.std(col), np.median(col), np.percentile(col, 25), np.percentile(col, 75)])
    return pd.DataFrame(stats, columns=[select_by, "n", "mean", "stdev", "median", "p25", "p75"])



# ----- Gender-swapping -----

def stringify_gs_train_result(result):
    result_f_idx = result[result["x0_f"]==1].index
    result_m_idx = result[result["x0_m"]==1].index
    output = ""
    output = output + "Size:\t{}\n".format(len(result))
    output = output + "Size F:\t{}\n".format(len(result.loc[result_f_idx]))
    output = output + "Size M:\t{}\n".format(len(result.loc[result_m_idx]))
    output = output + "R2:\t{}\n".format(r2_score(result["y"], result["y_hat"]))
    output = output + "R2 F:\t{}\n".format(r2_score(result["y"][result_f_idx], result["y_hat"][result_f_idx]))
    output = output + "R2 M:\t{}\n".format(r2_score(result["y"][result_m_idx], result["y_hat"][result_m_idx]))
    output = output + "MAE:\t{}\n".format(mean_absolute_error(result["y"], result["y_hat"]))
    output = output + "MAE F:\t{}\n".format(mean_absolute_error(result["y"][result_f_idx], result["y_hat"][result_f_idx]))
    output = output + "MAE M:\t{}\n".format(mean_absolute_error(result["y"][result_m_idx], result["y_hat"][result_m_idx]))
    output = output + "Mean y:\t{}\n".format(np.mean(result["y"]))
    output = output + "Mean y F:\t{}\n".format(np.mean(result["y"][result_f_idx]))
    output = output + "Mean y M:\t{}\n".format(np.mean(result["y"][result_m_idx]))
    output = output + "Mean y_hat:\t{}\n".format(np.mean(result["y_hat"]))
    output = output + "Mean y_hat F:\t{}\n".format(np.mean(result["y_hat"][result_f_idx]))
    output = output + "Mean y_hat M:\t{}\n".format(np.mean(result["y_hat"][result_m_idx]))
    return output



def compare_gs_results(result, switched_result, n_bins=10):
    r_idx_f = result[result["x0_f"]==1].index
    r_idx_m = result[result["x0_m"]==1].index
    r_sw_idx_f = switched_result[switched_result["x0_f"]==1].index
    r_sw_idx_m = switched_result[switched_result["x0_m"]==1].index
    
    output = ""
    output = output + "Size:\t{}\t\t{}\n".format(len(result), len(switched_result))
    output = output + "Size F:\t{}\t\t{}\n".format(len(result.loc[r_idx_f]), len(switched_result.loc[r_sw_idx_f]))
    output = output + "Size M:\t{}\t\t{}\n".format(len(result.loc[r_idx_m]), len(switched_result.loc[r_sw_idx_m]))
    output = output + "R2:\t{}\t{}\n".format(r2_score(result["y"], result["y_hat"]), r2_score(switched_result["y"], switched_result["y_hat"]))
    output = output + "R2 F:\t{}\t{}\n".format(r2_score(result["y"][r_idx_f], result["y_hat"][r_idx_f]), r2_score(switched_result["y"][r_sw_idx_f], switched_result["y_hat"][r_sw_idx_f]))
    output = output + "R2 M:\t{}\t{}\n".format(r2_score(result["y"][r_idx_m], result["y_hat"][r_idx_m]), r2_score(switched_result["y"][r_sw_idx_m], switched_result["y_hat"][r_sw_idx_m]))
    output = output + "MAE:\t{}\t{}\n".format(mean_absolute_error(result["y"], result["y_hat"]), mean_absolute_error(switched_result["y"], switched_result["y_hat"]))
    output = output + "MAE F:\t{}\t{}\n".format(mean_absolute_error(result["y"][r_idx_f], result["y_hat"][r_idx_f]), mean_absolute_error(switched_result["y"][r_sw_idx_f], switched_result["y_hat"][r_sw_idx_f]))
    output = output + "MAE M:\t{}\t{}\n".format(mean_absolute_error(result["y"][r_idx_m], result["y_hat"][r_idx_m]), mean_absolute_error(switched_result["y"][r_sw_idx_m], switched_result["y_hat"][r_sw_idx_m]))
    output = output + "Mean y F:\t{}\t{}\n".format(np.mean(result["y"][r_idx_f]), np.mean(switched_result["y"][r_sw_idx_f]))
    output = output + "Mean y M:\t{}\t{}\n".format(np.mean(result["y"][r_idx_m]), np.mean(switched_result["y"][r_sw_idx_m]))
    output = output + "Mean y_hat F:\t{}\t{}\n".format(np.mean(result["y_hat"][r_idx_f]), np.mean(switched_result["y_hat"][r_sw_idx_f]))
    output = output + "Mean y_hat M:\t{}\t{}\n".format(np.mean(result["y_hat"][r_idx_m]), np.mean(switched_result["y_hat"][r_sw_idx_m]))

    intervals = pd.qcut(result["y_hat"][r_idx_f].values, n_bins, duplicates="drop")
    output = output + "WD (F, S(F)=M):\t{}\n".format(wasserstein_distance(result["y_hat"][r_idx_f], switched_result["y_hat"][r_sw_idx_m]))
    output = output + "WD (F, S(M)=F):\t{}\n".format(wasserstein_distance(result["y_hat"][r_idx_f], switched_result["y_hat"][r_sw_idx_f]))
    intervals = pd.qcut(result["y_hat"][r_idx_m].values, n_bins, duplicates="drop")
    output = output + "WD (M, S(M)=F):\t{}\n".format(wasserstein_distance(result["y_hat"][r_idx_m], switched_result["y_hat"][r_sw_idx_f]))
    output = output + "WD (M, S(F)=M):\t{}\n".format(wasserstein_distance(result["y_hat"][r_idx_m], switched_result["y_hat"][r_sw_idx_m]))
    output = output + "\n"
    return output 


def plot_gs_results(result, switched_result, x, y, subtitle, x_categories=None):
    test_result = result
    test_result_sw = switched_result

    f_sdf_test = data_stats(test_result[test_result["x0_f"]==1], x, y)
    m_sdf_test = data_stats(test_result[test_result["x0_m"]==1], x, y)
    f_sdf_test_sw = data_stats(test_result_sw[test_result_sw["x0_f"]==1], x, y)
    m_sdf_test_sw = data_stats(test_result_sw[test_result_sw["x0_m"]==1], x, y)
    f_sdf_test = pd.merge(f_sdf_test, m_sdf_test_sw, on=x, suffixes=("", "_switched"))
    m_sdf_test = pd.merge(m_sdf_test, f_sdf_test_sw, on=x, suffixes=("", "_switched"))
    
    if x_categories is not None:
        f_sdf_test[x] = pd.Categorical(f_sdf_test[x], categories=x_categories, ordered=True)
        m_sdf_test[x] = pd.Categorical(m_sdf_test[x], categories=x_categories, ordered=True)

    fig, (ax_f, ax_m) = plt.subplots(1, 2, figsize=(20, 5))
    ax_f.set_title("Female", fontsize=16)
    ax_f.plot(f_sdf_test[x], f_sdf_test["mean"], "-", label="mean")
    ax_f.plot(f_sdf_test[x], f_sdf_test["mean_switched"], "-", label="switched mean")
    ax_f.fill_between(f_sdf_test[x], f_sdf_test["p25"], f_sdf_test["p75"], alpha=0.2)
    ax_f.legend(["mean", "switched_mean"])

    ax_m.set_title("Male", fontsize=16)
    ax_m.plot(m_sdf_test[x], m_sdf_test["mean"], "-", label="mean")
    ax_m.plot(m_sdf_test[x], m_sdf_test["mean_switched"], "-", label="switched mean")
    ax_m.fill_between(m_sdf_test[x], m_sdf_test["p25"], m_sdf_test["p75"], alpha=0.2)
    ax_m.legend(["mean", "switched_mean"])

    fig.suptitle(subtitle, fontsize=20)
    return fig


# ---- Male-Baseline ----

def stringify_mb_result(result, heading):
    r = heading + "\n"
    r = r + "\tSize: {}\n".format(len(result))
    r = r + "\tr2 score: {}\n".format(r2_score(result["y"], result["y_hat"]))
    r = r + "\tMAE: {}\n".format(mean_absolute_error(result["y"], result["y_hat"]))
    r = r + "\tMean y_hat: {}\n".format(np.mean(result["y_hat"]))
    r = r + "\tMedian y_hat: {}\n".format(np.median(result["y_hat"]))
    r = r + "\tMean y / y_hat: {}\n".format(np.mean(result["y/y_hat"]))
    r = r + "\tMedian y / y_hat: {}\n".format(np.median(result["y/y_hat"]))
    r = r + "\tMean y_hat - y: {}\n".format(np.mean(result["y_hat-y"]))
    r = r + "\tMedian y_hat - y: {}\n".format(np.median(result["y_hat-y"]))
    r = r + "\tMean y_hat + y: {}\n".format(np.mean(result["y_hat+y"]))
    r = r + "\tMedian y_hat + y: {}\n".format(np.median(result["y_hat+y"]))
    r = r + "\tMean (y_hat - y) \ (y_hat + y): {}\n".format(np.mean(result["(y_hat-y)/(y_hat+y)"]))
    r = r + "\tMedian (y_hat - y) \ (y_hat + y): {}\n".format(np.median(result["(y_hat-y)/(y_hat+y)"]))
    return r


def compare_mb_results(train_result, test_results):
    r = stringify_mb_result(train_result, "Training Dataset")
    for name, result in test_results.items():
        r = r + stringify_mb_result(result, "Test dataset: {}".format(name))
    return r

def combine_mb_gender_predictions(test_results, select_by):
    y_comb = {n: data_stats(r, select_by, "y") for n,r in test_results.items()}
    y_hat_comb = {n: data_stats(r, select_by, "y_hat") for n,r in test_results.items()}

    return {
        "y": y_comb["female"].merge(y_comb["male"], on=select_by, suffixes=("_female", "_male"), how="outer"),
        "y_hat": y_hat_comb["female"].merge(y_hat_comb["male"], on=select_by, suffixes=("_female", "_male"), how="outer")
    }


def plot_mb_test_results(test_results, x, y, aggregator="mean", axis_callbacks=None, plot_options=None):
    """
    :aggregator str: which statistic aggregation function to use. either 'mean' or 'median'.
    :axis callbacks list: list of functions to be executed on the axis. The axis is passed as single argument and can be adjusted this way
    """
    if type(y) == str:
        y = [y]
    if axis_callbacks is None:
        axis_callbacks = []
    if plot_options is None:
        plot_options = {}

    fig, axes = plt.subplots(1, len(test_results), figsize=(20,5))
    for i,name in enumerate(test_results.keys()):
        stats = { y_col: data_stats(test_results[name], x, y_col) for y_col in y }
        try:
            ax = axes[i]
        except TypeError:
            ax = axes
        for cb in axis_callbacks:
            cb(ax)
        ax.set_title(name, fontsize=16)
        for col, s in stats.items():
            ax.plot(s[x], s[aggregator], label=col)
        ax.fill_between(stats[y[0]][x], stats[y[0]]["p25"], stats[y[0]]["p75"], alpha=0.2)
        ax.legend()
    fig.suptitle("x: {}; y: {}; statistical aggregator: {}".format(x, y, aggregator), fontsize=20)
    return fig
        
    

# ----- GEO DATA -----
def find_continent(country):
    if not country:
        return ""
    elif country == "VA":
        return "EU"
    elif country == "UM":
        return "NA"
    elif len(country) == 2:
        a2 = pycountry.countries.get(alpha_2=country).alpha_2
        return pycountry_convert.country_alpha2_to_continent_code(a2)
    return ""


def country_processing(df, country_col="country", importance_threshold=1, add_continents=True, continent_col="continent"):
    """
    :importance_threshold int: indicator up to which stacked importance the countries will be thinned out. 0-remove all countries, 1-keep all countries
    """
    print("Number of unique countries in initial data set", len(df[country_col].unique()))
    df.loc[:,country_col] = df[country_col].replace({"UK": "GB", "KO": "KR"}) # replace known faulty codes
    
    for c in df[country_col].unique():
        if c:
            try:
                country = pycountry.countries.get(alpha_2=c)
            except:
                print("Unknown country code: ", c)
    
    if add_continents:
        df.loc[:,continent_col] = df[country_col].map(find_continent)
    
    cumsum = df[df[country_col] != ""][country_col].value_counts(normalize=True).cumsum()
    df[country_col] = df[country_col].map(
            lambda c: c if c in cumsum[cumsum <= importance_threshold].index.values else "Other")
    print("Number of unique countries in processed data set", len(df[country_col].unique()))
