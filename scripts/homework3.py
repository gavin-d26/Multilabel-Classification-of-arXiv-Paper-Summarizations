import argparse
from itertools import product
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    accuracy_score,
)
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

HP_TUNE = False
SAVE_PREPROCS = False

# paths to save the preprocessors
pretrained_vectorizer_path = "vectorizer.pkl"
pretrained_pca_path = "pca.pkl"
pretrained_scaler_path = "scaler.pkl"


def produce_target_columns(df):
    outs = df.explode("terms")
    outs["values"] = 1
    outs["new_index"] = outs.index
    outs = outs.pivot(index=["new_index"], columns="terms", values="values")
    outs.fillna(0, inplace=True)
    df = pd.concat([df, outs], axis=1)
    return df


def clean_sentence_text(series):
    series = series.str.strip()
    series = series.str.lower()
    series = series.apply(lambda x: "".join([item for item in x if not item.isdigit()]))
    return series


def write_metrics(
    model_name,
    test_preds,
    test_targets,
    val_preds,
    val_targets,
    train_time,
    inf_time,
    results_file_obj,
    print_test=False,
):
    print("-" * 50, file=results_file_obj)
    print(f"{model_name}", file=results_file_obj)
    print("", file=results_file_obj)

    # validation set
    print("Validation set", file=results_file_obj)
    print(
        f"Validation F1 Macro: {f1_score(val_targets, val_preds, average='macro')}",
        file=results_file_obj,
    )
    print(
        f"Validation F1 Micro: {f1_score(val_targets, val_preds, average='micro')}",
        file=results_file_obj,
    )
    # print(f"Validation Accuracy: {accuracy_score(val_targets, val_preds)}", file=results_file_obj)
    print("Validation Classification Report", file=results_file_obj)
    print(classification_report(val_targets, val_preds), file=results_file_obj)
    print("", file=results_file_obj)
    print(f"training time: {train_time}", file=results_file_obj)
    print(f"inference time: {inf_time}", file=results_file_obj)
    # test set
    if print_test:
        print("Test set", file=results_file_obj)
        print(
            f"Test F1 Macro: {f1_score(test_targets, test_preds, average='macro')}",
            file=results_file_obj,
        )
        print(
            f"Test F1 Micro: {f1_score(test_targets, test_preds, average='micro')}",
            file=results_file_obj,
        )
        # print(f"Test Accuracy: {accuracy_score(test_targets, test_preds)}", file=results_file_obj)
        print("Test Classification Report", file=results_file_obj)
        print(classification_report(test_targets, test_preds), file=results_file_obj)
        print(f"training time: {train_time}", file=results_file_obj)
        print("", file=results_file_obj)


if __name__ == "__main__":

    # parse two --data and --output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset_path = args.data
    output_path = args.output
    results_file_obj = open(output_path, "w")

    # Load the dataset
    df = pd.read_json(dataset_path)
    df = produce_target_columns(df)

    train_set, valtest = train_test_split(df, test_size=0.30, random_state=1234)
    val_set, test_set = train_test_split(valtest, test_size=0.50, random_state=1234)
    # This will give you a 70/15/15 split.
    print(len(train_set), len(val_set), len(test_set))

    # select classes with >1% samples
    class_percents = train_set.iloc[:, -88:].sum(axis=0) / len(train_set) * 100
    non_zero_classes = class_percents > 1
    num_classes = non_zero_classes.astype(int).sum()
    train_set = pd.concat(
        (
            train_set.loc[:, ["titles", "summaries"]],
            train_set.iloc[:, -88:].loc[:, non_zero_classes],
        ),
        axis=1,
    )
    val_set = pd.concat(
        (
            val_set.loc[:, ["titles", "summaries"]],
            val_set.iloc[:, -88:].loc[:, non_zero_classes],
        ),
        axis=1,
    )
    test_set = pd.concat(
        (
            test_set.loc[:, ["titles", "summaries"]],
            test_set.iloc[:, -88:].loc[:, non_zero_classes],
        ),
        axis=1,
    )

    # create preprocessors for the model

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=2000, ngram_range=(1, 2)
    )
    vectorizer.fit(clean_sentence_text(train_set.summaries + " " + train_set.titles))

    # if SAVE_PREPROCS:
    #     pickle.dump(vectorizer, open(pretrained_vectorizer_path, "wb"))

    # inputs = vectorizer.transform(
    #     clean_sentence_text(train_set.summaries + " " + train_set.titles)
    # ).toarray()

    # # select the best features
    # if pretrained_pca_path is not None and SAVE_PREPROCS is False:
    #     selector = pickle.load(open(pretrained_pca_path, "rb"))
    # else:
    #     selector = PCA(n_components=250)
    #     selector.fit(inputs)

    # if SAVE_PREPROCS:
    #     pickle.dump(selector, open(pretrained_pca_path, "wb"))

    # selected_inputs = selector.transform(inputs)

    # # scale the features between 0 and 1
    # if pretrained_scaler_path is not None and SAVE_PREPROCS is False:
    #     scaler = pickle.load(open(pretrained_scaler_path, "rb"))
    # else:
    #     scaler = MinMaxScaler()
    #     scaler.fit(selected_inputs)

    # if SAVE_PREPROCS:
    #     pickle.dump(scaler, open(pretrained_scaler_path, "wb"))

    # scaled_inputs = scaler.transform(selected_inputs)
    # scaled_inputs[scaled_inputs < 0] = 0

    def preprocess_data(df):
        inputs = vectorizer.transform(
            clean_sentence_text(df.summaries + " " + df.titles)
        ).toarray()
        return inputs, df.iloc[:, -num_classes:].values

    models = {
        # "NaiveBayes": MultinomialNB,
        "LinearSVC": LinearSVC,
        # "DecisionTree": DecisionTreeClassifier,
        # "LogisticRegression": LogisticRegression,
        # "KNeighbors": KNeighborsClassifier,
    }

    X_train, Y_train = preprocess_data(train_set)
    X_val, Y_val = preprocess_data(val_set)
    X_test, Y_test = preprocess_data(test_set)

    if HP_TUNE:

        param_grids = {
            "LinearSVC": [
                {"C": C, "penalty": penalty}
                for C, penalty, in product([0.01, 0.1, 1, 10], ["l1", "l2"])
            ],
            "DecisionTree": [
                {
                    "criterion": criterion,
                    "max_depth": depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }
                for criterion, depth, min_samples_split, min_samples_leaf in product(
                    ["gini", "entropy", "log_loss"],
                    [120, None],
                    [2, 10, 20, 50],
                    [1, 5, 10],
                )
            ],
            "LogisticRegression": [
                {"C": C, "penalty": penalty, "solver": solver, "max_iter": max_iter}
                for C, penalty, solver, max_iter in product(
                    [0.01, 0.1, 1, 10],
                    ["l2", "elasticnet", "none"],
                    ["liblinear", "saga", "newton-cg"],
                    [100, 200, 500],
                )
            ],
            "NaiveBayes": [
                {"alpha": alpha, "fit_prior": fit_prior}
                for alpha, fit_prior in product([0.1, 0.5, 1.0, 2.0], [True, False])
            ],
        }
        # custom grid search with validation set
        best_models = {}

        for model_name, model_class in models.items():
            print(f"Tuning hyperparameters for {model_name}...", file=results_file_obj)
            best_score = -np.inf
            best_params = None
            best_model = None

            # loop through all hyperparameter combinations
            for params in param_grids[model_name]:
                # Set model parameters
                try:
                    est = model_class(**params)
                    model = OneVsRestClassifier(est)

                    # Train the model on the training data
                    model.fit(X_train, Y_train)
                except:
                    continue

                # Evaluate on the validation set
                y_pred = model.predict(X_val)
                score = f1_score(Y_val, y_pred, average="micro")

                # track the best score and parameters
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model

            # store the best model and its details
            best_models[model_name] = (best_model, best_params, best_score)
            print(
                f"Best Parameters for {model_name}: {best_params}",
                file=results_file_obj,
            )
            print(
                f"Validation F1 for {model_name}: {best_score:.4f}",
                file=results_file_obj,
            )
            # print(f"Classification Report for {model_name}:\n",
            #     classification_report(Y_val, best_model.predict(X_val)), file=results_file_obj)
            print("-" * 80, file=results_file_obj)

        # summary of the best models
        print("Summary of Best Models:", file=results_file_obj)
        for model_name, (model, params, score) in best_models.items():
            print(
                f"{model_name}: Best Score = {score:.4f}, Best Params = {params}",
                file=results_file_obj,
            )

        results_file_obj.close()
    else:

        tuned_params = {
            "LinearSVC": {"C": 1, "penalty": "l2"},
        }

        for model_name, model_class in models.items():

            print(f"Training {model_name}...")

            # train and evaluate the defualt HP model
            model = OneVsRestClassifier(model_class(**tuned_params[model_name]))
            start = time.time()
            model.fit(X_train, Y_train)
            elapsed = time.time() - start
            start = time.time()
            val_preds = model.predict(X_val)
            infelapsed = time.time() - start
            test_preds = model.predict(X_test)
            write_metrics(
                model_name,
                test_preds,
                Y_test,
                val_preds,
                Y_val,
                elapsed,
                infelapsed,
                results_file_obj,
                print_test=True,
            )

    results_file_obj.close()
