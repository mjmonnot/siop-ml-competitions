import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src.config import TARGETS, TEXT_COLS, N_SPLITS, RANDOM_STATE
from src.io import load_data, split_by_dataset
from src.text import concat_text
from src.features import make_vectorizer
from src.models import make_model
from src.evaluation import cross_val_mean_r, summarize_cv


def main(data_path):
    df = load_data(data_path)
    train_df, dev_df, test_df = split_by_dataset(df)

    X_text = concat_text(train_df, TEXT_COLS)

    vectorizer = make_vectorizer()
    X = vectorizer.fit_transform(X_text)

    cv_results = {}
    oof_preds = {}

    for t in TARGETS:
        y = train_df[t].values
        model = make_model(RANDOM_STATE)

        r, oof = cross_val_mean_r(
            X, y, model, N_SPLITS, RANDOM_STATE
        )
        cv_results[t] = r
        oof_preds[t] = oof

    summary = summarize_cv(cv_results)

    # Save CV summary
    cv_df = summary["per_trait_r"].to_frame("mean_r")
    cv_df.loc["MEAN"] = summary["mean_r"]
    cv_df.to_csv("results/cv/cv_summary.csv")

    # Plot
    cv_df.drop("MEAN").plot(kind="bar", legend=False)
    plt.title("2019 CV Pearson r by Trait")
    plt.ylabel("r")
    plt.tight_layout()
    plt.savefig("results/figures/cv_r_by_trait.png")

    # Train full models + predict dev/test
    X_full = vectorizer.transform(concat_text(train_df, TEXT_COLS))
    X_dev = vectorizer.transform(concat_text(dev_df, TEXT_COLS))
    X_test = vectorizer.transform(concat_text(test_df, TEXT_COLS))

    dev_out = dev_df[["Respondent_ID"]].copy()
    test_out = test_df[["Respondent_ID"]].copy()

    for t in TARGETS:
        model = make_model(RANDOM_STATE)
        model.fit(X_full, train_df[t].values)
        dev_out[t.replace("_Scale_score", "_Pred")] = model.predict(X_dev)
        test_out[t.replace("_Scale_score", "_Pred")] = model.predict(X_test)

    dev_out.to_csv("results/submissions/submission_dev.csv", index=False)
    test_out.to_csv("results/submissions/submission_test.csv", index=False)

    print("CV mean r:", summary["mean_r"])
    print(summary["per_trait_r"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    main(args.data_path)
