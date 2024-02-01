from typing import Any, Final, Optional, Iterable, Tuple, Union, List, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin


class BinningResult:
    def __init__(self, bins: Iterable[int], column: str, sign: bool):
        self.bins: Final[Iterable[int]] = bins
        self.column: Final[str] = column
        self.sign: Final[bool] = sign

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data[f'{self.column}_bins'] = pd.cut(data[self.column], self.bins, right=not self.sign, precision=0)
        data = data.drop(columns="bins")
        return data


class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, y: str, n_threshold: int, y_threshold: int, p_threshold: float, sign: bool = False):
        self.y: Final[str] = y
        self.n_threshold: Final[int] = n_threshold
        self.y_threshold: Final[int] = y_threshold
        self.p_threshold: Final[float] = p_threshold
        self.sign: Final[bool] = sign
        self.columns: Optional[List[str]] = None
        self.bins_summary_: Optional[Dict[str, pd.DataFrame]] = None
        self.woe_summary_: Optional[Dict[str, pd.DataFrame]] = None

    def initialize_summary(self, dataset: pd.DataFrame, column: str) -> pd.DataFrame:
        summary = dataset.groupby([column]).agg({self.y: ["mean", "size", "std"]})
        summary = summary.reset_index().droplevel(level=0, axis=1)
        summary.columns = [column, "means", "nsamples", "std_dev"]
        summary["del_flag"] = 0
        summary["std_dev"] = summary["std_dev"].fillna(0)
        return summary.sort_values([column], ascending=self.sign)

    def combine_bins(self, dataset: pd.DataFrame, column: str) -> pd.DataFrame:
        summary = self.initialize_summary(dataset, column)
        while True:
            i = 0
            summary = summary[summary.del_flag != 1]
            summary = summary.reset_index(drop=True)
            while True:
                j = i + 1
                if j >= len(summary):
                    break
                if summary.iloc[j].means < summary.iloc[i].means:
                    i = i + 1
                    continue
                else:
                    while True:
                        n = summary.iloc[j].nsamples + summary.iloc[i].nsamples
                        m = (summary.iloc[j].nsamples * summary.iloc[j].means +
                             summary.iloc[i].nsamples * summary.iloc[i].means) / n

                        if n == 2:
                            s = np.std([summary.iloc[j].means, summary.iloc[i].means])
                        else:
                            s = np.sqrt((summary.iloc[j].nsamples * (summary.iloc[j].std_dev ** 2) +
                                         summary.iloc[i].nsamples * (summary.iloc[i].std_dev ** 2)) / n)
                        summary.loc[i, "nsamples"] = n
                        summary.loc[i, "means"] = m
                        summary.loc[i, "std_dev"] = s
                        summary.loc[j, "del_flag"] = 1
                        j = j + 1
                        if j >= len(summary):
                            break
                        if summary.loc[j, "means"] < summary.loc[i, "means"]:
                            i = j
                            break
                if j >= len(summary):
                    break
            dels = np.sum(summary["del_flag"])
            if dels == 0:
                break

        return summary

    def calculate_pvalues(self, bin_summary: pd.DataFrame) -> pd.DataFrame:
        summary = bin_summary.copy()
        while True:
            summary["means_lead"] = summary["means"].shift(-1)
            summary["nsamples_lead"] = summary["nsamples"].shift(-1)
            summary["std_dev_lead"] = summary["std_dev"].shift(-1)

            summary["est_nsamples"] = summary["nsamples_lead"] + summary["nsamples"]
            summary["est_means"] = (summary["means_lead"] * summary["nsamples_lead"] +
                                    summary["means"] * summary["nsamples"]) / summary["est_nsamples"]

            summary["est_std_dev2"] = (summary["nsamples_lead"] * summary["std_dev_lead"] ** 2 +
                                       summary["nsamples"] * summary["std_dev"] ** 2) / (summary["est_nsamples"] - 2)

            summary["z_value"] = (summary["means"] - summary["means_lead"]) / np.sqrt(
                summary["est_std_dev2"] * (1 / summary["nsamples"] + 1 / summary["nsamples_lead"]))

            summary["p_value"] = 1 - stats.norm.cdf(summary["z_value"])

            summary["p_value"] = summary.apply(
                lambda row: row["p_value"] + 1 if (row["nsamples"] < self.n_threshold) |
                                                  (row["nsamples_lead"] < self.n_threshold) |
                                                  (row["means"] * row["nsamples"] < self.y_threshold) |
                                                  (row["means_lead"] * row["nsamples_lead"] < self.y_threshold)
                else row["p_value"], axis=1)

            max_p = max(summary["p_value"])
            row_of_maxp = summary['p_value'].idxmax()
            row_delete = row_of_maxp + 1

            if max_p > self.p_threshold:
                summary = summary.drop(summary.index[row_delete])
                summary = summary.reset_index(drop=True)
            else:
                break

            summary["means"] = summary.apply(lambda row: row["est_means"] if row["p_value"] == max_p else row["means"],
                                             axis=1)
            summary["nsamples"] = summary.apply(
                lambda row: row["est_nsamples"] if row["p_value"] == max_p else row["nsamples"], axis=1)
            summary["std_dev"] = summary.apply(
                lambda row: np.sqrt(row["est_std_dev2"]) if row["p_value"] == max_p else row["std_dev"], axis=1)

        return summary

    @staticmethod
    def calculate_woe(pvalue_summary: pd.DataFrame,
                      column: str) -> Tuple[Union[int, Iterable[int]], pd.DataFrame]:
        woe_summary = pvalue_summary[[column, "nsamples", "means"]]

        woe_summary["bads"] = woe_summary["means"] * woe_summary["nsamples"]
        woe_summary["goods"] = woe_summary["nsamples"] - woe_summary["bads"]

        total_goods = np.sum(woe_summary["goods"])
        total_bads = np.sum(woe_summary["bads"])

        woe_summary["dist_good"] = woe_summary["goods"] / total_goods
        woe_summary["dist_bad"] = woe_summary["bads"] / total_bads

        woe_summary[f'WOE_{column}'] = np.log(woe_summary["dist_good"] / woe_summary["dist_bad"])

        woe_summary["IV_components"] = ((woe_summary["dist_good"] - woe_summary["dist_bad"])
                                        * woe_summary[f'WOE_{column}'])

        total_iv = np.sum(woe_summary["IV_components"])

        return total_iv, woe_summary

    @staticmethod
    def generate_bin_labels(row: pd.DataFrame, column: str) -> str:
        return "-".join(map(str, np.sort([row[column], row[f'{column}_shift']])))

    def generate_final_dataset(self, dataset: pd.DataFrame, column: str, woe_summary: pd.DataFrame) -> BinningResult:
        shift_var = -1 if self.sign else 1

        woe_summary[str(column) + "_shift"] = woe_summary[column].shift(shift_var)

        if self.sign:
            woe_summary.loc[len(woe_summary) - 1, str(column) + "_shift"] = np.inf
            bins = np.sort(list(woe_summary[column]) + [np.Inf, -np.Inf])
        else:
            woe_summary.loc[0, str(column) + "_shift"] = -np.inf
            bins = np.sort(list(woe_summary[column]) + [np.Inf, -np.Inf])

        woe_summary["labels"] = woe_summary.apply(lambda x: self.generate_bin_labels(x, column), axis=1)

        dataset["bins"] = pd.cut(dataset[column], bins, right=not self.sign, precision=0)

        dataset["bins"] = dataset["bins"].astype(str)
        dataset["bins"] = dataset['bins'].map(lambda x: x.lstrip('[').rstrip(')'))

        return BinningResult(bins, column, self.sign)

    def fit(self, dataset: pd.DataFrame, y: Any = None) -> 'BinningTransformer':
        self.columns = dataset.columns.tolist()
        self.woe_summary_ = {}
        for column in self.columns:
            bins_summary = self.combine_bins(dataset, column)
            pvalues_summary = self.calculate_pvalues(bins_summary)
            _, woe_summary = self.calculate_woe(pvalues_summary, column)
            self.woe_summary_[column] = woe_summary
        return self

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            dataset = self.generate_final_dataset(dataset, column, self.woe_summary_[column]).transform(dataset)
        return dataset

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        output_features = []
        for column in self.columns:
            output_features.append(f'{column}_bins')
        return output_features

    def fit_transform_one_column(self, dataset: pd.DataFrame, column: str) -> 'BinningResult':
        bins_summary = self.combine_bins(dataset, column)
        pvalues_summary = self.calculate_pvalues(bins_summary)
        (_, woe_summary) = self.calculate_woe(pvalues_summary, column)
        return self.generate_final_dataset(dataset, column, woe_summary).transform(dataset)

