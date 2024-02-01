from collections.abc import Generator
from datetime import datetime
from typing import List, Optional, Iterable, Union
from sklearn.compose import make_column_transformer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.woe_bin import BinningTransformer

plt.style.use('ggplot')


def generate_date_intervals(start_date: datetime,
                            end_date: datetime,
                            number_of_intervals: int) -> Generator[datetime]:
    """
    Splits date into intervals

    :param start_date: datetime - date from
    :param end_date: datetime- date to
    :param number_of_intervals: int - num of intervals
    :return: yield data range
    """
    diff = (end_date - start_date) / number_of_intervals
    for i in range(number_of_intervals):
        yield start_date + diff * i
    yield end_date


def draw_graph(result: dict, title: str, type_data: str = 'Risk'):
    """
    Plots diagrams

    :param result: dict - key: column value, values: values in periods
    :param title: string - title of plot
    :param type_data: string - label y axis
    """
    plt.figure(figsize=(16, 6))
    for key, value in result.items():
        plt.plot(list(value.keys()), list(value.values()), 'o-', label=key)
        for a, b in zip(list(value.keys()), list(value.values())):
            plt.text(a, b, str(np.round(b, 3)), fontsize=10, bbox=dict(facecolor='white', alpha=0.75))
    plt.title(title)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Date', fontsize=13)
    plt.ylabel(type_data, fontsize=13)
    if type_data != "Num of bads":
        plt.legend(fontsize=11)
    plt.show()


def plot_risk_categorical(df: pd.DataFrame,
                          target: str,
                          columns_to_show: Optional[List[str]] = None,
                          date_column: Optional[str] = None,
                          bad: Union[int, str] = 1, data_bin_n: int = 4, bin_by_month: bool = False):
    """
    Plots risk diagrams of categorical columns by date

    :param df: pd.DataFrame - input data
    :param target: string - targets column name
    :param columns_to_show: list - column names to show
    :param date_column: string - date column name
    :param bad: int or string - value of incidents in target column
    :param data_bin_n: int - number of data bins, data is split for equal values in bins
    :param bin_by_month: bool - if True month type of binning
    """
    if columns_to_show is None:
        columns_to_show = __get_numerical_columns(df, exclude=[target])

    if date_column is None:
        date_column = __get_date_columns(df, exclude=[target])[0]

    df = df.sort_values(by=date_column)
    result_nums = {}
    for column in columns_to_show:
        print_general_column_statistics(df, target, column, bad)

        result = {}
        result_share = {}
        for unique_value_in_column in df[column].unique():
            value_data = df[df[column] == unique_value_in_column]
            if bin_by_month:
                date_intervals = list(pd.date_range(min(df[date_column].dt.date), max(df[date_column].dt.date),
                                                    freq='M'))
            else:
                date_intervals = generate_date_intervals(min(df[date_column]), max(df[date_column]), data_bin_n)

            period = {}
            period_share = {}
            period_nums = {}

            for date_index, date in enumerate(date_intervals):
                if date_index == 0:
                    value_data_in_date_range = value_data[value_data[date_column] < date]
                    all_data_up_to_current_date = df[df[date_column] < date]
                else:
                    value_data_in_date_range = value_data[(value_data[date_column] >= date_intervals[date_index - 1])
                                                          & (value_data[date_column] < date)]
                    all_data_up_to_current_date = df[(df[date_column] >= date_intervals[date_index - 1])
                                                     & (df[date_column] < date)]

                current_result = 0 if value_data_in_date_range.empty else (
                        value_data_in_date_range[value_data_in_date_range[target] == bad].shape[0] /
                        value_data_in_date_range.shape[0]
                )

                period[date.strftime('%d-%m-%Y')] = current_result

                if all_data_up_to_current_date.shape[0] == 0:
                    current_result = 0
                    num_res = 0
                else:
                    current_result = (
                            all_data_up_to_current_date[all_data_up_to_current_date[column] == unique_value_in_column]
                            .shape[0] / all_data_up_to_current_date.shape[0]
                    )
                    num_res = all_data_up_to_current_date[all_data_up_to_current_date[target] == bad].shape[0]

                period_share[date.strftime('%d-%m-%Y')] = current_result
                period_nums[date.strftime('%d-%m-%Y')] = num_res

            result[unique_value_in_column] = period
            result_share[unique_value_in_column] = period_share
            result_nums[unique_value_in_column] = period_nums

        draw_graph(result, f"{column} risk")
        draw_graph(result_share, f"{column} share", "Share")

    draw_graph(result_nums, "Number of incidents", "Num of bads")


def print_general_column_statistics(df: pd.DataFrame, target: str, column: str, bad: Union[str, int]):
    print(f"{column} risks stats")
    stats = pd.DataFrame(data={"values": df[column].value_counts(),
                               "num_incidents": df[df[target] == bad][column].value_counts(),
                               "risk": df[df[target] == bad][column].value_counts() / df[column].value_counts()})
    print(stats.to_string())


def plot_risk_continuous(df: pd.DataFrame, target: str, columns_to_show: List[str] = None, date_column: str = None,
                         bad: int = 1, data_bin_n: int = 4, bin_by_month: bool = False, n_threshold: int = 50,
                         y_threshold: int = 10, p_threshold: float = 0.35, sign: bool = False):
    """
    Plots risk diagrams of continuous columns by date

    :param df: pd.DataFrame - input data
    :param target: string - targets column name
    :param columns_to_show: list - column names to show
    :param date_column: string - date column name
    :param bad: int or string - value of incidents in target column
    :param data_bin_n: int - number of data bins, data is split for equal values in bins
    :param bin_by_month: bool - if True month type of binning

    :param n_threshold: int - maximum num of values in one bin
    :param y_threshold: int - maximum num of bads in one value
    :param p_threshold: float - threshold for stop binning
    :param sign: bool - if True, shift final dataset from binning
    """
    if columns_to_show is None:
        columns_to_show = __get_numerical_columns(df, exclude=[target])

    if date_column is None:
        date_column = __get_date_columns(df, exclude=[target])[0]
    columns_to_show += [target]
    if date_column in columns_to_show:
        columns_to_show.remove(date_column)
    bin_object = make_column_transformer(
            (BinningTransformer(target, n_threshold=30, y_threshold=10, p_threshold=0.3, sign=False), columns_to_show),
            remainder="passthrough",
            verbose_feature_names_out=False
    )
    res = bin_object.fit_transform(df)
    new_df = pd.DataFrame(res, columns=columns_to_show + list(bin_object.get_feature_names_out()))
    new_df = new_df.drop(columns=f'{target}_bins')
    new_columns_to_show = [f for f in new_df.columns if "_bins" in f]

    plot_risk_categorical(new_df, target, new_columns_to_show, date_column, bad, data_bin_n, bin_by_month)


def __get_date_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]]) -> List[str]:
    return [
        column
        for column in df.columns
        if __is_date_type(df[column].dtype) and (exclude is None or column not in exclude)
    ]


def __get_numerical_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]]) -> List[str]:
    return [
        column
        for column in df.columns
        if __is_numeric_type(df[column].dtype) and (exclude is None or column not in exclude)
    ]


def __is_date_type(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.datetime64)


def __is_numeric_type(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.number)

