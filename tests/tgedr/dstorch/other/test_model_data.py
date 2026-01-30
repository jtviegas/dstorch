"""Module with example function unit test for description purposes."""
from pathlib import Path
import pytest
import pandas as pd

from tgedr.dstorch.other.model_data import BaseTokenizer, CsvLoader
from tgedr.dstorch.other.model_data import ModelData


def test_model_data(dataset_valuations_url):

    model_data: ModelData = ModelData.get_instance(data_url=dataset_valuations_url, 
                                        loader=CsvLoader(config=(["ticker", "company", "news"], ["y"])))
    assert 700 < model_data._x.shape[0] # type: ignore
    assert 700 < model_data._y.shape[0] # type: ignore
    assert 3 == model_data._x.shape[1] # type: ignore
    assert 1 == model_data._y.shape[1] # type: ignore
