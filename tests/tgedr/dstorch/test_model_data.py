"""Module with example function unit test for description purposes."""
from pathlib import Path
import pytest
import pandas as pd

from tgedr.dstorch.model_data import BaseTokenizer, CsvLoader
from tgedr.dstorch.model_data import ModelData


@pytest.fixture
def dataset_valuations_url(resources_folder) -> str:
    return str(Path(resources_folder) / "valuations.csv")

def test_csv_loader(dataset_valuations_url):
    ds = CsvLoader(config=(["ticker", "company", "news"], ["y"])).load(url=dataset_valuations_url)
    assert ds is not None
    assert 700 < ds[0].shape[0]
    assert 3 == ds[0].shape[1]
    assert 700 < ds[1].shape[0]
    assert 1 == ds[1].shape[1]

def test_model_data(dataset_valuations_url):

    model_data: ModelData = ModelData.get_instance(data_url=dataset_valuations_url, 
                                        loader=CsvLoader(config=(["ticker", "company", "news"], ["y"])))
    assert 700 < model_data._x.shape[0] # type: ignore
    assert 700 < model_data._y.shape[0] # type: ignore
    assert 3 == model_data._x.shape[1] # type: ignore
    assert 1 == model_data._y.shape[1] # type: ignore

def test_base_tokenizer(dataset_valuations_url):
    model_data: ModelData = ModelData.get_instance(data_url=dataset_valuations_url, 
                                        loader=CsvLoader(config=(["ticker", "company", "news"], ["y"])))
    
    tokenizer = BaseTokenizer()
    tokenizer.tokenize(x=model_data._x) # type: ignore