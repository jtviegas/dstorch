from tgedr.dstorch.other.model_data import CsvLoader


def test_csv_loader(dataset_valuations_url):
    ds = CsvLoader(config=(["ticker", "company", "news"], ["y"])).load(url=dataset_valuations_url)
    assert ds is not None
    assert 700 < ds[0].shape[0]
    assert 3 == ds[0].shape[1]
    assert 700 < ds[1].shape[0]
    assert 1 == ds[1].shape[1]