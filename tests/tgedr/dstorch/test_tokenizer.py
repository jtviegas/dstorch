



import numpy as np
from tgedr.dstorch import tokenizer
from tgedr.dstorch.tokenizer import BaseTokenizer


def test_base_tokenizer():

  SEQUENCE_MAX_LEN = 30000

  data : np.array = np.array([
    [
      "Apple is looking",
      "Autonomous cars"
    ],
    ["The Pope is visiting", 
     "Mary was reading a book"]
  ])
  expected_data: np.array = np.array([
    [
      ['Apple', 'is', 'looking'] + [BaseTokenizer._EMPTY_VAL] * (SEQUENCE_MAX_LEN - 3),
      ['Autonomous', 'cars'] + [BaseTokenizer._EMPTY_VAL] * (SEQUENCE_MAX_LEN - 2)
    ],
    [
      ['The', 'Pope', 'is', 'visiting'] + [BaseTokenizer._EMPTY_VAL] * (SEQUENCE_MAX_LEN - 4), 
      ['Mary', 'was', 'reading', 'a', 'book'] + [BaseTokenizer._EMPTY_VAL] * (SEQUENCE_MAX_LEN - 5)
    ]
  ], dtype=object)  
  
  tokenizer = BaseTokenizer()
  actual = tokenizer.tokenize(x=data)
  assert np.array_equal(actual, expected_data)
