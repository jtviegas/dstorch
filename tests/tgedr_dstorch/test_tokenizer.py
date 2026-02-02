import numpy as np
from tgedr_dstorch.tokenizer import BaseTokenizer


def test_base_tokenizer():

  SEQUENCE_MAX_LEN = 30000
  PADDING_TOKEN = "<PAD>"

  data : np.array = np.array([
    [
      "Apple is looking",
      "Autonomous cars"
    ],
    ["The Pope is visiting", 
     "Mary was reading a book"]
  ])
  tokenizer = BaseTokenizer()
  expected_data: np.array = np.array([
    [
      ['Apple', 'is', 'looking'] + [PADDING_TOKEN] * (SEQUENCE_MAX_LEN - 3),
      ['Autonomous', 'cars'] + [PADDING_TOKEN] * (SEQUENCE_MAX_LEN - 2)
    ],
    [
      ['The', 'Pope', 'is', 'visiting'] + [PADDING_TOKEN] * (SEQUENCE_MAX_LEN - 4), 
      ['Mary', 'was', 'reading', 'a', 'book'] + [PADDING_TOKEN] * (SEQUENCE_MAX_LEN - 5)
    ]
  ], dtype=object)  
  
  actual = tokenizer.tokenize(x=data, sequence_len=SEQUENCE_MAX_LEN, padding=PADDING_TOKEN)
  assert np.array_equal(actual, expected_data)
