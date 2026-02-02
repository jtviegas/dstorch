import numpy as np
from tgedr_dstorch.vocabulary import Vocabulary, VocabularyBuilder


def test_flatten_text_lists():

    builder = VocabularyBuilder()
    nested_texts = [
        "hello",
        ["world", ["this", "is"], "a", "test"],
        [["of", "the"], "flatten", "function"],
    ]
    flattened = builder._flatten_text_lists(nested_texts)
    expected = ["hello", "world", "this", "is", "a", "test", "of", "the", "flatten", "function"]
    assert flattened == expected


def test_vocabulary_builder():

  SEQUENCE_MAX_LEN = 10
  VOCAB_SIZE = 12
  data : list = [
    [
      ['Apple', 'looking'] + ["<PAD>"] * (SEQUENCE_MAX_LEN - 2),
      ['Autonomous', 'cars'] + ["<PAD>"] * (SEQUENCE_MAX_LEN - 2)
    ],
    [
      ['The', 'Pope'] + ["<PAD>"] * (SEQUENCE_MAX_LEN - 2), 
      ['Mary', 'was'] + ["<PAD>"] * (SEQUENCE_MAX_LEN - 2)
    ]
  ]
  expected_data: Vocabulary = Vocabulary(
    id2token={0: '<PAD>', 1: '<UNK>', 2: 'Apple', 3: 'Autonomous', 4: 'Mary', 5: 'Pope', 6: 'The', 7: 'cars', 8: 'looking', 9: 'was'}, 
    token2id={'<PAD>': 0, '<UNK>': 1, 'Apple': 2, 'Autonomous': 3, 'Mary': 4, 'Pope': 5, 'The': 6, 'cars': 7, 'looking': 8, 'was': 9}
   )

  builder = VocabularyBuilder()
  actual = builder.build_vocabulary(
    texts=data,
    vocab_size=VOCAB_SIZE,
    num_stopwords=0,
    padding_token="<PAD>",
    unknown_token="<UNK>"
  )
  assert actual == expected_data
