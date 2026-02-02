from typing import Any, Callable
import numpy as np
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class Vocabulary:
    """Immutable vocabulary mapping for text tokenization.

    Provides bidirectional mapping between tokens and their integer IDs.

    Attributes:
        id2token: List mapping token IDs to token strings.
        token2id: Dictionary mapping token strings to their IDs.
    """

    id2token: list[str]
    token2id: dict[str, int]

    @property
    def size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self.id2token)

    def encode(self, token: str) -> int:
        """Encode a token string to its corresponding ID.

        Args:
            token: The token string to encode.

        Returns:
            The integer ID of the token, or the ID of the unknown token if not found.

        """
        return self.token2id.get(token, self.token2id.get("<UNK>"))

    def decode(self, token_id: int) -> str:
        """Decode a token ID to its corresponding string.

        Args:
            token_id: The integer ID of the token to decode.

        Returns:
            The token string corresponding to the given ID.

        """
        return self.id2token[token_id]


class VocabularyBuilder:
    """Builder class for creating Vocabulary instances.

    Allows incremental addition of tokens and finalization into an immutable Vocabulary.
    """

    def _flatten_text_lists(self, texts: list) -> list:
        result = []
        for element in texts:
            if isinstance(element, str):
                result.append(element)
            else:
                result.extend(self._flatten_text_lists(element))
        return result

    def build_vocabulary(
        self,
        texts: list,
        vocab_size: int,
        num_stopwords: int = 1,
        padding_token: str = "<PAD>",
        unknown_token: str = "<UNK>",
    ) -> Vocabulary:

        _texts = self._flatten_text_lists(texts)
        # count frequency of tokens
        counts = Counter(_texts)

        # configurable num_stopwords and vocab_size
        vocab = [x[0] for x in counts.most_common(vocab_size + num_stopwords)]
        special_tokens = [x for x in [padding_token, unknown_token] if x not in vocab] # Padding and unknown tokens
        tokentries = list({*special_tokens, *list(vocab[num_stopwords:])})
        tokentries.sort()
        id2tok = dict(zip(range(len(tokentries)), tokentries, strict=True))

        # compute reverse index
        tok2id = dict(zip(tokentries, range(len(tokentries)), strict=True))

        return Vocabulary(id2token=id2tok, token2id=tok2id)
