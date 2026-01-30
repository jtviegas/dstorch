
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
        special_tokens = [padding_token, unknown_token]  # Padding and unknown tokens
        tokentries = [*special_tokens, *list(vocab[num_stopwords:])]
        id2tok = dict(zip(range(len(tokentries)), tokentries, strict=True))

        # compute reverse index
        tok2id = dict(zip(tokentries, range(len(tokentries)), strict=True))

        return Vocabulary(id2token=id2tok, token2id=tok2id)
