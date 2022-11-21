"""
Tokenizers serve to tranlsate text into data that can be used by a model.
Referencing Huggingface
"""


"""
1. Normalizations: Cleanup and remove spaces, accents, unicode normalization, etc
2. pre-tokenization (splitting the input into words)
3. Running the input through the model (using pre-tokenizaed words to rpoduce a sequence of tokens)
post-processing: Adding the special tokens to the tokenizer, generating attention mask and token type ids
"""

from typing import Dict, List

# vocabulary
vocab: Dict[str, int] = {}
id_to_token: Dict[int, str] = {}

unk_token = "<unk>"  # unknown
cls_token = "<cls>"  # classifier
sep_token = "<sep>"  # separator
pad_token = "<pad>"  # padding

sequence_start_token = "<s>"
sequence_end_token = "</s>"
mask_token = "<mask>"
max_length = 100

encoded_input = {
    "input_ids": List[int],
    "token_type_ids": List[int],
    "attention_mask": List[int],
}


def lowercase(text: str) -> str:
    """make the text lowercase"""
    return text.lower()


def whitespace(tokens: List[str]) -> List[str]:
    """remove whitespace from tokens"""
    return [token.strip() for token in tokens]


def pre_tokenize(tokens: List[str], pre_tokenizers: List):
    tokenized_tokens = []
    for pre_tokenizer in pre_tokenizers:
        tokenized_tokens.append(pre_tokenizer(tokens))

    return tokenized_tokens


def normalize(text: str, normalizers: List):
    return text.lower()


def create_vocab() -> Dict[str, int]:
    """Create a vocab object which maps a word to id"""
    vocab: Dict[str, int] = {}
    return vocab


def create_id_to_token() -> Dict[int, str]:
    """Create a id to token object which maps an id to a word"""
    id_to_token: Dict[int, str] = {}
    return id_to_token


def tokenize(text: str) -> List[str]:
    """convert a string in a sequence to tokens."""
    return text.split()


def convert_token_to_id(vocab: Dict[str, int], token: str) -> int:
    """Convert a token to an id using a vocab"""
    return vocab.get(token, unk_token)


def convert_id_to_token(vocab: Dict[str, int], index: int) -> int:
    """Convert a token to an id using a vocab"""
    return vocab.get(index)


def convert_tokens_to_ids(tokens: List[str]) -> List[int]:
    """Convert a list of tokens to a list of token ids"""
    ids = []
    for token in tokens:
        ids.append(convert_token_to_id(token))
    return ids


def add_word_to_vocab(vocab, word) -> None:
    if word not in vocab:
        vocab[word] = len(vocab.keys())
