import unittest

import tokenizers.tokenizer_tools as tt


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.sequence = "Using a Transformer network is simple"
        self.vocab = tt.create_vocab()

    def test_tokenize(self):
        tokenized_text = tt.tokenize(self.sequence)
        expected_output = ["Using", "a", "Transformer", "network", "is", "simple"]

        self.assertEqual(tokenized_text, expected_output)

    def test_add_word_to_vocab(self):
        word = "transformer"
        tt.add_word_to_vocab(self.vocab, word)

        expected_token = 0
        token_id = self.vocab.get(word, None)

        self.assertIsNotNone(token_id)
        self.assertEqual(token_id, expected_token)

    def test_convert_token_to_id(self):

        tokenized_text = tt.tokenize(self.sequence)
        tt.convert_tokens_to_ids(tokenized_text)

        self.vocab
        pass


if __name__ == "__main__":
    unittest.main()
