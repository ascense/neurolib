import nltk
import numpy
import csv
import itertools
import os
import codecs
import logging


class DataSet(object):
    UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'
    SENTENCE_START_TOKEN = 'SENTENCE_START'
    SENTENCE_END_TOKEN = 'SENTENCE_END'

    def __init__(self, text, vocabulary_size):
        self._vocabulary_size = vocabulary_size

        sentences = self._parse_sentences(text)
        self._index_to_word, self._word_to_index = self._build_vocabulary(
            sentences,
            self._vocabulary_size
        )
        self._sentences = self._to_index(sentences)

    @staticmethod
    def from_txt(path, vocabulary_size):
        logging.info("Reading data set from txt: '%s'", os.path.abspath(path))

        with codecs.open(path, encoding='utf_8', mode='r') as fobj:
            fobj.read(1) # skip bom
            lines = fobj.read()

        return DataSet(lines, vocabulary_size)

    @staticmethod
    def from_csv(path, vocabulary_size):
        logging.info("Reading data set from CSV: '%s'", path)

        with open(path, 'rb') as fobj:
            reader = csv.reader(fobj, skipinitialspace=True)
            reader.next()

            lines = [line[0].decode('utf-8') for line in reader]

        return DataSet(lines, vocabulary_size)

    def word_to_index(self, word):
        if word in self._word_to_index:
            return self._word_to_index[word]
        return self._word_to_index[self.UNKNOWN_TOKEN]

    def index_to_word(self, index):
        if 0 <= index < len(self._index_to_word):
            return self._index_to_word[index]
        return self.UNKNOWN_TOKEN

    def _to_index(self, tokenized_sentences):
        return numpy.asarray([
            [self.word_to_index(word) for word in sentence]
            for sentence in tokenized_sentences
        ])

    @classmethod
    def _build_vocabulary(cls, sentences, vocab_size):
        word_freq = nltk.FreqDist(itertools.chain(*sentences))
        logging.info("Found %d unique word tokens", len(word_freq.items()))

        vocab = word_freq.most_common(vocab_size - 1)
        index_to_word = [freq[0] for freq in vocab]
        index_to_word.append(cls.UNKNOWN_TOKEN)
        word_to_index = dict([(w, i) for (i, w) in enumerate(index_to_word)])

        return (index_to_word, word_to_index)

    @classmethod
    def _parse_sentences(cls, text):
        """Parse and tokenize sentences from iterable"""
        template = u"{} {} {}".format(cls.SENTENCE_START_TOKEN,
                                      '{}',
                                      cls.SENTENCE_END_TOKEN)

        sentences = nltk.sent_tokenize(text.lower())
        logging.info("Found %d sentences", len(sentences))
        sentences = [
            nltk.word_tokenize(template.format(sentence))
            for sentence in sentences
        ]

        return sentences


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG)
    DataSet.from_txt(sys.argv[1], int(sys.argv[2]))

