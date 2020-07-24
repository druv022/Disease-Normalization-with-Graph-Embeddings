import nltk
import regex as re
import spacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from nerds.doc.document import Document
from nerds.util.logging import get_logger

log = get_logger()

# run "python -m spacy download en" to download the default model
# disable named entity recognition (we train our own NER models)
NLP = spacy.load('en', disable=['ner'])
# enable sentence boundary detector (gives better results)
NLP.add_pipe(NLP.create_pipe("sentencizer"))
MAX_LEN_FACTOR = 1.1

# regular expression for tokenization
# TOKENIZATION_REGEXP = re.compile("([\pL]+|[\d]+|[^\pL])")
TOKENIZATION_REGEXP = re.compile("([^a-zA-Z0-9_-])")

# regular expression for sentence splitting
# https://regex101.com/r/nG1gU7/27
SENTENCE_REGEXP = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")

# list of default stop words from NLTK
STOP_WORDS = set(stopwords.words('english'))


def text_to_sentences(string, method="nltk"):
    """ Given a document (free-form string), we split it to sentences.

        Args:
            string (str): The input string, corresponding to a document.
            method: splitting method, one of ("regexp", "nltk", "spacy")

        Returns:
            list(str), list(start, end): List of segmented sentences with their offsets
            (end is the last character of str).
    """
    # split to sentences
    if method == "regexp":
        sentences = SENTENCE_REGEXP.split(string)
    elif method == "nltk":
        sentences = nltk.sent_tokenize(string)
    elif method == "spacy":
        _check_and_fix_max_length(string)
        # disable dependency parsing and POS tagging
        with NLP.disable_pipes('parser', 'tagger'):
            sentences = [sent.text for sent in NLP(string).sents]
    elif method == "newline":
        sentences = string.splitlines()
    else:
        raise TypeError("Sentence splitting method is not supported.")
    # find offsets of sentences
    offsets = get_partition_offsets(string, sentences)
    return sentences, offsets


def _check_and_fix_max_length(string):
    if len(string) > NLP.max_length:
        log.debug("Length of text {} exceeds spaCy's default max_length {}: trying to increase max_length...".format(
            len(string), NLP.max_length))
        NLP.max_length = len(string) * MAX_LEN_FACTOR


def get_partition_offsets(string, partition):
    """ Finds offsets of substrings obtained as a result of partition of a free-form string.
    Args:
        string (str): free-form string, text
        partition (list(str)): substrings whose offsets need to be found
    Returns:
        list(2-tuple(start, end)): List of offsets of substrings (end is the last character).
    """
    offsets = []
    from_index = 0
    for part in partition:
        start = string.find(part, from_index)
        end = start + len(part) - 1
        from_index = end + 1
        offsets += [(start, end)]
    return offsets


def get_partition_by_offsets(string, given_offsets):
    if not given_offsets:
        return [string], [(0, len(string) - 1)]

    # remove overlaps
    removals = set()
    for offset in given_offsets:
        for other_offset in given_offsets:
            if offset == other_offset:
                continue
            if other_offset[0] <= offset[0] and offset[1] <= other_offset[1]:
                removals.add(offset)
                break
    given_offsets = [o for o in given_offsets if o not in removals]

    # split
    snippets, offsets = [], []
    for i in range(len(given_offsets)):
        if i == 0:
            if given_offsets[i][0] > 0:
                offsets.append((0, given_offsets[i][0] - 1))
                snippets.append(string[offsets[-1][0]: offsets[-1][1] + 1])
        else:
            offsets.append((given_offsets[i - 1][1] + 1, given_offsets[i][0] - 1))
            snippets.append(string[offsets[-1][0]: offsets[-1][1] + 1])
        offsets.append((given_offsets[i][0], given_offsets[i][1]))
        snippets.append(string[offsets[-1][0]: offsets[-1][1] + 1])
    # end
    if given_offsets[-1][1] < len(string) - 1:
        offsets.append((given_offsets[-1][1] + 1, len(string) - 1))
        snippets.append(string[offsets[-1][0]: offsets[-1][1] + 1])
    return snippets, offsets


def text_to_tokens(string, method="nltk", skip_empty=True):
    """ Given a sentence (free-form string), we tokenize it according to a
        method. Two methods are currently supported: (i) "regexp", which
        tokenizes on every word, symbol and number and (ii) "statistical",
        which uses SpaCy's trained statistical tokenizer.

        Args:
            skip_empty:
            string (str): The input sentence.
            method (str, optional): Either "regexp" or "statistical".
                Defaults to "regexp"

        Returns:
            list(str): List of tokens.

        Raises:
            TypeError: If the tokenization method is not supported.
    """
    if method == "regexp":
        return [token for token in TOKENIZATION_REGEXP.split(string) if not skip_empty or token.strip()]
    elif method == "nltk":
        return [token for token in nltk.word_tokenize(string) if not skip_empty or token.strip()]
    elif method == "spacy":
        _check_and_fix_max_length(string)
        # disable dependency parsing and POS tagging
        with NLP.disable_pipes('parser', 'tagger'):
            return [token.text for token in NLP(string) if not skip_empty or token.text.strip()]
    else:
        raise TypeError("Tokenization method is not supported.")


def tokens_to_pos_tags(tokens, method="nltk"):
    """ Given a tokenized sentence, we use the NLTK or Spacy tagger to give us POS tags.

        Args:
            method:
            tokens (list(str)): The input sentence
                post-tokenization.

        Returns:
            list(str): List of POS tags.
    """
    if method == "nltk":
        return [tag for _, tag in nltk.pos_tag(tokens)]
    elif method == "spacy":
        string = " ".join(tokens)
        _check_and_fix_max_length(string)
        # disable dependency parsing
        with NLP.disable_pipes('parser'):
            return [token.pos_ for token in NLP(string)]
    else:
        raise TypeError("Part-of-speech tagging method is not supported.")


def text_to_spacy_document(string):
    """ Given a document (free-form string), this function converts it so a Spacy format document
        containing tokens, pos tags, dependencies, etc.

    :param string:
    :return: Spacy format document
    """
    _check_and_fix_max_length(string)
    # dependency parsing and POS tagging are enabled
    return NLP(string)


def document_to_tokens(document, method="nltk", skip_empty=True):
    """ Tokenizes a document if required, type-agnostic.
    """
    if isinstance(document, list):
        # assume already tokenized
        return document
    elif isinstance(document, str):
        # string
        return text_to_tokens(document, method=method, skip_empty=skip_empty)
    elif isinstance(document, Document):
        # Document or AnnotatedDocument
        return text_to_tokens(document.plain_text_, method=method, skip_empty=skip_empty)
    else:
        raise TypeError("Cannot determine document type '{}'".format(type(document)))


def remove_stop_words(string, method="nltk"):
    tokens = []
    if method == "nltk":
        for token in text_to_tokens(string, method=method):
            if token not in STOP_WORDS:
                tokens.append(token)
    elif method == "spacy":
        document = text_to_spacy_document(string)
        for token in document:
            if not token.is_stop:
                tokens.append(token.text)
    else:
        raise TypeError("Stop words removal method is not supported.")
    return " ".join(tokens)


def lemmatize(string, method="nltk"):
    tokens = []
    if method == "nltk":
        lemmatizer = WordNetLemmatizer()
        for token in text_to_tokens(string, method=method):
            tokens.append(lemmatizer.lemmatize(token))
    elif method == "spacy":
        document = text_to_spacy_document(string)
        for token in document:
            tokens.append(token.lemma_ if lemmatize else token.text)
    else:
        raise TypeError("Lemmatization method is not supported.")
    return " ".join(tokens)


def remove_stop_words_and_lemmatize(string, method="nltk"):
    tokens = []
    if method == "nltk":
        lemmatizer = WordNetLemmatizer()
        for token in text_to_tokens(string, method=method):
            if token not in STOP_WORDS:
                tokens.append(lemmatizer.lemmatize(token))
    elif method == "spacy":
        document = text_to_spacy_document(string)
        for token in document:
            if not token.is_stop:
                tokens.append(token.lemma_)
    else:
        raise TypeError("Method is not supported.")
    return " ".join(tokens)
