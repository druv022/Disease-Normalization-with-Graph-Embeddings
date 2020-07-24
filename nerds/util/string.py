import regex as re
import string

NON_ALPHANUMERIC_REGEXP = re.compile("[^\s\d\pL]")
MULTIPLE_WHITESPACES_REGEXP = re.compile("\s+")
NUMBER_REGEXP = re.compile("\d")


def replace_non_alphanumeric(string, repl=""):
    """ Replaces all non-alphanumeric characters in a string.

        A non-alphanumeric character is anything that is not a letter in any
        language (UTF-8 characters included), or a number. The default behavior
        of this function is to eliminate them, i.e. replace them with the empty
        string (''), but the replacement string is a parameter.

        Args:
            string (str): The input string where all non-alphanumeric
                characters will be replaced.
            repl (str, optional): The replacement string for the
                non-alphanumeric characters. Defaults to ''.

        Returns:
            str: The input string, where the non-alphanumeric characters have
                been replaced.
    """
    return NON_ALPHANUMERIC_REGEXP.sub(repl, string).strip()


def eliminate_multiple_whitespaces(string, repl=" "):
    """ If a string contains multiple whitespaces, this function eliminates them.

        The default behavior of this function is to replace the multiple
        whitespaces with a single space (' '), but the replacement string is a
        parameter.

        Args:
            string (str): The input string where multiple whitespaces will be
                eliminated.
            repl (str, optional): The replacement string for multiple
                whitespaces. Defaults to ' '.

        Returns:
            str: The input string where the multiple whitespaces have been
                eliminated.
    """
    return MULTIPLE_WHITESPACES_REGEXP.sub(repl, string).strip()


def ispunct(target):
    """ This function checks whether a string contains only punctuation characters.
        :param target: The string to be checked.
        :return: True if the string contains only punctuation characters, otherwise False
        """
    for c in target:
        if c not in string.punctuation:
            return False

    return True


def normalize_whitespaces(target):
    """ This function removes all whitespace characters including tabular and newline characters
        :param target: The string to be normalized
        :return: The normalized string
        """
    return " ".join(target.split())


def remove_all_whitespaces(target):
    return "".join(target.split())


def hasnumber(target):
    """ Checks if a string contains a number.
        :param target: The string
        :return: True if it does, False otherwise
        """
    return True if NUMBER_REGEXP.search(target) else False
