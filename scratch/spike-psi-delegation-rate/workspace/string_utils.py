def reverse_word_order(sentence):
    """Reverse the word order of a sentence.

    Args:
        sentence (str): The input sentence.

    Returns:
        str: The sentence with words in reverse order.
    """
    return ' '.join(sentence.split()[::-1])


def count_vowels(s):
    """Count the number of vowels in a string.

    Args:
        s (str): The input string.

    Returns:
        int: The count of vowels (a, e, i, o, u).
    """
    return sum(1 for char in s.lower() if char in 'aeiou')