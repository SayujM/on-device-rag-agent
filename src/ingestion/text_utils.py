
import re

# A simple list of common English stop words for basic filtering.
# For a more robust solution in a production system, a library like NLTK would be preferable.
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
    'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

def tokenize_text(text: str) -> list[str]:
    """
    A robust tokenizer that lowercases, removes punctuation, and removes stop words.
    This ensures that the search is case-insensitive and focuses on meaningful terms.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list[str]: A list of processed tokens.
    """
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Split into words (tokenize)
    tokens = text.split()
    # 4. Remove stop words
    return [token for token in tokens if token not in STOP_WORDS]
