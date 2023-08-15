# Count occurrences of all the unique items
L = ['a', 'b', 'c', 'b', 'a', 'a', 'a']
from collections import Counter
coefs = list(dict(Counter(L)).values())
print(coefs)
# Prints Counter({'a': 4, 'b': 2, 'c': 1})