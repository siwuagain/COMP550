from nltk.tokenize import word_tokenize
import unidecode

import string

class PreProcessor:

  #iter: str | list[str]

  #punctuation
  def rmv_punct(self, iter) -> list[str]:

    if isinstance(iter, str):
      for punct in string.punctuation:
        iter = iter.replace(punct, '')
      return iter

    return list(filter(lambda s: len(s) > 0, [self.rmv_punct(word) for word in iter]))
  
  #lowercase
  def to_lowercase(self, iter) -> list[str]:
    return [word.lower() for word in iter]
  
  #accent
  def rmv_diacritic(self, iter) -> list[str]:
    return [unidecode.unidecode(word) for word in iter]

  #tokenize
  def tokenize(self, iter) -> list[str]:
    return word_tokenize(iter)
  