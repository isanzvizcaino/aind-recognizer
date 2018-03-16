import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # Implementation of the recognizer
    for item in test_set.get_all_sequences():
        X, length = test_set.get_item_Xlengths(item)
        LogLvalues = {}
        best_guess = None
        max_score = None
        for word, model in models.items():
            try:
                LogLvalues[word] = model.score(X, length)
                if max_score is None or max_score < LogLvalues[word]:
                    max_score, best_guess = LogLvalues[word], word
            except:
                LogLvalues[word] = None
        probabilities.append(LogLvalues)
        guesses.append(best_guess)

    return probabilities, guesses
