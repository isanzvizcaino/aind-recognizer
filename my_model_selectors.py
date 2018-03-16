import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implementation of model selection based on BIC scores

        # Compute BIC score for every model
        # find the lowest value and, thus, the best number of components
        best_num_components = None
        min_BIC = None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Compute BIC score according to the equation
                # Bayesian information criteria: BIC = -2 * logL + p * logN
                # Components:
                logL = self.base_model(num_components).score(self.X, self.lengths)
                p = num_components**2 + 2*len(self.X[0])*num_components - 1
                logN = np.log(len(self.X))
                # BIC score:
                BIC = -2 * logL + p * logN
                # Find the lowest BIC score depending on num_components
                if min_BIC is None or BIC < min_BIC:
                  min_BIC = BIC
                  best_num_components = num_components
            except:
                pass
        # Apply base_model method with num_states = best_num_components (if any)
        if best_num_components is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Compute DIC score for every model
        # find the highest value and, thus, the best number of components
        best_num_components = None
        max_DIC = None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Compute DIC score according to the equation
                # Discriminative Information Criterion: DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                # Components:
                log_P_X_i = self.base_model(num_components).score(self.X, self.lengths)
                sum_log_P_X_all_but_i = 0.
                words = list(self.words.keys())
                M = len(words)
                words.remove(self.this_word)
                for word in words:
                    try:
                        model_selector_all_but_i = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)
                        sum_log_P_X_all_but_i += model_selector_all_but_i.base_model(num_components).score(model_selector_all_but_i.X, model_selector_all_but_i.lengths)
                    except:
                        M = M - 1

                    # DIC score:
                    DIC = log_P_X_i - sum_log_P_X_all_but_i / (M - 1)

                    # Find the highest DIC score depending on num_components
                    if max_DIC is None or max_DIC < DIC:
                      max_DIC = DIC
                      best_num_components = num_components
            except:
                pass

        # Apply base_model method with num_states = best_num_components (if any)
        if best_num_components is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implementation of model selection using CV

        # Compute k-fold average log Likelihood of every model and
        # find the highest average value and, thus, the best number of components
        best_num_components = None
        max_avg_logL = None
        for num_components in range(self.min_n_components, self.max_n_components + 1):
            list_logL = []
            try:
                # Sum all log Likelihoods for every possible number of components
                split_method = KFold(n_splits=min(3, len(self.sequences)))
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_length = combine_sequences(cv_train_idx,self.sequences)
                    test_X, test_length = combine_sequences(cv_test_idx,self.sequences)
                    try:
                        train_model = self.base_model(num_components).fit(train_X, train_length)
                        logL = train_model.score(cv_test_idx, test_length)
                        list_logL.append(logL)
                    except:
                        pass
                # Compute average log Likelihood
                if count_logL > 0:
                    avg_logL = np.mean(list_logL)
                    # Find the higher average log Likelihood depending on num_components
                    if max_avg_logL is None or max_avg_logL < avg_logL:
                      max_avg_logL = avg_logL
                      best_num_components =  num_components
            except:
                pass
        # Apply base_model method with num_states = best_num_components (if any)
        if best_num_components is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)
