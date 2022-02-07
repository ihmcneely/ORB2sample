# -*- coding: UTF-8 -*-

# Authors: Trey McNeely and Galen Vincent
# April 2021
# Updated Nov 2021

# This file contains functions for the simulation study of the local two-sample
# test for independence via a permutation and Monte Carlo test.

import numpy as np
import scipy.sparse as sparse
import scipy.special as special
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.ensemble as ens
import sklearn.neural_network as nnet
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import math
import pandas as pd
import copy
import warnings
from tqdm import tqdm
import rfcde

def arma11(L, phi=0.8, theta=0):
    eps = np.random.normal(size=L + 101) * np.sqrt(1-phi**2)
    out = np.zeros(L + 100)
    out[0] = eps[0]
    for ii in range(1, L + 100):
        out[ii] = phi * out[ii - 1] + eps[ii] + theta*eps[ii-1]

    return out[-L:]

def epanechnikov_factory(r):
    def epanechnikov(d):
        return 3/4*(1-(d/r)**2)
    return epanechnikov

def prob_class_loss(Y, Y_pred):
    pos_cases = np.where(Y == 1)
    return (Y_pred**2).mean() - 2/len(Y)*Y_pred[pos_cases].sum()

def HH(x, delta):
    return x * (np.abs(x) > delta)

def HHz(x, z, delta):
    return x * (np.logical_and(np.abs(x) > delta, np.abs(z) > delta))

class chain:
    def __init__(self, labels, k=4, groups=None):
        '''
        When k=0, the chain is reduced to resampling *with* replacement.
        Thus, for balanced classes, this will be *close* to the permutation,
        but not exact.
        '''
        if groups is not None:
            assert groups.shape == labels.shape
        self.data = labels # pd Series
        self.k = k
        self.groups = groups # pd Series or None
        if groups is None:
            self.groups = pd.Series(['a' for ii in range(labels.shape[0])])
        self.perm = False
        if math.isnan(k):
            self.perm = True
        self.states = float('nan')

        # Trivial case, no need to treat as markov
        if k == 0:
            self.transition = np.mean(labels)
            return

        def find_ngrams(input_list, n):
            return zip(*[input_list[i:] for i in range(n)])

        tmp = [list(find_ngrams(np.asarray(
            labels.loc[self.groups == group], dtype=str), k+1
        )) for group in self.groups.unique()]
        ngrams = list(itertools.chain.from_iterable(tmp))

        ngram_strs = [''.join(x) for x in ngrams]
        ngram_idx = [int(x, 2) for x in ngram_strs]

        # Possible states.
        skip = 2**k
        self.states = [format(x, 'b').zfill(k+1) for x in range(2**(k + 1))]

        # Compute probability of transitions from each state to the next '1' state.
        self.A = np.bincount(ngram_idx, minlength=2*skip)
        to1 = np.zeros(skip)
        for ii in range(skip):
            if np.isnan(self.A[2*ii+1]) or np.isnan(self.A[2*ii]):
                to1[ii] = 0.5
            else:
                to1[ii] = self.A[2*ii+1] / (self.A[2*ii]+self.A[2*ii+1])

        to0 = 1 - to1
        to1 = to1.tolist()
        to0 = to0.tolist()
        '''
        Fill transition matrix. Has the following structure, where p 
        is pulled from the appropriate index of to1 above:
             000  001  010  011  100  101  110  111
        000[ 1-p   p                               ]
        001[           1-p   p                     ]
        010[                     1-p   p           ]
        011[                               1-p   p ]
        100[ 1-p   p                               ]
        101[           1-p   p                     ]
        110[                     1-p   p           ]
        111[                               1-p   p ]
        '''
        iidx = list(range(4*skip))
        iidx = [np.mod(ii, 2*skip) for ii in iidx]
        jidx = list(range(2*skip))
        jidx = [np.mod(2*ii+1, 2*skip) for ii in jidx]
        jidx.extend([ii-1 for ii in jidx])
        vals = to1
        vals.extend(to1)
        vals.extend(to0)
        vals.extend(to0)
        self.transition = sparse.csr_matrix((vals, (iidx, jidx)))

    def draw_group(self, L):
        if self.k==0:
            Y = np.asarray(np.random.uniform(size=L) < self.transition,
                           dtype=int)
            return Y

        else:
            # Burn-in
            idx = np.random.choice(self.states.__len__(), p=self.A/np.sum(self.A))
            for ii in range(100*self.k):
                state1 = (2*idx) % (2**(self.k+1)) + 1
                if np.random.uniform() < self.transition[idx, state1]:
                    idx = state1
                else:
                    idx = state1 - 1
            # Draw
            Y = np.zeros(L)
            Y[0] = idx
            for ii in range(L-1):
                state1 = (2*Y[ii]) % (2**(self.k+1)) + 1
                if np.random.uniform() < self.transition[Y[ii], state1]:
                    Y[ii+1] = state1
                else:
                    Y[ii+1] = state1 - 1

            return Y % 2

    def draw(self, L, groups=None):
        if groups is None:
            groups = pd.Series(['a' for ii in range(L)])
        assert groups.shape[0] == L

        Y = [self.draw_group(sum(groups == group)) for group in groups.unique()]
        return list(itertools.chain.from_iterable(Y))

    def relabel(self, data, groupvar='ID'):
        if groupvar is None:
            out = self.draw(len(data))
        else:
            out = self.draw(len(data), data[groupvar]).values
        return out

class TCchain:
    '''
    This class is identical to chain, but it only looks at subsampled labels,
    then fills in according to a floored linear interpolation. The result is
    that it relabels at x intervals, then fills 1's between 1's and 0's
    elsewhere. Useful for relabeling 30-minute data with 6-hour labels (freq=6).

    NOTE: REQUIRES Pandas series of labels, groups, and timestamp.
    '''
    def __init__(self, labels, groups, time, freq=6, k=4, infreq=.5, hurdat=None):
        '''
        When k=0, the chain is reduced to resampling *with* replacement.
        Thus, for balanced classes, this will be *close* to the permutation,
        but not exact.

        freq: desired frequency of relabeling (6 = 6 hours)
        infreq: frequency of input data (.5 = 30 minutes)
        '''
        assert groups.shape == labels.shape
        self.data = labels # pd Series
        self.k = k
        self.groups = groups # pd Series
        self.time = time # pd Series, converted to hour of day
        self.freq = freq # Label sample rate, usually 6 hours.
        self.infreq = infreq
        self.perm = False
        if math.isnan(k):
            self.perm = True
        self.states = float('nan')
        self.hurdat = hurdat

        # subset for 6-hour entries:
        hours = time.dt.hour + time.dt.minute/60
        idx = (hours % self.freq) == 0
        data_subset = self.data.loc[idx]
        groups_subset = self.groups.loc[idx]
        time_subset = self.time.loc[idx]

        # Trivial case, no need to treat as markov
        if k==0:
            self.transition = np.mean(data_subset)
            return

        def find_ngrams(input_list, n):
            return zip(*[input_list[i:] for i in range(n)])

        tmp = [list(find_ngrams(np.asarray(
            data_subset.loc[groups_subset == group], dtype=str), k+1
        )) for group in groups_subset.unique()]
        ngrams = list(itertools.chain.from_iterable(tmp))

        ngram_strs = [''.join(x) for x in ngrams]
        ngram_idx = [int(x, 2) for x in ngram_strs]

        # Possible states.
        skip = 2**k
        self.states = [format(x, 'b').zfill(k+1) for x in range(2**(k + 1))]

        # Compute probability of transitions from each state to the next '1' state.
        self.A = np.bincount(ngram_idx, minlength=2*skip)
        to1 = np.zeros(skip)
        for ii in range(skip):
            if np.isnan(self.A[2*ii+1]) or np.isnan(self.A[2*ii]):
                to1[ii] = 0.5
            else:
                to1[ii] = self.A[2*ii+1] / (self.A[2*ii]+self.A[2*ii+1])

        to0 = 1 - to1
        to1 = to1.tolist()
        to0 = to0.tolist()
        '''
        Fill transition matrix. Has the following structure, where p 
        is pulled from the appropriate index of to1 above:
             000  001  010  011  100  101  110  111
        000[ 1-p   p                               ]
        001[           1-p   p                     ]
        010[                     1-p   p           ]
        011[                               1-p   p ]
        100[ 1-p   p                               ]
        101[           1-p   p                     ]
        110[                     1-p   p           ]
        111[                               1-p   p ]
        '''
        iidx = list(range(4*skip))
        iidx = [np.mod(ii, 2*skip) for ii in iidx]
        jidx = list(range(2*skip))
        jidx = [np.mod(2*ii+1, 2*skip) for ii in jidx]
        jidx.extend([ii-1 for ii in jidx])
        vals = to1
        vals.extend(to1)
        vals.extend(to0)
        vals.extend(to0)
        self.transition = sparse.csr_matrix((vals, (iidx, jidx)))

    def draw_group(self, L):
        if self.k == 0:
            Y = np.asarray(np.random.uniform(size=L) < self.transition,
                           dtype=int)
            return Y

        else:
            # Burn-in
            idx = np.random.choice(self.states.__len__(), p=self.A/np.sum(self.A))
            for ii in range(100*self.k):
                state1 = (2*idx) % (2**(self.k+1)) + 1
                if np.random.uniform() < self.transition[idx, state1]:
                    idx = state1
                else:
                    idx = state1 - 1
            # Draw
            Y = np.zeros(L)
            Y[0] = idx
            for ii in range(L-1):
                state1 = (2*Y[ii]) % (2**(self.k+1)) + 1
                if np.random.uniform() < self.transition[Y[ii], state1]:
                    Y[ii+1] = state1
                else:
                    Y[ii+1] = state1 - 1

            return Y % 2

    def draw(self, L, groups=None):
        if groups is None:
            groups = pd.Series(['a' for ii in range(L)])
        assert groups.shape[0] == L

        Y = [self.draw_group(sum(groups == group)) for group in groups.unique()]
        return list(itertools.chain.from_iterable(Y))

    def relabel(self, data, groupvar='ID', timecol='timestamp'):
        data6hrfull = pd.concat([self.hurdat.storms.loc[self.hurdat.storms.ID == x, :] for x in data[groupvar].unique()])
        groups = data6hrfull.ID

        newlab = self.draw(len(groups), groups)
        full = pd.DataFrame({'time': data[timecol],
                             'group': data[groupvar]})
        subsample = data6hrfull.copy()
        subsample['Y'] = newlab
        dfs = []
        for group in subsample.ID.unique():
            tmp = subsample.loc[subsample.ID == group, ['DATETIME', 'Y']]
            tmp = tmp.resample('0.5H', on='DATETIME').mean().interpolate()
            tmp['Y'] = np.floor(tmp['Y'])
            dfs.append(full.loc[full.group == group].merge(tmp, how='left', left_on='time', right_on='DATETIME'))

        out = pd.concat(dfs)['Y'].fillna(method='ffill').fillna(value=0).astype('int')
        return out.values

class permuter:
    def __init__(self, labels):
        self.labels = labels

    def relabel(self, data, groupvar=None):
        if groupvar is not None:
            raise Exception("Grouping not yet implemented for permuter class.")
        if len(self.labels) == len(data):
            return np.random.permutation(self.labels)
        else:
            raise Exception("Data length does not match label inputs.")

class random_forest_relabeler:
    '''
    Random forest relabeler, which fits a distribution 

                P(Y_t = 1 | Y_{t-1}, ... , Y_{t-k}, (covariates))

    using random forest classification (estimating probabilities), then samples 
    from this estimated distribution to produce new labels Y.

    params:
        - data (pd.DataFrame): DataFrame containing labels Y in {0, 1} and other
        covariates for fitting classifier. The label column must be named 'Y'.
        - covariates (list): List of column names indicating which columns 
        should be used to fit the relabeler regression.
        - k (int): Memory of the Y sequence.
    '''
    def __init__(self, data_orig, covariates = ["z"], k = 5, cheating = False):
        
        assert 'Y' in data_orig, "'Y' must be a column in provided data."
        assert k >= 0, "k must be >= 0."
        self.orig_covariates = copy.deepcopy(covariates)
        self.covariates = covariates
        self.k = k
        self.cheating = cheating
        
        if self.cheating:
            pass
        
        else:
            data = data_orig.copy(deep = True)

            # Add last k lagged Y values as covariates for the regression
            data = self._add_shifts(data)
            for ii in range(self.k):
                self.covariates.append('Y-' + str(ii+1))

            # Fit RF classifier
            self.rf = ens.RandomForestClassifier(
                n_estimators = 250, # Number of trees
                #min_samples_leaf = int(len(data)*0.05), # Minimum leaf size
                min_samples_leaf = 15,
                n_jobs = 1, # Use parallel computating with this many CPUs (set to -1 to use all available)
                criterion = "entropy"
            )

            X = data.dropna()[self.covariates].values.reshape(-1, len(self.covariates))
            Y = data.dropna()['Y'].values

            self.rf.fit(X, Y)

    def _add_shifts(self, data):
        if self.k == 0: 
            pass
        else:
            for ii in range(self.k):
                data['Y-' + str(ii+1)] = data['Y'].shift(ii+1)

        return data

    def relabel(self, data_orig):
        L = len(data_orig)
        data = data_orig.copy(deep = True) 

        # Use probability forest to generate new label realization
        for cov in self.orig_covariates:
            assert cov in data, "One or more covariates used to fit relabeler is not present in provided data."
        assert 'Y' in data, "'Y' is not a column in provided data."
        
        # Get marginal probs P(Y) and P(Z)
        marginals = np.zeros(len(self.orig_covariates) + 1)
        marginals[0] = np.mean(data['Y']) # Y marginal
        for ii in range(1, len(marginals)):
            marginals[ii] = np.mean(data[self.orig_covariates[ii-1]]) # Covariate marginals
        
        # Burn in
        data_new = np.empty((200, len(marginals) + self.k))
        data_new[:] = np.nan
        for ii in range(1, len(marginals)): # Fill up covariates with their marginals
            data_new[:,ii] = marginals[ii]
        data_new[0:self.k, 0] = np.random.binomial(1, marginals[0], self.k) # Randomly generate the first k values of Y

        for jj in range(self.k, 200): # Burn in first 200 values
            data_new[jj, -self.k:] = np.flip(data_new[(jj-self.k):jj, 0])
            prob = self.rf.predict_proba(np.array([data_new[jj, 1:]]))
            data_new[jj, 0] = np.random.binomial(1, prob[0,1])

        # Actual data generation
        data_generated = np.empty((L + self.k, len(marginals) + self.k))
        data_generated[:] = np.nan

        data_generated[0:self.k, :] = data_new[-self.k:, :] # fill first k rows with burn in data
        for ii in range(len(self.orig_covariates)): # fill covariated from the provided data
            data_generated[self.k:,1 + ii] = data[self.orig_covariates[ii]].values

        for jj in range(self.k, L+self.k): # Generate sequence of Ys
            data_generated[jj, -self.k:] = np.flip(data_generated[(jj-self.k):jj, 0])
            prob = self.rf.predict_proba(np.array([data_generated[jj, 1:]]))
            data_generated[jj, 0] = np.random.binomial(1, prob[0,1])

        return data_generated[self.k:, 0]

    def get_prob_estimates(self, data_orig):
        data = data_orig.copy(deep = True) 

        # Add lagged Y values as covariates
        data = self._add_shifts(data)

        for cov in self.covariates:
            assert cov in data, "One or more covariates used to fit relabeler is not present in provided data."
        assert 'Y' in data, "'Y' is not a column in provided data."

        # Deal with burn-in period by filling with randomly 
        # generated values from marginal distribution for Y.
        ####
        marginal_prob = data['Y'].mean()
        ran = pd.DataFrame(
            1 * (np.random.uniform(size=data.shape) < marginal_prob), 
            columns=data.columns, 
            index=data.index)
        data.where(data.notna(), ran, inplace=True)
        ####
        
        X = data[self.covariates].values.reshape(-1, len(self.covariates))
        probs = self.rf.predict_proba(X)
        
        return probs[:,1]

class logistic_relabeler:
    def __init__(self, data_orig, covariates = ["z"], k = 5):
        
        assert 'Y' in data_orig, "'Y' must be a column in provided data."
        assert k >= 0, "k must be >= 0."
        self.orig_covariates = copy.deepcopy(covariates)
        self.covariates = covariates
        self.k = k
        
        data = data_orig.copy(deep = True)

        # Add last k lagged Y values as covariates for the regression
        data = self._add_shifts(data)
        for ii in range(self.k):
            self.covariates.append('Y-' + str(ii+1))

        # Fit RF classifier
        self.reg = lm.LogisticRegression(
            penalty = 'none'
        )

        X = data.dropna()[self.covariates].values.reshape(-1, len(self.covariates))
        Y = data.dropna()['Y'].values

        self.reg.fit(X, Y)

    def _add_shifts(self, data):
        if self.k == 0: 
            pass
        else:
            for ii in range(self.k):
                data['Y-' + str(ii+1)] = data['Y'].shift(ii+1)

        return data

    def relabel(self, data_orig):
        L = len(data_orig)
        data = data_orig.copy(deep = True) 

        # Use probability forest to generate new label realization
        for cov in self.orig_covariates:
            assert cov in data, "One or more covariates used to fit relabeler is not present in provided data."
        assert 'Y' in data, "'Y' is not a column in provided data."
        
        # Get marginal probs P(Y) and P(Z)
        marginals = np.zeros(len(self.orig_covariates) + 1)
        marginals[0] = np.mean(data['Y']) # Y marginal
        for ii in range(1, len(marginals)):
            marginals[ii] = np.mean(data[self.orig_covariates[ii-1]]) # Covariate marginals
        
        # Burn in
        data_new = np.empty((200, len(marginals) + self.k))
        data_new[:] = np.nan
        for ii in range(1, len(marginals)): # Fill up covariates with their marginals
            data_new[:,ii] = marginals[ii]
        data_new[0:self.k, 0] = np.random.binomial(1, marginals[0], self.k) # Randomly generate the first k values of Y

        for jj in range(self.k, 200): # Burn in first 200 values
            data_new[jj, -self.k:] = np.flip(data_new[(jj-self.k):jj, 0])
            prob = self.reg.predict_proba(np.array([data_new[jj, 1:]]))
            data_new[jj, 0] = np.random.binomial(1, prob[0,1])

        # Actual data generation
        data_generated = np.empty((L + self.k, len(marginals) + self.k))
        data_generated[:] = np.nan

        data_generated[0:self.k, :] = data_new[-self.k:, :] # fill first k rows with burn in data
        for ii in range(len(self.orig_covariates)): # fill covariated from the provided data
            data_generated[self.k:,1 + ii] = data[self.orig_covariates[ii]].values

        for jj in range(self.k, L+self.k): # Generate sequence of Ys
            data_generated[jj, -self.k:] = np.flip(data_generated[(jj-self.k):jj, 0])
            prob = self.reg.predict_proba(np.array([data_generated[jj, 1:]]))
            data_generated[jj, 0] = np.random.binomial(1, prob[0,1])

        return data_generated[self.k:, 0]

    def get_prob_estimates(self, data_orig):
        data = data_orig.copy(deep = True) 

        # Add lagged Y values as covariates
        data = self._add_shifts(data)

        for cov in self.covariates:
            assert cov in data, "One or more covariates used to fit relabeler is not present in provided data."
        assert 'Y' in data, "'Y' is not a column in provided data."

        # Deal with burn-in period by filling with randomly 
        # generated values from marginal distribution for Y.
        ####
        marginal_prob = data['Y'].mean()
        ran = pd.DataFrame(
            1 * (np.random.uniform(size=data.shape) < marginal_prob), 
            columns=data.columns, 
            index=data.index)
        data.where(data.notna(), ran, inplace=True)
        ####
        
        X = data[self.covariates].values.reshape(-1, len(self.covariates))
        probs = self.reg.predict_proba(X)
        
        return probs[:,1]

## TODO: test this and adjust structure of network/fitting procude
class MLP_relabeler:
    '''
    Multi-layer perceptron relabeler, which fits a distribution 

                P(Y_t = 1 | Y_{t-1}, ... , Y_{t-k}, (covariates))

    using MLP classification (estimating probabilities), then samples 
    from this estimated distribution to produce new labels Y.

    params:
        - data_orig (pd.DataFrame): DataFrame containing labels Y in {0, 1} and other
        covariates for fitting classifier. The label column must be named 'Y'.
        - covariates (list): List of column names indicating which columns 
        should be used to fit the relabeler regression.
        - k (int): Memory of the Y sequence.
    '''
    def __init__(self, data_orig, covariates = ["z"], k = 5):
        
        assert 'Y' in data_orig, "'Y' must be a column in provided data."
        assert k >= 0, "k must be >= 0."
        self.orig_covariates = copy.deepcopy(covariates)
        self.covariates = covariates
        self.k = k
        
        data = data_orig.copy(deep = True)

        # Add last k lagged Y values as covariates for the regression
        data = self._add_shifts(data)
        for ii in range(self.k):
            self.covariates.append('Y-' + str(ii+1))

        # Fit MLP classifier
        self.reg = nnet.MLPClassifier(
            hidden_layer_sizes = (30, 30, 30),
            activation = "relu",
            solver = "adam",
            learning_rate_init = 0.005,
            learning_rate = "constant", 
            max_iter = 500,
            early_stopping=True,
            n_iter_no_change=23,
            alpha = 0.0005,
            verbose = True
        )

        X = data.dropna()[self.covariates].values.reshape(-1, len(self.covariates))
        Y = data.dropna()['Y'].values

        self.reg.fit(X, Y)

    def _add_shifts(self, data):
        if self.k == 0: 
            pass
        else:
            for ii in range(self.k):
                data['Y-' + str(ii+1)] = data['Y'].shift(ii+1)

        return data
    
    def relabel(self, data_orig):
        L = len(data_orig)
        data = data_orig.copy(deep = True) 

        # Use probability forest to generate new label realization
        for cov in self.orig_covariates:
            assert cov in data, "One or more covariates used to fit relabeler is not present in provided data."
        assert 'Y' in data, "'Y' is not a column in provided data."
        
        # Get marginal probs P(Y) and P(Z)
        marginals = np.zeros(len(self.orig_covariates) + 1)
        marginals[0] = np.mean(data['Y']) # Y marginal
        for ii in range(1, len(marginals)):
            marginals[ii] = np.mean(data[self.orig_covariates[ii-1]]) # Covariate marginals
        
        # Burn in
        data_new = np.empty((200, len(marginals) + self.k))
        data_new[:] = np.nan
        for ii in range(1, len(marginals)): # Fill up covariates with their marginals
            data_new[:,ii] = marginals[ii]
        data_new[0:self.k, 0] = np.random.binomial(1, marginals[0], self.k) # Randomly generate the first k values of Y

        for jj in range(self.k, 200): # Burn in first 200 values
            data_new[jj, -self.k:] = np.flip(data_new[(jj-self.k):jj, 0])
            prob = self.reg.predict_proba(np.array([data_new[jj, 1:]]))
            data_new[jj, 0] = np.random.binomial(1, prob[0,1])

        # Actual data generation
        data_generated = np.empty((L + self.k, len(marginals) + self.k))
        data_generated[:] = np.nan

        data_generated[0:self.k, :] = data_new[-self.k:, :] # fill first k rows with burn in data
        for ii in range(len(self.orig_covariates)): # fill covariated from the provided data
            data_generated[self.k:,1 + ii] = data[self.orig_covariates[ii]].values

        for jj in range(self.k, L+self.k): # Generate sequence of Ys
            data_generated[jj, -self.k:] = np.flip(data_generated[(jj-self.k):jj, 0])
            prob = self.reg.predict_proba(np.array([data_generated[jj, 1:]]))
            data_generated[jj, 0] = np.random.binomial(1, prob[0,1])

        return data_generated[self.k:, 0]

    def get_prob_estimates(self, data_orig):
        data = data_orig.copy(deep = True) 

        # Add lagged Y values as covariates
        data = self._add_shifts(data)

        for cov in self.covariates:
            assert cov in data, "One or more covariates used to fit relabeler is not present in provided data."
        assert 'Y' in data, "'Y' is not a column in provided data."

        # Deal with burn-in period by filling with randomly 
        # generated values from marginal distribution for Y.
        ####
        marginal_prob = data['Y'].mean()
        ran = pd.DataFrame(
            1 * (np.random.uniform(size=data.shape) < marginal_prob), 
            columns=data.columns, 
            index=data.index)
        data.where(data.notna(), ran, inplace=True)
        ####
        
        X = data[self.covariates].values.reshape(-1, len(self.covariates))
        probs = self.reg.predict_proba(X)
        
        return probs[:,1]

## TODO: test this
class MLP_regressor:
    def __init__(self, variables=["x", "z"]):
        self.regression = nnet.MLPClassifier()
        self.variables = variables

    def fit(self, data):
        self.regression.set_params(
            hidden_layer_sizes = (20, 20),
            activation = "relu",
            solver = "lbfgs",
            verbose = True
        )

        assert 'Y' in data, "Y must be a column of provided data"
        for var in self.variables:
            assert var in data, "One or more variable is not present in provided data."

        X = data[self.variables].values.reshape(-1, len(self.variables))
        Y = data['Y'].values

        self.regression.fit(X, Y)

    def predict(self, data):
        for var in self.variables:
            assert var in data, "One or more variable used to fit model is not present in provided data."

        X = data[self.variables].values.reshape(-1, len(self.variables))
        return self.regression.predict_proba(X)[:,1]

class random_forest_regressor:
    def __init__(self, variables=["x", "z"]):
        self.regression = ens.RandomForestClassifier()
        self.variables = variables

    def fit(self, data):
        self.regression.set_params(
            n_estimators = 250, # Number of trees
            #min_samples_leaf = int(len(data)*0.05), # Minimum leaf size
            min_samples_leaf = 15,
            n_jobs = 1, # Use parallel computating with this many CPUs (set to -1 to use all available)
            criterion = "entropy"
        )

        assert 'Y' in data, "Y must be a column of provided data"
        for var in self.variables:
            assert var in data, "One or more variable is not present in provided data."

        X = data[self.variables].values.reshape(-1, len(self.variables))
        Y = data['Y'].values

        self.regression.fit(X, Y)

    def predict(self, data):
        for var in self.variables:
            assert var in data, "One or more variable used to fit model is not present in provided data."

        X = data[self.variables].values.reshape(-1, len(self.variables))
        return self.regression.predict_proba(X)[:, 1]

class functional_rf_regressor:
    def __init__(self, variables=["x"], n_trees=250, mtry=2, node_size=20,
                 n_basis=15, basis_system='cosine', lens=None, flambda=10):
        self.regression = rfcde.RFCDE(n_trees=n_trees,
                                      mtry=mtry,
                                      node_size=node_size,
                                      n_basis=n_basis,
                                      basis_system=basis_system)
        self.variables = variables
        self.lens = lens
        self.flambda = flambda

    def fit(self, data):
        assert 'Y' in data, "Y must be a column of provided data"
        for var in self.variables:
            assert var in data, "One or more variable is not present in provided data."

        X = data[self.variables].values.astype(float)
        Y = data['Y'].values.astype(float)

        self.regression.train(X, Y, lens=self.lens, flambda=self.flambda)

    def predict(self, data):
        for var in self.variables:
            assert var in data, "One or more variable used to fit model is not present in provided data."

        X = data[self.variables].values.astype(float)
        return self.regression.predict_mean(X)

class logistic_regressor:
    def __init__(self, variables=["x"]):
        self.regression = lm.LogisticRegression(penalty='none')
        self.variables = variables

    def fit(self, data):
        self.regression.fit(data[self.variables].values.reshape(-1, len(self.variables)),
                           data['Y'].values)

    def predict(self, data):
        return self.regression.predict_proba(
            data[self.variables].values.reshape(-1, len(self.variables))
        )[:, 1]

class knn_regressor:
    '''
    K-nearest neighbor regression.

    Note: For >1 dimension, all dimensions should be on the same scale.
    '''
    def __init__(self, variables=["x"], k=None):
        self.regression = None
        self.variables = variables
        self.k = k

    def fit(self, data):
        n = len(data)
        if self.k == 'heuristic':
            self.k = int(np.floor(np.sqrt(n)))
            self.regression = nn.KNeighborsClassifier(n_neighbors=self.k)
            self.regression.fit(
                data[self.variables].values.reshape(-1, len(self.variables)),
                data['Y'].values
            )
        elif self.k is None:
            ks = [2**ii+1 for ii in range(3, int(np.log2(n)+1))]
            loss = np.zeros(int(np.log2(n)-2))
            ii = 0
            for kk in ks:
                self.regression = nn.KNeighborsClassifier(n_neighbors=kk)
                loss[ii] = cross_val_score(self.regression,
                                           data[self.variables].values.reshape(-1, len(self.variables)),
                                           data['Y'].values,
                                           cv=10,
                                           scoring=make_scorer(
                                               prob_class_loss,
                                               needs_proba=True
                                           )
                                           ).mean()
                ii += 1
            self.k = ks[np.where(loss == loss.min())[0][0]]
            self.regression = nn.KNeighborsClassifier(n_neighbors=self.k)
            self.regression.fit(
                data[self.variables].values.reshape(-1, len(self.variables)), 
                data['Y'].values
            )
        else:
            self.regression = nn.KNeighborsClassifier(k=self.k)
            self.regression.fit(
                data[self.variables].values.reshape(-1, len(self.variables)), 
                data['Y'].values
            )

    def predict(self, data):
        return self.regression.predict_proba(
            data[self.variables].values.reshape(-1, len(self.variables))
        )[:, 1]

class NW_regressor:
    '''
    Nadaraya-Watson kernel regression using the epanechnikov kernel.

    Note: for >1 dimension, all dimensions should be on the same scale, sd=1.
    '''
    def __init__(self, variables=["x"], r=None):
        self.regression = None
        self.variables = variables
        self.r = r

    def fit(self, data):
        if self.r == 'heuristic':
            self.r = (data[self.variables].std()/data[self.variables].__len__()**.2).values[0]
            self.regression = nn.RadiusNeighborsClassifier(
                    radius=self.r, weights=epanechnikov_factory(self.r)
                )
            self.regression.set_params(outlier_label=2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.regression.fit(
                    data[self.variables].values.reshape(-1, len(self.variables)),
                    data['Y'].values
                )
        elif self.r is None:
            rs = [(ii+1)**2/10 for ii in range(10)]
            loss = np.zeros(10)
            ii = 0
            for rr in rs:
                self.regression = nn.RadiusNeighborsClassifier(
                    radius=rr, weights=epanechnikov_factory(rr)
                )
                self.regression.set_params(outlier_label=2)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    loss[ii] = cross_val_score(self.regression,
                                               data[self.variables].values.reshape(-1, len(self.variables)),
                                               data['Y'].values,
                                               cv=10,
                                               scoring=make_scorer(
                                                   prob_class_loss,
                                                   needs_proba=True
                                               )
                                               ).mean()
                ii += 1
            self.r = rs[np.where(loss == loss.min())[0][0]]
            self.regression = nn.RadiusNeighborsClassifier(
                radius=self.r, weights=epanechnikov_factory(self.r)
            )
            self.regression.set_params(outlier_label=2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.regression.fit(
                    data[self.variables].values.reshape(-1, len(self.variables)), 
                    data['Y'].values
                )
        else:
            self.regression = nn.RadiusNeighborsClassifier(
                radius=self.r, weights=epanechnikov_factory(self.r)
            )
            self.regression.set_params(outlier_label=2)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.regression.fit(
                    data[self.variables].values.reshape(-1, len(self.variables)), 
                    data['Y'].values
                )

    def predict(self, data):
        with warnings.catch_warnings():
            return self.regression.predict_proba(
                data[self.variables].values.reshape(-1, len(self.variables))
            )[:, 1]

class test2sample:
    def __init__(self, train=None, test=None):
        self.train_data = train
        self.test_data = test
        self.tested = False
        self.B = None

    def simulate_data(self, L_train, L_test=250, phi=0.8, phiprime=0.8,
                      theta=0, gamma=0, rho=0, delta=0.5, eta=0,
                      beta=0, delta_z=False):

        # Training data generation
        self.train_data = pd.DataFrame({
            't': range(1, L_train + 1),
            'x`': arma11(L_train, phi, theta),
            'x``': arma11(L_train, phiprime, theta),
            'z': arma11(L_train, phi, theta)
        })
        self.train_data['x'] = (eta ** .5) * self.train_data['z'] + ((1 - eta) ** .5) * self.train_data['x``']

        # Testing data generation
        self.test_data = pd.DataFrame({
            't': range(1, L_test + 1),
            'x`': arma11(L_test, phi, theta),
            'x``': arma11(L_test, phiprime, theta),
            'z': arma11(L_test, phi, theta)
        })
        self.test_data['x'] = eta ** .5 * self.test_data['z'] + (1 - eta) ** .5 * self.test_data['x``']

        if delta_z:
            self.train_data['H'] = HHz(self.train_data['x'], self.train_data['z'], delta)
            self.train_data['p'] = special.expit(
                gamma * self.train_data['H'] + beta * self.train_data['z'] + rho * self.train_data['x`'])

            self.test_data['H'] = HHz(self.test_data['x'], self.test_data['z'], delta)
            self.test_data['p'] = special.expit(
                gamma * self.test_data['H'] + beta * self.test_data['z'] + rho * self.test_data['x`'])
        else:
            self.train_data['H'] = HH(self.train_data['x'], delta)
            self.train_data['p'] = special.expit(
                gamma * self.train_data['H'] + beta * self.train_data['z'] + rho * self.train_data['x`'])

            self.test_data['H'] = HH(self.test_data['x'], delta)
            self.test_data['p'] = special.expit(
                gamma * self.test_data['H'] + beta * self.test_data['z'] + rho * self.test_data['x`'])

        self.train_data['Y'] = 1 * (np.random.uniform(size=L_train) < self.train_data['p'])
        self.test_data['Y'] = 1 * (np.random.uniform(size=L_test) < self.test_data['p'])
        self.tested = False
        self.B = None

    def test(self, relabeler, regression, B=200, pb=True, dir='==', groupvar='ID'):
        # relabeler = a procedure for drawing label sequences from a marginal dsn
        # B=test replications
        # skip=how far apart are our test points?
        # report=how many replications do I wait to update the user?
        self.B = B
        self.r0 = copy.copy(regression)
        self.r0.fit(self.train_data)
        self.p0 = self.r0.predict(self.test_data) - self.train_data['Y'].mean()

        self.test_data.loc[:, 'test_stat'] = self.p0

        self.P = np.zeros((len(self.test_data), B))
        if pb:
            for ii in tqdm(range(B), desc='Computing Null', leave=False):
                relabeled_data = self.train_data.copy()
                for jj in range(501):
                    if jj == 500:
                        raise Exception("Relabeling procedure returned uniform label in 500 attempts.")
                    relabeled_data['Y'] = relabeler.relabel(relabeled_data, groupvar=groupvar)
                    if relabeled_data['Y'].unique().__len__() == 2:
                        break
                r = copy.copy(regression)
                r.fit(relabeled_data)
                self.P[:, ii] = r.predict(self.test_data) - relabeled_data['Y'].mean()
        else:
            for ii in range(B):
                relabeled_data = self.train_data.copy()
                for jj in range(501):
                    if jj == 500:
                        raise Exception("Relabeling procedure returned uniform label in 500 attempts.")
                    relabeled_data['Y'] = relabeler.relabel(relabeled_data, groupvar=groupvar)
                    if relabeled_data['Y'].unique().__len__() == 2:
                        break
                r = copy.copy(regression)
                r.fit(relabeled_data)
                self.P[:, ii] = r.predict(self.test_data) - relabeled_data['Y'].mean()

        self.test_data['fitted_p'] = self.p0 + self.train_data['Y'].mean()
        pvals = np.zeros(len(self.test_data))

        if dir == '=' or dir == "==":
            for ii in range(len(self.test_data)):
                pvals[ii] = (np.sum(np.abs(self.p0[ii]) < np.abs(self.P[ii, :])) + 1) / (B + 1)
        elif dir == '<':
            for ii in range(len(self.test_data)):
                pvals[ii] = (np.sum(np.abs(self.p0[ii] > self.P[ii, :])) + 1) / (B + 1)
        elif dir == '>':
            for ii in range(len(self.test_data)):
                pvals[ii] = (np.sum(np.abs(self.p0[ii] < self.P[ii, :])) + 1) / (B + 1)

        self.test_data['pvals'] = pvals
        self.tested = True

    def conditional_test(self, relabeler, regression_full, regression_partial, B=200, pb=True):
        self.B = B
        self.reg_full0 = copy.copy(regression_full)
        self.reg_part0 = copy.copy(regression_partial)
        self.reg_full0.fit(self.train_data)
        self.reg_part0.fit(self.train_data)

        self.test_data['p_est_full'] = self.reg_full0.predict(self.test_data)
        self.test_data['p_est_part'] = self.reg_part0.predict(self.test_data)
        self.p0 = self.test_data['p_est_full'] - self.test_data['p_est_part']
        self.test_data['test_stat'] = np.abs(self.p0)

        self.P = np.zeros((len(self.test_data), B))
        if pb:
            for ii in tqdm(range(B), desc='Computing Null', leave=False):
                relabeled_data = self.train_data.copy()
                relabeled_data['Y'] = relabeler.relabel(relabeled_data)
                reg_full = copy.copy(regression_full)
                reg_part = copy.copy(regression_partial)
                reg_full.fit(relabeled_data)
                reg_part.fit(relabeled_data)
                self.P[:, ii] = reg_full.predict(self.test_data) - reg_part.predict(self.test_data)
        else:
            for ii in range(B):
                relabeled_data = self.train_data.copy()
                relabeled_data['Y'] = relabeler.relabel(relabeled_data)
                reg_full = copy.copy(regression_full)
                reg_part = copy.copy(regression_partial)
                reg_full.fit(relabeled_data)
                reg_part.fit(relabeled_data)
                self.P[:, ii] = reg_full.predict(self.test_data) - reg_part.predict(self.test_data)

        pvals = np.zeros(len(self.test_data))
        for ii in range(len(self.test_data)):
            pvals[ii] = (
                                np.sum(np.abs(self.p0[ii]) < np.abs(self.P[ii, :])) + 1
                        ) / (B + 1)
        self.test_data['pvals'] = pvals
        self.tested = True

    def get_global(self):
        if not self.tested:
            raise Exception("Test statistics not computed!")

        glob_obs = np.power(self.p0, 2).sum()
        glob_null = np.power(self.P, 2).sum(axis=0)
        return (len(np.where(glob_null > glob_obs)[0]) + 1) / (self.B + 1)