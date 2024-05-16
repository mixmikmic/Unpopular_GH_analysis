# Online-LDA

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def parse_sessions_list(sessions, event_set):

    D = len(sessions)
    
    eventsids = list()
    eventscts = list()
    for D in range(0, D):
        events = sessions[D]
        ddict = dict()
        for e in events:
            if (e in event_set):
                eventtoken = event_set[e]
                if (not eventtoken in ddict):
                    ddict[eventtoken] = 0
                ddict[eventtoken] += 1
        eventsids.append(ddict.keys())
        eventscts.append(ddict.values())

    return((eventsids, eventscts))

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, event_set, K, D, alpha = None, eta = None, tau0 = 1024, kappa = 0.7):
        self._events = dict()
        for events in event_set:
            events = events.lower()
            self._events[events] = len(self._events)

        self._K = K
        self._W = len(self._events)
        self._D = D
        self._alpha = alpha if alpha else 1.0 / K
        self._eta = eta if eta else 1.0 / K
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1 * n.random.gamma(100.0, 1.0 / 100.0, (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def update_lambda(self, sessions):

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(sessions)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(sessions, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) +             rhot * (self._eta + self._D * sstats / len(sessions))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, sessions, gamma):
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(sessions).__name__ == 'string'):
            temp = list()
            temp.append(sessions)
            sessions = temp

        (eventsids, eventscts) = parse_sessions_list(sessions, self._events)
        batchD = len(sessions)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(sessions | theta, id)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = eventsids[d]
            cts = n.array(eventscts[d])
            phinorm = n.zeros(len(ids))

            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(sessions)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(model._eta*model._W) - 
                              gammaln(n.sum(model._lambda, 1)))

        return (score)


    def do_e_step(self, sessions):
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(sessions).__name__ == 'string'):
            temp = list()
            temp.append(sessions)
            sessions = temp

        (eventsids, eventscts) = parse_sessions_list(sessions, self._events)
        batchD = len(sessions)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = eventsids[d]
            cts = eventscts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad *                     n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import mongo_client
from bson import ObjectId

project_id = "517eda23c82561f72a000005"

# The number of documents to analyze each k
batchsize = 500
# The total number of documents in Wikipedia
D = mongo_client.get_session_count(project_id)
# The number of topics
K = 6

event_set = mongo_client.get_events_ids_by_project_id(project_id)
W = len(event_set)
model = onlineldavb.OnlineLDA(event_set, K, D)

for k, (n_skip,n_limit) in enumerate(build_batches(n, batch_size)):

    sessions = mongo_client.get_sessions_batch(project_id, n_skip, n_limit)
        
    (gamma, bound) = model.update_lambda(sessions)

    (event_tokens, event_counts) = onlineldavb.parse_sessions_list(sessions, model._event_set)
    pereventsbound = bound * len(sessions) / (D * sum(map(sum, event_counts)))

    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' %         (k, model._rhot, numpy.exp(-pereventsbound))

    if (k % 10 == 0):
        numpy.savetxt('lambda-%d.dat' % k, model._lambda)
        numpy.savetxt('gamma-%d.dat' % k, gamma)



