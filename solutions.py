# solutions.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

'''Implement the methods from the classes in inference.py here'''

import util
import inference
from util import raiseNotDefined
import random
import busters


def normalize(self):
    """
    Normalize the distribution such that the total value of all keys sums
    to 1. The ratio of values for all keys will remain the same. In the case
    where the total value of the distribution is 0, do nothing.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> dist.normalize()
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
    >>> dist['e'] = 4
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
    >>> empty = DiscreteDistribution()
    >>> empty.normalize()
    >>> empty
    {}
    """
    "*** YOUR CODE HERE ***"
    total = self.total()
    # empty and sum euqals 0
    if len(self.items()) != 0 and total != 0:
        for i in self.keys():
            self[i] = float(self[i] / total)
    # None if empty or sum to 0
    else:
        return None




def sample(self):
    """
    Draw a random sample from the distribution and return the key, weighted
    by the values associated with each key.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> N = 100000.0
    >>> samples = [dist.sample() for _ in range(int(N))]
    >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
    0.2
    >>> round(samples.count('b') * 1.0/N, 1)
    0.4
    >>> round(samples.count('c') * 1.0/N, 1)
    0.4
    >>> round(samples.count('d') * 1.0/N, 1)
    0.0
    """
    "*** YOUR CODE HERE ***"
    # choose a number
    probability = random.random()
    # create a list
    keys = []
    for k in self.keys():
        keys.append(k)

    # set the lower limit
    lower_limit = 0
    for i in range(len(keys)):
        # set the upper limit
        upper_limit = (lower_limit + self[keys[i]] / self.total())
        if probability > lower_limit and probability <= upper_limit:
            return keys[i]
        # update the lower limit
        lower_limit = upper_limit


def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
    """
    Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
    """
    "*** YOUR CODE HERE ***"
    # Get the distance between pacman and ghost
    distance = util.manhattanDistance(pacmanPosition, ghostPosition)
    # special cases
    if noisyDistance is None:
        if ghostPosition == jailPosition:
            return 1
    if noisyDistance is None or ghostPosition == jailPosition:
        return 0

    # conditional probability
    return busters.getObservationProbability(noisyDistance, distance)



def observeUpdate(self, observation, gameState):
    """
    Update beliefs based on the distance observation and Pacman's position.

    The observation is the noisy Manhattan distance to the ghost you are
    tracking.

    self.allPositions is a list of the possible ghost positions, including
    the jail position. You should only consider positions that are in
    self.allPositions.

    The update model is not entirely stationary: it may depend on Pacman's
    current position. However, this is not a problem, as Pacman's current
    position is known.
    """
    "*** YOUR CODE HERE ***"
    # positions
    pacPos = gameState.getPacmanPosition()
    jailPos = self.getJailPosition()

    # update the old position by multiplying
    # onew_pro = old_prob * observation_prob
    for pos in self.allPositions:
        self.beliefs[pos] = self.getObservationProb(observation, pacPos, pos, jailPos) * self.beliefs[pos]
    self.beliefs.normalize()




def elapseTime(self, gameState):
    """
    Predict beliefs in response to a time step passing from the current
    state.

    The transition model is not entirely stationary: it may depend on
    Pacman's current position. However, this is not a problem, as Pacman's
    current position is known.
    """
    "*** YOUR CODE HERE ***"
    distribution = inference.DiscreteDistribution()
    # loop over every possible positions
    for oldPos in self.allPositions:
        # obtain the distribution over new positions
        newPosDist = self.getPositionDistribution(gameState, oldPos)
        # old probability = old position in the beliefs
        oldProbability = self.beliefs[oldPos]
        for new in newPosDist.keys():
            distribution[new] = distribution[new] + oldProbability * newPosDist[new]
    # update the beliefs
    self.beliefs = distribution
