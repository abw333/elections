import matplotlib.pyplot
import numpy as np
import scipy.stats

'''This script uses a beta prior distribution and Bayesian updates to calculate
   the probability that a candidate wins an election given partial results.'''

def line(i, total):
    '''Helper function that determines the style of a
       matplotlib line based on how many lines will be
       drawn and the relative position of the current line.'''
    if total == 2:
        if i:
            return 'r-'
        else:
            return 'b-'

    median = (total - 1) / 2
    if i < median - .5:
        return 'b-'
    if i > median + .5:
        return 'r-'

    return 'g-'

def alpha_beta(mu, sigma):
    '''Returns the alpha and beta parameters for a beta
       distribution with the given mean and standard deviation.'''
    alpha = mu ** 2 * ((1 - mu) / sigma ** 2 - 1 / mu)
    beta = alpha * (1 / mu - 1)
    return alpha, beta

def winning_probabilities(prior_mu, prior_sigma, total_votes,
                          fraction_counted, fractions_in_favor):
    '''Given the mean and standard deviation for a beta distribution prior, the
       total amount of votes, and the fraction that have been counted, returns
       the probability of winning for the given fractions of votes in favor.'''
    prior_alpha, prior_beta = alpha_beta(prior_mu, prior_sigma)
    counted_votes = total_votes * fraction_counted

    winning_ps = []
    for fraction_in_favor in fractions_in_favor:
        votes_in_favor = counted_votes * fraction_in_favor
        votes_against = counted_votes - votes_in_favor
        beta_dist = scipy.stats.beta(prior_alpha + votes_in_favor,
                                     prior_beta + votes_against)
        winning_ps.append(1 - beta_dist.cdf(.5))

    return winning_ps

def varying_mu(figure_num, prior_mus, prior_sigma,
               total_votes, fraction_counted, fractions_in_favor):
    '''Produces a matplotlib figure that shows the effect of
       varying the mean of the beta distribution prior.'''
    matplotlib.pyplot.figure(figure_num)

    matplotlib.pyplot.title('Varying mu of beta prior')
    matplotlib.pyplot.xlabel('Percentage of counted votes in favor')
    matplotlib.pyplot.ylabel('Probability of winning')

    handles = []
    for i, prior_mu in enumerate(prior_mus):
        winning_ps = winning_probabilities(prior_mu, prior_sigma, total_votes,
                                           fraction_counted, fractions_in_favor)
        handles.append(matplotlib.pyplot.plot(fractions_in_favor, winning_ps,
                                              line(i, len(prior_mus)),
                                              label=prior_mu)[0])

    matplotlib.pyplot.legend(handles=handles, loc='upper left')

def varying_sigma(figure_num, prior_mu, prior_sigmas,
                  total_votes, fraction_counted, fractions_in_favor):
    '''Produces a matplotlib figure that shows the effect of varying
       the standard deviation of the beta distribution prior.'''
    matplotlib.pyplot.figure(figure_num)

    matplotlib.pyplot.title('Varying sigma of beta prior')
    matplotlib.pyplot.xlabel('Percentage of counted votes in favor')
    matplotlib.pyplot.ylabel('Probability of winning')

    handles = []
    for i, prior_sigma in enumerate(prior_sigmas):
        winning_ps = winning_probabilities(prior_mu, prior_sigma, total_votes,
                                           fraction_counted, fractions_in_favor)
        handles.append(matplotlib.pyplot.plot(fractions_in_favor, winning_ps,
                                              line(i, len(prior_sigmas)),
                                              label=prior_sigma)[0])

    matplotlib.pyplot.legend(handles=handles, loc='upper left')

def varying_total_votes(figure_num, prior_mu, prior_sigma,
                        total_voteses, fraction_counted, fractions_in_favor):
    '''Produces a matplotlib figure that shows the
       effect of varying the number of total votes.'''
    matplotlib.pyplot.figure(figure_num)

    matplotlib.pyplot.title('Varying total votes')
    matplotlib.pyplot.xlabel('Percentage of counted votes in favor')
    matplotlib.pyplot.ylabel('Probability of winning')

    handles = []
    for i, total_votes in enumerate(total_voteses):
        winning_ps = winning_probabilities(prior_mu, prior_sigma, total_votes,
                                           fraction_counted, fractions_in_favor)
        handles.append(matplotlib.pyplot.plot(fractions_in_favor, winning_ps,
                                              line(i, len(total_voteses)),
                                              label=total_votes)[0])

    matplotlib.pyplot.legend(handles=handles, loc='lower right')

def varying_fraction_counted(figure_num, prior_mu, prior_sigma,
                             total_votes, fraction_counteds, fractions_in_favor):
    '''Produces a matplotlib figure that shows the effect of
       varying the fraction of votes that has been counted.'''
    matplotlib.pyplot.figure(figure_num)

    matplotlib.pyplot.title('Varying fraction counted')
    matplotlib.pyplot.xlabel('Percentage of counted votes in favor')
    matplotlib.pyplot.ylabel('Probability of winning')

    handles = []
    for i, fraction_counted in enumerate(fraction_counteds):
        winning_ps = winning_probabilities(prior_mu, prior_sigma, total_votes,
                                           fraction_counted, fractions_in_favor)
        handles.append(matplotlib.pyplot.plot(fractions_in_favor, winning_ps,
                                              line(i, len(fraction_counteds)),
                                              label=fraction_counted)[0])

    matplotlib.pyplot.legend(handles=handles, loc='upper left')

if __name__ == '__main__':
    default_prior_mu = .4
    default_prior_sigma = .05
    default_total_votes = 100000
    default_fraction_counted = .01

    varying_mu(0, [.1, .2, .3, .4, .5, .6, .7, .8, .9],
               default_prior_sigma, default_total_votes,
               default_fraction_counted, np.linspace(.4, .6, 200))

    varying_sigma(1, default_prior_mu, [.02, .04, .06, .08, 1],
                  default_total_votes, default_fraction_counted,
                  np.linspace(.45, .65, 200))

    varying_total_votes(2, default_prior_mu, default_prior_sigma,
                        [10000, 100000, 1000000, 10000000],
                        default_fraction_counted, np.linspace(.4, .8, 400))

    varying_fraction_counted(3, default_prior_mu, default_prior_sigma,
                             default_total_votes, [.001, .01, .02, .05, .1],
                             np.linspace(.4, .8, 400))

    matplotlib.pyplot.show()
