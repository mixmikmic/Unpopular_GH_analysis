import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Graphing helper function
def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def generate_return_rates(return_mean, return_std, years):
    return return_mean + return_std * np.random.randn(years)

return_rates = generate_return_rates(8, 10, 1000)
year_range = range(1000)
setup_graph(title='rate of return by year', x_label='year', y_label='interest rate', fig_size=(12,6))
plt.plot(year_range, return_rates)

hist, bins = np.histogram(return_rates, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
setup_graph(title='rate of return', x_label='return rate %', y_label='frequency of return rate', fig_size=(12,6))
plt.bar(center, hist, align='center', width=width)
plt.show()

def calculate_fund_returns(contribution_per_year, returns_by_year, starting_principal=0):
    """ Calculate the fund returns for the sequence of years given.
    
    Parameters:
        * contributsions_per_year - float representing dollars contributed per year
        * returns_by_year - list of percent returns per year like [.1, -.05, ...].
    Returns the fund value by year - a list like this:
        [1000, 1026.25, 1223.75, 1100.75, ...]
    """
    principal = starting_principal
    value_by_year = []
    for return_rate in returns_by_year:
        # Add the contribution first thing each year
        principal = principal + contribution_per_year
        
        # Calculate growth/loss
        principal = principal * (1 + return_rate)
        
        value_by_year.append(principal)

    return value_by_year

calculate_fund_returns(0, [0.07]*10, 1000)

1000 * 1.07

1070 * 1.07

years = range(30)
m1_return_rates = [percent / 100 for percent in generate_return_rates(8, 2, 30)]
m1_value_by_year = calculate_fund_returns(5500, m1_return_rates)

m1_value_by_year

setup_graph(title='Mutual fund 1 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, m1_value_by_year)

def find_average_fund_return(return_mean, return_std, years, addition_per_year, num_trials):
    return_total = 0
    for i in range(num_trials):
        m1_return_rates = [percent / 100 for percent in generate_return_rates(return_mean, return_std, years)]
        m1_value_by_year = calculate_fund_returns(addition_per_year, m1_return_rates)
        final_return = m1_value_by_year[-1]  # [-1] gets the last element in the list
        return_total += final_return
    return return_total / num_trials

m1_average_final_value = find_average_fund_return(8, 2, 30, 5500, 100000)
m1_average_final_value

def percent_to_decimal(percent_iterator):
    for p in percent_iterator:
        yield p / 100

num_trials = 100000
returns_per_trial = (calculate_fund_returns(5500, percent_to_decimal(generate_return_rates(8, 2, 30)))[-1] for i in range(num_trials))
avg_returns = sum(returns_per_trial) / num_trials
avg_returns

m2_return_rates = [percent / 100 for percent in generate_return_rates(8, 10, 30)]
m2_value_by_year = calculate_fund_returns(5500, m2_return_rates)
setup_graph(title='Mutual fund 2 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, m2_value_by_year)

m2_average_final_value = find_average_fund_return(8, 10, 30, 5500, 100000)
m2_average_final_value

m3_return_rates = [percent / 100 for percent in generate_return_rates(8, 30, 30)]
m3_value_by_year = calculate_fund_returns(5500, m3_return_rates)
setup_graph(title='Mutual fund 3 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, m3_value_by_year)

m3_average_final_value = find_average_fund_return(8, 30, 30, 5500, 100000)
m1_average_final_value

print('Fund 1 (8% mean, 2% standard dev) = {}'.format(m1_average_final_value))
print('Fund 2 (8% mean, 10% standard dev) = {}'.format(m2_average_final_value))
print('Fund 3 (8% mean, 30% standard dev) = {}'.format(m3_average_final_value))

def find_probability_of_reaching_goal(return_mean, return_std, years, addition_per_year, num_trials, goal):
    reached_goal = 0
    for i in range(num_trials):
        m1_return_rates = [percent / 100 for percent in generate_return_rates(return_mean, return_std, years)]
        m1_value_by_year = calculate_fund_returns(addition_per_year, m1_return_rates)
        final_return = m1_value_by_year[-1]  # [-1] gets the last element in the list
        if final_return >= goal:
            reached_goal += 1
    return reached_goal / num_trials

m1_probability_of_reaching_600000 = find_probability_of_reaching_goal(8, 2, 30, 5500, 100000, 600000)
m1_probability_of_reaching_600000

m2_probability_of_reaching_600000 = find_probability_of_reaching_goal(8, 10, 30, 5500, 100000, 600000)
m2_probability_of_reaching_600000

m1_probability_of_reaching_600000 = find_probability_of_reaching_goal(8, 30, 30, 5500, 100000, 600000)
m1_probability_of_reaching_600000

def get_fund1_returns():
    fund1_return_rates = [percent / 100 for percent in generate_return_rates(8, 4, 10)] +                          [percent / 100 for percent in generate_return_rates(6, 3, 10)] +                          [percent / 100 for percent in generate_return_rates(4, 2, 10)]
    fund1_value_by_year = calculate_fund_returns(5500, fund1_return_rates)
    return fund1_value_by_year

setup_graph(title='Fund 1 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, get_fund1_returns())

def get_fund2_returns():
    fund_return_rates = [percent / 100 for percent in generate_return_rates(8, 10, 10)] +                         [percent / 100 for percent in generate_return_rates(6, 5, 10)] +                         [percent / 100 for percent in generate_return_rates(4, 2, 10)]
    fund_value_by_year = calculate_fund_returns(5500, fund_return_rates)
    return fund_value_by_year

setup_graph(title='Fund 2 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, get_fund2_returns())

def get_fund3_returns():
    fund_return_rates = [percent / 100 for percent in generate_return_rates(8, 30, 10)] +                         [percent / 100 for percent in generate_return_rates(6, 5, 10)] +                         [percent / 100 for percent in generate_return_rates(4, 2, 10)]
    fund_value_by_year = calculate_fund_returns(5500, fund_return_rates)
    return fund_value_by_year

setup_graph(title='Fund 3 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, get_fund3_returns())

def get_fund4_returns():
    fund_return_rates = [percent / 100 for percent in generate_return_rates(8, 30, 10)] +                         [percent / 100 for percent in generate_return_rates(6, 10, 10)] +                         [percent / 100 for percent in generate_return_rates(4, 2, 10)]
    fund_value_by_year = calculate_fund_returns(5500, fund_return_rates)
    return fund_value_by_year

setup_graph(title='Fund 4 returns', x_label='year', y_label='fund value in $', fig_size=(12,6))
plt.plot(years, get_fund4_returns())

num_trials = 100000
fund1_trial_returns = [get_fund1_returns()[-1] for i in range(num_trials)]
fund1_avg_returns = sum(fund1_trial_returns) / num_trials
fund1_avg_returns

fund2_trial_returns = [get_fund2_returns()[-1] for i in range(num_trials)]
fund2_avg_returns = sum(fund2_trial_returns) / num_trials
fund2_avg_returns

fund3_trial_returns = [get_fund3_returns()[-1] for i in range(num_trials)]
fund3_avg_returns = sum(fund3_trial_returns) / num_trials
fund3_avg_returns

fund4_trial_returns = [get_fund4_returns()[-1] for i in range(num_trials)]
fund4_avg_returns = sum(fund4_trial_returns) / num_trials
fund4_avg_returns

len([i for i in fund1_trial_returns if i >= 400000]) / num_trials

len([i for i in fund2_trial_returns if i >= 400000]) / num_trials

len([i for i in fund3_trial_returns if i >= 400000]) / num_trials

len([i for i in fund4_trial_returns if i >= 400000]) / num_trials

len([i for i in fund1_trial_returns if i >= 500000]) / num_trials

len([i for i in fund2_trial_returns if i >= 500000]) / num_trials

len([i for i in fund3_trial_returns if i >= 500000]) / num_trials

len([i for i in fund4_trial_returns if i >= 500000]) / num_trials

