import pandas as pd

reward_file = pd.read_csv("reward_record.monitor_NEW_CSV.csv")
reward_file.info()

reward_file['avg_cumulative'] = reward_file['Rewards'].expanding().mean()
reward_file['total_timestep'] = reward_file['Timesteps'].expanding().sum()

import seaborn as sns
ax = sns.scatterplot(data=reward_file, x='total_timestep', y='avg_cumulative', color='red')
ax.set(xlabel="Total Timesteps", ylabel = "Average Cumulative Reward")

ax = sns.scatterplot(data=reward_file, x='total_timestep', y='Rewards')
ax.set(xlabel="Total Timesteps", ylabel = "Rewards Per Episode")