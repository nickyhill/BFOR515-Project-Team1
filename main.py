import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

#load dataset
df = pd.read_csv('cybersecurity_intrusion_data.csv')

#identify columns we are working with
required_cols = ['network_packet_size', 'session_duration']

df_clean = df.dropna(subset=required_cols).copy()

#print the number of columms in the dataframe
print(df_clean.describe())
print(df_clean.value_counts())

#create variables called network_packet_size and session_duration to be used for x and y values in the scatterplot
network_packet_size = df_clean['network_packet_size']
session_duration = df_clean['session_duration']

#create a scatterplot that compares network packet size to session duration
duration_plot = sns.scatterplot(x=network_packet_size, y=session_duration)
plt.title('Network Packet Size vs Session Duration')
plt.xlabel('Network Packet Size')
plt.ylabel('Session Duration')
plt.show()

#create a pairplot that creates a scatter of all of the columns so that we can identify what variables to hone in on
required_cols = ['session_id', 'network_packet_size', 'protocol_type', 'login_attempts', 'session_duration', 'encryption_used', 'ip_reputation_score', 'failed_logins', 'browser_type', 'unusual_time_access', 'attack_detected']
scat = sns.pairplot(df_clean[required_cols], diag_kind='kde', kind='kde')
plt.show()
