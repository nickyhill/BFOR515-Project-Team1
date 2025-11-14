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

print(df_clean.describe)
print(df_clean.value_counts)

login_attempts = df_clean['login_attempts']
session_duration = df_clean['session_duration']

duration_plot = sns.boxplot(x=login_attempts, y=session_duration)
plt.title('Login Attempts vs Session Duration')
plt.xlabel('Login Attempts')
plt.ylabel('Session Duration')
plt.show()

network_packet_size = df_clean['network_packet_size']

duration_plot = sns.scatterplot(x=network_packet_size, y=session_duration)
plt.title('Network Packet Size vs Session Duration')
plt.xlabel('Network Packet Size')
plt.ylabel('Session Duration')
plt.show()


required_cols = ['session_id', 'network_packet_size', 'protocol_type', 'login_attempts', 'session_duration', 'encryption_used', 'ip_reputation_score', 'failed_logins', 'browser_type', 'unusual_time_access', 'attack_detected']
scat = sns.pairplot(df_clean[required_cols], diag_kind='kde', kind='kde')
plt.show()
