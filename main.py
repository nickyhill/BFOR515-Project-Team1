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
