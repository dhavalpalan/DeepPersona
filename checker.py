import pandas as pd
df = pd.read_csv("sessions.csv")
print(df.shape)
print(df.groupby('is_bot').mean())
print((df < 0).sum())