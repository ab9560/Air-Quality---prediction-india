import pandas as pd

# Load raw
df = pd.read_csv("raw.csv", header=None)
df.columns = ["Country","State","City","Station","Datetime","Lat","Lon","Pollutant","Min","Max","Avg"]

# Convert Avg to number - this fixes the error
df['Avg'] = pd.to_numeric(df['Avg'], errors='coerce')
df = df.dropna(subset=['Avg']) # remove rows where Avg is text

print("Total rows after cleaning:", len(df))

# Convert time
df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Datetime'])

# Pivot WHOLE INDIA data
wide = df.pivot_table(
    index=['City','Station','Datetime'], 
    columns='Pollutant', 
    values='Avg', 
    aggfunc='mean'
).reset_index()

wide.columns.name = None
wide = wide.sort_values('Datetime')
wide = wide.ffill().bfill()

wide.to_csv("full_clean.csv", index=False)
print("SUCCESS! full_clean.csv created")
print("Shape:", wide.shape)
print(wide.columns.tolist())
print(wide.head())