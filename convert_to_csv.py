import wfdb
import pandas as pd

# Read the record (without the file extension)
record_name = '/Users/alinawaf/Desktop/Research/ECG-VECG/MIMIC_Dataset/s40689238/40689238'
record = wfdb.rdrecord(record_name)

# Extract signal data and channel names
signals = record.p_signal
channel_names = record.sig_name

# Create a DataFrame and save as CSV
df = pd.DataFrame(signals, columns=channel_names)
df.to_csv(f'{record_name}.csv', index=False)

print(f"Saved as {record_name}.csv")