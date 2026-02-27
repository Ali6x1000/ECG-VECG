import os
import wfdb 
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/alinawaf/Desktop/Research/ECG-VECG/CardioVectorLib')
# Import CardioVectorLib modules


from cardiovector import plotting, preprocessing, reconstruction as rec


## Config ## 
Path_Name = "/Users/alinawaf/Desktop/Research/ECG-VECG/Output/1/00001_-0_0000"



##  Ploting vcg from digitizzed ECG after applying base line wandering filter  ##
record = wfdb.rdrecord(Path_Name , physical = False)
record.base_datetime = None



wfdb.plot_wfdb(record, figsize=(15, 12))

signals = record.d_signal  # <-- use d_signal instead of p_signal
lead_names = [name.lower() for name in record.sig_name]

# Example: extract lead II
lead_II = signals[:, lead_names.index("ii")]

# Plot all leads individually
for i, channel in enumerate(signals.T):
    plt.figure(figsize=(12, 3))
    plt.plot(channel)
    plt.title(f'Lead {record.sig_name[i]} (digital values)')
    plt.xlabel('Sample')
    plt.ylabel('Digital value')
    plt.show()













record.sig_name = [s.lower() for s in record.sig_name]
filtered_record = preprocessing.remove_baseline_wandering(record)
record = filtered_record

print(record.sig_name)



print("=== KORS RECORD  ===")
kors_record = rec.kors_vcg(record)


sig = kors_record.d_signal  # shape (5000, 3)
vx, vy, vz = sig.T

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(vx[0:5000], vy[0:5000], vz[0:5000])  # 2s window
ax.set_xlabel("Vx")
ax.set_ylabel("Vy")
ax.set_zlabel("Vz")
plt.show()


print("== VCG data ==")
