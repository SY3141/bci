import mne
import time

# 1. Load the standard 10-05 montage
montage = mne.channels.make_standard_montage('standard_1005')

# 2. Visualize in 2D (topomap)
montage.plot(kind='topomap', show_names=True)

# 3. Visualize in 3D
montage.plot(kind='3d', show_names=True)



time.sleep(120)  # Keep the plots open for 5 seconds