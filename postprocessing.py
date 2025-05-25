import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# === 1. Load the data ===
# file_path = 'bubble_261_kinematics.csv'
file_path = 'bubble_223_kinematics.csv'
df = pd.read_csv(file_path, comment='#')

# === 2. Define the time range ===
start_time = 11
end_time   = 18
df_range = df[(df['time'] >= start_time) & (df['time'] <= end_time)].reset_index(drop=True)

# === 3. Less‐aggressive smoothing params ===
#    Use a small odd window (e.g. 5) and polyorder=2 (or up to window_length-1)
window_length = 100                          # try 3 or 5
polyorder     = min(3, window_length-1)     # up to window_length-1

# Make sure window_length ≤ n_points
n_points = len(df_range)
if window_length > n_points:
    window_length = n_points if n_points % 2 == 1 else n_points-1
    polyorder     = min(polyorder, window_length-1)

# === 4. Apply smoothing ===
for col in ['position_x','position_y','velocity_x','velocity_y','acceleration_x','acceleration_y']:
    df_range[f'{col}_smooth'] = savgol_filter(df_range[col], window_length, polyorder)
    # rolling mean 
    # df_range[f'{col}_smooth'] = df_range[col].rolling(window=window_length, min_periods=1).mean()

# === 5. Plotting ===
def plot_pair(y1, y2, ylabel, title):
    plt.figure()
    plt.plot(df_range['time'], df_range[f'{y1}_smooth'], label=f'{y1} (smoothed)')
    plt.plot(df_range['time'], df_range[f'{y2}_smooth'], label=f'{y2} (smoothed)')
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(f'{y1}_{y2}.png', dpi=300, bbox_inches='tight')

plot_pair('position_x','position_y',   'Position',     'Less‐Smoothed Position vs Time')
plot_pair('velocity_x','velocity_y',   'Velocity',     'Less‐Smoothed Velocity vs Time')
# plot_pair('acceleration_x','acceleration_y', 'Acceleration', 'Less‐Smoothed Acceleration vs Time')
