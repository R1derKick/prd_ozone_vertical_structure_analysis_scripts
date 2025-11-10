import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings('ignore')

# Set font configuration
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['mathtext.sf'] = 'Arial'

# Define time period configuration
TIME_PERIODS = {
    "Onset": {
        "Clean night": [
            ("2025/3/21  1:00:00", "2025/3/21  6:00:00"),
            ("2025/3/21  20:00:00", "2025/3/22  6:00:00"),
            ("2025/3/22  21:00:00", "2025/3/23  0:00:00")
        ],
        "Clean daytime": [
            ("2025/3/21  7:00:00", "2025/3/21  13:00:00"),
            ("2025/3/22  7:00:00", "2025/3/22  13:00:00")
        ],
        "Polluted night": [
            ("2025/3/21  19:00:00", "2025/3/21  19:00:00"),
            ("2025/3/22  19:00:00", "2025/3/22  20:00:00")
        ],
        "Polluted daytime": [
            ("2025/3/21  14:00:00", "2025/3/21  18:00:00"),
            ("2025/3/22  14:00:00", "2025/3/22  18:00:00")
        ]
    },
    "Peak": {
        "Clean night": [
            ("2025/3/23  1:00:00", "2025/3/23  6:00:00"),
            ("2025/3/23  19:00:00", "2025/3/24  0:00:00")
        ],
        "Clean daytime": [
            ("2025/3/23  7:00:00", "2025/3/23  12:00:00")
        ],
        "Polluted daytime": [
            ("2025/3/23  13:00:00", "2025/3/23  18:00:00")
        ]
    },
    "Persistence": {
        "Clean night": [
            ("2025/3/24  1:00:00", "2025/3/24  6:00:00"),
            ("2025/3/24  19:00:00", "2025/3/25  6:00:00"),
            ("2025/3/25  19:00:00", "2025/3/26  0:00:00")
        ],
        "Clean daytime": [
            ("2025/3/24  7:00:00", "2025/3/24  13:00:00"),
            ("2025/3/25  7:00:00", "2025/3/25  13:00:00"),
            ("2025/3/25  18:00:00", "2025/3/25  18:00:00")
        ],
        "Polluted daytime": [
            ("2025/3/24  14:00:00", "2025/3/24  18:00:00"),
            ("2025/3/25  14:00:00", "2025/3/25  17:00:00")
        ]
    },
    "Dissipation": {
        "Clean night": [
            ("2025/3/26  1:00:00", "2025/3/26  6:00:00"),
            ("2025/3/26  19:00:00", "2025/3/27  6:00:00"),
            ("2025/3/27  19:00:00", "2025/3/28  0:00:00")
        ],
        "Clean daytime": [
            ("2025/3/26  7:00:00", "2025/3/26  18:00:00"),
            ("2025/3/27  7:00:00", "2025/3/27  18:00:00")
        ]
    }
}

# Define color configuration
COLORS = {
    'Clean night': '#1f77b4',  # Blue
    'Clean daytime': '#2ca02c',  # Green
    'Polluted night': '#ff7f0e',  # Orange
    'Polluted daytime': '#d62728'  # Red
}


def parse_data(input_file_path):
    """Read .csv file and convert 999 values to NaN"""
    data = pd.read_csv(input_file_path, encoding='utf-8')

    # Convert "observation_time" column to datetime format
    data['监测时间'] = pd.to_datetime(data['监测时间'])  # Keep original column name from data

    # Get height values (columns starting from the third column)
    heights = data.columns[2:].tolist()

    # Replace 999 values with NaN
    for col in data.columns[2:]:
        data[col] = data[col].replace(999, np.nan)

    return data, heights


def process_mixing_layer_height_data(input_mixing_layer_height_folder_path):
    """Process three-layer mixing layer height data"""
    blh_dfs = []
    for input_file in Path(input_mixing_layer_height_folder_path).glob('*.CSV'):
        try:
            df_mlh = pd.read_csv(input_file, skiprows=[1], encoding='utf-8-sig')
            df_mlh['date_stamp'] = df_mlh['date_stamp'].str.strip('\ufeff')
            df_mlh['date_stamp'] = df_mlh['date_stamp'].str.strip("b'")
            df_mlh['datetime'] = pd.to_datetime(df_mlh['date_stamp'])
            df_mlh.replace(-999, float('nan'), inplace=True)

            # Check for multiple bl_height columns
            bl_height_cols = [col for col in df_mlh.columns if 'bl_height' in col]
            print(f"Found bl_height columns: {bl_height_cols}")

            # Select required columns: datetime and all bl_height columns
            cols_to_keep = ['datetime'] + bl_height_cols
            blh_dfs.append(df_mlh[cols_to_keep])

        except Exception as e:
            print(f"Error processing mixing layer height file {input_file.name}: {str(e)}")

    if not blh_dfs:
        print("Warning: No valid mixing layer height data found")
        return pd.DataFrame()

    combined_df = pd.concat(blh_dfs)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    combined_df['date'] = combined_df['datetime'].dt.date
    combined_df['hour'] = combined_df['datetime'].dt.hour

    # Get all bl_height column names
    bl_height_cols = [col for col in combined_df.columns if 'bl_height' in col]

    hourly_data = []
    unique_dates = combined_df['date'].unique()

    for date in unique_dates:
        day_data = combined_df[combined_df['date'] == date]
        hours = day_data['hour'].unique()
        for hour in hours:
            hour_data = day_data[day_data['hour'] == hour]
            if len(hour_data) > 0:
                timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=int(hour))
                hourly_record = {'datetime': timestamp}

                # Calculate hourly average for each mixing layer height
                for col in bl_height_cols:
                    if col in hour_data.columns:
                        avg_height = hour_data[col].mean()
                        hourly_record[col] = avg_height

                hourly_data.append(hourly_record)

    hourly_df = pd.DataFrame(hourly_data)
    if not hourly_df.empty:
        hourly_df = hourly_df.sort_values('datetime').reset_index(drop=True)

        # Print range for each mixing layer height
        for col in bl_height_cols:
            if col in hourly_df.columns:
                min_val = hourly_df[col].min()
                max_val = hourly_df[col].max()
                print(f"{col} range: {min_val:.1f} to {max_val:.1f} m")

        return hourly_df
    else:
        print("Warning: Data is empty after hourly averaging")
        return pd.DataFrame()


def calculate_u_v_components(horizontal_speed, horizontal_direction):
    """Calculate u and v wind components"""
    # Convert meteorological wind direction to radians in mathematical coordinate system
    direction_rad = np.radians(horizontal_direction)
    direction_rad = np.pi / 2 - direction_rad  # Convert to mathematical coordinate system

    # North wind (0°) corresponds to positive v direction, East wind (90°) corresponds to positive u direction
    u_component = horizontal_speed * np.cos(direction_rad)
    v_component = horizontal_speed * np.sin(direction_rad)

    return u_component, v_component


def calculate_hourly_tke(times, u_data, v_data, w_data, heights,
                         time_window='forward'):
    """
    Calculate hourly turbulent kinetic energy (TKE)

    Parameters:
    -----------
    time_window : str
        'forward': Use data after the hour (e.g., 16:00 uses data from 16:00-16:50)
        'backward': Use data before the hour (e.g., 16:00 uses data from 15:00-15:50)
    """

    # Determine time range
    min_time = times.min().floor('h')  # Floor to nearest hour
    max_time = times.max().ceil('h')  # Ceil to nearest hour

    # Create hourly time series
    hourly_times = pd.date_range(start=min_time, end=max_time, freq='h')

    # Initialize TKE data storage
    tke_data = {height: [] for height in heights}

    # Store actual time points used
    actual_times = []

    # Calculate TKE for each hour
    for hour_time in hourly_times:
        if time_window == 'backward':
            # Use data before the hour (15:00-15:50 -> 16:00)
            start_time = hour_time - timedelta(hours=1)
            end_time = hour_time - timedelta(minutes=10)  # Up to x:50
            label_time = hour_time
        else:  # 'forward'
            # Use data after the hour (16:00-16:50 -> 16:00)
            start_time = hour_time
            end_time = hour_time + timedelta(minutes=50)
            label_time = hour_time

        # Get indices of data within the time window
        hour_indices = (times >= start_time) & (times <= end_time)

        # Skip if no data for this hour
        if not hour_indices.any():
            continue

        actual_times.append(label_time)

        # Calculate TKE for each height in the hour
        for j, height in enumerate(heights):
            u_hour = u_data[hour_indices, j]
            v_hour = v_data[hour_indices, j]
            w_hour = w_data[hour_indices, j]

            # Remove NaN values
            valid_indices = ~np.isnan(u_hour) & ~np.isnan(v_hour) & ~np.isnan(w_hour)

            # Calculate TKE only if enough valid data points (at least 3)
            if np.sum(valid_indices) >= 3:
                u_valid = u_hour[valid_indices]
                v_valid = v_hour[valid_indices]
                w_valid = w_hour[valid_indices]

                # Calculate variance (sample variance)
                u_var = np.var(u_valid, ddof=1)
                v_var = np.var(v_valid, ddof=1)
                w_var = np.var(w_valid, ddof=1)

                # Calculate TKE
                tke = 0.5 * (u_var + v_var + w_var)
                tke_data[height].append(tke)
            else:
                tke_data[height].append(np.nan)

    # Assemble results into DataFrame
    result_df = pd.DataFrame()
    result_df['observation_time'] = actual_times
    for height in heights:
        result_df[height] = tke_data[height]

    return result_df


def classify_tke_and_blh_by_periods(tke_df, heights, blh_data=None):
    """
    Classify and calculate statistics for TKE data and mixing layer height data by time periods

    Returns:
    --------
    dict: Contains statistical data for each episode and time period
    """
    results = {}

    for phase, time_periods in TIME_PERIODS.items():
        results[phase] = {}

        for period_type, time_ranges in time_periods.items():
            # Collect all TKE data for this time period
            tke_period_data = {height: [] for height in heights}
            blh_period_data = []

            for start_str, end_str in time_ranges:
                start_time = pd.to_datetime(start_str)
                end_time = pd.to_datetime(end_str)

                # Find TKE data within the time range
                tke_mask = (tke_df['observation_time'] >= start_time) & (tke_df['observation_time'] <= end_time)
                period_tke_data = tke_df[tke_mask]

                # Collect TKE data for each height
                for height in heights:
                    valid_values = period_tke_data[height].dropna().values
                    if len(valid_values) > 0:
                        tke_period_data[height].extend(valid_values)

                # Collect mixing layer height data if available
                if blh_data is not None and not blh_data.empty:
                    blh_mask = (blh_data['datetime'] >= start_time) & (blh_data['datetime'] <= end_time)
                    if blh_mask.any():
                        # Get first layer of mixing layer height data (usually primary)
                        bl_height_cols = [col for col in blh_data.columns if 'bl_height' in col]
                        if len(bl_height_cols) > 0:
                            period_blh_values = blh_data.loc[blh_mask, bl_height_cols[0]].dropna().values
                            blh_period_data.extend(period_blh_values)

            # Calculate TKE statistics
            tke_stats = {}
            for height in heights:
                if len(tke_period_data[height]) > 0:
                    values = np.array(tke_period_data[height])
                    tke_stats[height] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
                else:
                    tke_stats[height] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'count': 0
                    }

            # Calculate mixing layer height statistics
            blh_stats = {}
            if len(blh_period_data) > 0:
                blh_values = np.array(blh_period_data)
                blh_stats = {
                    'mean': np.mean(blh_values),
                    'std': np.std(blh_values),
                    'count': len(blh_values)
                }
            else:
                blh_stats = {
                    'mean': np.nan,
                    'std': np.nan,
                    'count': 0
                }

            results[phase][period_type] = {
                'tke': tke_stats,
                'blh': blh_stats
            }

    return results


def plot_phase_tke_profiles_with_blh(phase_stats, heights, save_path=None,
                                     figsize=(15, 3.69), xlim=None):
    """
    Plot vertical profiles of TKE for four episodes with overlaid mixing layer height

    Parameters:
    -----------
    phase_stats : dict
        Dictionary containing statistical data for each episode
    heights : list
        List of heights
    save_path : str or Path, optional
        Path to save the figure
    figsize : tuple
        Figure size
    xlim : tuple or None
        X-axis range
    """
    # Convert heights to numerical values (in kilometers)
    height_values = np.array([float(h) / 1000 for h in heights])

    # Create 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    axes = axes.flatten()

    # Set x-axis range
    if xlim is None:
        xlim = (0, 6)

    # Plot each episode
    phase_names = list(TIME_PERIODS.keys())

    for i, phase in enumerate(phase_names):
        ax = axes[i]

        # Plot TKE profiles for each time period
        for period_type in TIME_PERIODS[phase].keys():
            if period_type in phase_stats[phase]:
                stats = phase_stats[phase][period_type]

                # Extract TKE mean and standard deviation for this period
                tke_means = []
                tke_stds = []
                valid_heights = []

                for height in heights:
                    height_stats = stats['tke'][height]
                    if not np.isnan(height_stats['mean']) and height_stats['count'] > 0:
                        tke_means.append(height_stats['mean'])
                        tke_stds.append(height_stats['std'])
                        valid_heights.append(float(height) / 1000)  # Convert to kilometers

                if len(tke_means) > 0:
                    tke_means = np.array(tke_means)
                    tke_stds = np.array(tke_stds)
                    valid_heights = np.array(valid_heights)

                    # Plot mean value line
                    color = COLORS[period_type]
                    ax.plot(tke_means, valid_heights, color=color, linewidth=2,
                            label=period_type, linestyle='-')

                    # Plot error range (filled)
                    ax.fill_betweenx(valid_heights,
                                     tke_means - tke_stds, tke_means + tke_stds,
                                     color=color, alpha=0.2)

                # Plot mixing layer height (horizontal line)
                blh_stats = stats['blh']
                if not np.isnan(blh_stats['mean']) and blh_stats['count'] > 0:
                    blh_height_km = blh_stats['mean'] / 1000.0  # Convert to kilometers
                    blh_std_km = blh_stats['std'] / 1000.0

                    # Plot mean mixing layer height line
                    ax.axhline(y=blh_height_km, color=color, linewidth=2,
                               linestyle='--', alpha=0.8)

        # Set subplot properties
        ax.set_title(phase, fontsize=18, fontweight='bold', pad=8)

        # Add subplot labels (a), (b), (c), (d)
        ax.text(0.99, 0.99, f'({chr(97 + i)})', transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha='right', va='top')

        # Set x-axis
        ax.set_xlim(xlim)
        ax.set_xticks(np.arange(0, 7, 2))

        # Set y-axis
        ax.set_ylim(0, 3)
        yticks = np.append(0.2, np.arange(1, 3.1, 1))  # 0.2, 1, 2, 3
        ax.set_yticks(yticks)

        # Set ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.tick_params(axis='both', which='major', direction='in',
                       length=6, labelsize=18)

        # Set font weight for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Set Y-axis label (only show for first subplot)
        if i == 0:
            ax.set_ylabel('Height (km, AGL)', fontsize=18,
                          labelpad=8, fontweight='bold')

    # Add X-axis label at the bottom center of the entire figure
    fig.text(0.5, -0.05, r'Turbulent kinetic energy (m$^{\mathbf{2}}$ s$^{\mathbf{-2}}$)',
             ha='center', va='bottom', fontsize=18, fontweight='bold')

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase TKE vertical profiles (with MLH) saved to: {save_path}")

    return fig, axes


def main(vertical_speed_input_file, horizontal_speed_input_file,
         horizontal_direction_input_file, output_file_path,
         blh_folder_path=None, time_window='backward', visualize=True):
    """
    Main function

    Parameters:
    -----------
    blh_folder_path : str or Path, optional
        Path to mixing layer height data folder
    time_window : str
        'forward': Use data after the hour
        'backward': Use data before the hour (e.g., 16:00 TKE uses 15:00-15:50 data)
    """

    # Read input files
    print("Reading data files...")
    vertical_speed_data, heights = parse_data(vertical_speed_input_file)
    horizontal_speed_data, _ = parse_data(horizontal_speed_input_file)
    horizontal_direction_data, _ = parse_data(horizontal_direction_input_file)

    print(f"Successfully read data, number of height layers: {len(heights)}")
    print(f"Height range: {heights[0]} - {heights[-1]} m")

    # Read mixing layer height data
    blh_data = None
    if blh_folder_path:
        print("\nReading mixing layer height data...")
        blh_data = process_mixing_layer_height_data(blh_folder_path)
        if blh_data.empty:
            print("Warning: Failed to read mixing layer height data. Only TKE profiles will be plotted.")
            blh_data = None
        else:
            print(f"Successfully read mixing layer height data, number of data points: {len(blh_data)}")

    # Extract time series
    times = vertical_speed_data['监测时间']
    print(f"Time range: {times.min()} to {times.max()}")

    # Convert data to numpy arrays
    w_data = vertical_speed_data[heights].values
    h_speed = horizontal_speed_data[heights].values
    h_direction = horizontal_direction_data[heights].values

    # Calculate u and v components
    print("Calculating u and v wind components...")
    u_data = np.zeros_like(h_speed)
    v_data = np.zeros_like(h_speed)

    for i in range(h_speed.shape[0]):
        u_row, v_row = calculate_u_v_components(h_speed[i], h_direction[i])
        u_data[i] = u_row
        v_data[i] = v_row

    # Calculate hourly TKE
    print(f"Calculating hourly TKE (time window mode: {time_window})...")
    tke_df = calculate_hourly_tke(times, u_data, v_data, w_data, heights,
                                  time_window=time_window)

    # Save results
    tke_df.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Hourly TKE calculation results saved to: {output_file_path}")
    print(f"Calculated TKE data for {len(tke_df)} hours")

    # Classify TKE and mixing layer height data by periods
    print("\nClassifying TKE and mixing layer height data by periods...")
    phase_stats = classify_tke_and_blh_by_periods(tke_df, heights, blh_data)

    # Output statistical information
    print("\n========== Statistical Information of TKE and Mixing Layer Height for Each Episode ==========")
    for phase, phase_data in phase_stats.items():
        print(f"\n{phase}:")
        for period_type, period_data in phase_data.items():
            # TKE statistics
            tke_data = period_data['tke']
            valid_means = [data['mean'] for data in tke_data.values()
                           if not np.isnan(data['mean']) and data['count'] > 0]
            if len(valid_means) > 0:
                avg_tke = np.mean(valid_means)
                data_count = sum([data['count'] for data in tke_data.values()])
                print(f"  {period_type}: Average TKE = {avg_tke:.4f} m²/s², Number of data points = {data_count}")
            else:
                print(f"  {period_type}: No valid TKE data")

            # Mixing layer height statistics
            blh_data_stats = period_data['blh']
            if not np.isnan(blh_data_stats['mean']) and blh_data_stats['count'] > 0:
                print(
                    f"                Average mixing layer height = {blh_data_stats['mean']:.2f} ± {blh_data_stats['std']:.2f} m, Number of data points = {blh_data_stats['count']}")
            else:
                print(f"                No valid mixing layer height data")

    print("=" * 45)

    # Visualization
    if visualize:
        print("\nGenerating visualization plots...")

        # Generate phase profile plot (with mixing layer height)
        profile_path = output_file_path.parent / f"{output_file_path.stem}_mlh_phase_vertical_profiles.png"

        fig, axes = plot_phase_tke_profiles_with_blh(
            phase_stats, heights,
            save_path=profile_path,
            figsize=(15, 3.69),
            xlim=(0, 6)
        )

    return tke_df, phase_stats


if __name__ == "__main__":
    # Set file paths (generic)
    input_folder_path = Path("./input_data/wind_profiler")

    vertical_speed_input_file = input_folder_path / "vertical_velocity_data.csv"
    horizontal_speed_input_file = input_folder_path / "horizontal_speed_data.csv"
    horizontal_direction_input_file = input_folder_path / "horizontal_direction_data.csv"
    output_file_path = input_folder_path / "hourly_turbulent_kinetic_energy.csv"

    # Mixing layer height data path
    blh_folder_path = Path("./input_data")

    # Create input directories if they don't exist
    input_folder_path.mkdir(parents=True, exist_ok=True)
    if blh_folder_path:
        blh_folder_path.mkdir(parents=True, exist_ok=True)

    # Run main program
    # time_window='backward' means 16:00 TKE is calculated from 15:00-15:50 data
    # time_window='forward' means 16:00 TKE is calculated from 16:00-16:50 data
    tke_result, phase_statistics = main(
        vertical_speed_input_file,
        horizontal_speed_input_file,
        horizontal_direction_input_file,
        output_file_path,
        blh_folder_path=blh_folder_path,  # Add mixing layer height data path
        time_window='backward',  # Use the described method: 16:00 uses 15:00-15:50 data
        visualize=True  # Generate visualization plots
    )