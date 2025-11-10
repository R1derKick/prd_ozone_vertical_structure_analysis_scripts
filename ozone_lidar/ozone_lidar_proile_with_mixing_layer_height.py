import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

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

# Define color configuration (using colors from code 2)
COLORS = {
    'Clean night': '#1f77b4',  # Blue
    'Clean daytime': '#2ca02c',  # Green
    'Polluted night': '#ff7f0e',  # Orange
    'Polluted daytime': '#d62728'  # Red
}


def parse_ozone_data(input_file_path):
    """Read ozone concentration data file"""
    try:
        # Read data
        data = pd.read_csv(input_file_path, encoding='utf-8')

        # Convert "监测时间" column to datetime format
        data['监测时间'] = pd.to_datetime(data['监测时间'])

        # Get height values (columns starting from the second column)
        all_heights = [float(h) for h in data.columns[1:].tolist()]

        # Filter out heights below 300 m
        height_mask = [h >= 300 for h in all_heights]
        heights = [h for h in all_heights if h >= 300]

        # Keep time column and data columns for heights above 300 m
        columns_to_keep = ['监测时间'] + [data.columns[i + 1] for i, keep in enumerate(height_mask) if keep]
        data = data[columns_to_keep]

        print(f"Successfully read ozone concentration data, number of records: {len(data)}")
        print(f"Original number of height layers: {len(all_heights)}, range: {min(all_heights)}-{max(all_heights)} m")
        print(f"Filtered number of height layers: {len(heights)}, range: {min(heights)}-{max(heights)} m")

        return data, heights
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None


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

                # Calculate hourly average for each layer of mixing layer height
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


def calculate_phase_ozone_and_blh_stats(ozone_data, heights, blh_data=None):
    """
    Classify and calculate statistics for ozone data and mixing layer height data by time periods

    Returns:
    --------
    dict: Contains statistical data for each episode and time period
    """
    results = {}

    for phase, time_periods in TIME_PERIODS.items():
        results[phase] = {}

        for period_type, time_ranges in time_periods.items():
            # Collect all ozone data for this time period
            all_ozone_data = [[] for _ in range(len(heights))]  # One list per height
            all_blh_data = []  # Mixing layer height data

            for start_str, end_str in time_ranges:
                start_time = pd.to_datetime(start_str)
                end_time = pd.to_datetime(end_str)

                # Get indices of ozone data within this time period
                period_indices = (ozone_data['监测时间'] >= start_time) & (ozone_data['监测时间'] <= end_time)

                # If there is ozone data in this period, collect it
                if period_indices.any():
                    # Extract ozone data for this period (excluding time column)
                    period_ozone_data = ozone_data.loc[period_indices, ozone_data.columns[1:]].values

                    # Add data to each height's list
                    for j in range(period_ozone_data.shape[1]):
                        # Remove NaN values
                        valid_data = period_ozone_data[:, j][~np.isnan(period_ozone_data[:, j])]
                        all_ozone_data[j].extend(valid_data)

                # If there is mixing layer height data, collect it too
                if blh_data is not None and not blh_data.empty:
                    blh_period_indices = (blh_data['datetime'] >= start_time) & (blh_data['datetime'] <= end_time)
                    if blh_period_indices.any():
                        # Get first layer of mixing layer height data (usually the main one)
                        bl_height_cols = [col for col in blh_data.columns if 'bl_height' in col]
                        if len(bl_height_cols) > 0:
                            period_blh_data = blh_data.loc[blh_period_indices, bl_height_cols[0]].dropna().values
                            all_blh_data.extend(period_blh_data)

            # Calculate mean and standard deviation of ozone for each height
            ozone_stats = {}
            for j, height in enumerate(heights):
                if len(all_ozone_data[j]) > 0:
                    values = np.array(all_ozone_data[j])
                    ozone_stats[str(height)] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
                else:
                    ozone_stats[str(height)] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'count': 0
                    }

            # Calculate mean and standard deviation of mixing layer height
            blh_stats = {}
            if len(all_blh_data) > 0:
                blh_values = np.array(all_blh_data)
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
                'ozone': ozone_stats,
                'blh': blh_stats
            }

    return results


def plot_phase_ozone_profiles_with_blh(phase_stats, heights, output_folder_path,
                                       figsize=(15, 3.69), xlim=None):
    """
    Plot vertical profiles of ozone concentration for four episodes with overlaid mixing layer height

    Parameters:
    -----------
    phase_stats : dict
        Dictionary containing statistical data for each episode
    heights : list
        List of heights
    output_folder_path : Path
        Output folder path
    figsize : tuple
        Figure size
    xlim : tuple or None
        X-axis range
    """
    # Ensure output folder exists
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Convert heights to kilometers
    heights_km = [h / 1000 for h in heights]

    # Create 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    # Set x-axis range
    if xlim is None:
        xlim = (0, 200)

    # Plot each episode
    phase_names = list(TIME_PERIODS.keys())

    for i, phase in enumerate(phase_names):
        ax = axes[i]

        # Plot ozone profiles for each time period
        for period_type in TIME_PERIODS[phase].keys():
            if period_type in phase_stats[phase]:
                # Extract mean and standard deviation of ozone for this period
                means = []
                stds = []
                valid_heights = []

                for height in heights:
                    height_stats = phase_stats[phase][period_type]['ozone'][str(height)]
                    if not np.isnan(height_stats['mean']) and height_stats['count'] > 0:
                        means.append(height_stats['mean'])
                        stds.append(height_stats['std'])
                        valid_heights.append(height / 1000)  # Convert to kilometers

                if len(means) > 0:
                    means = np.array(means)
                    stds = np.array(stds)
                    valid_heights = np.array(valid_heights)

                    # Plot mean value line
                    color = COLORS[period_type]
                    ax.plot(means, valid_heights, color=color, linewidth=3,
                            label=period_type, linestyle='-')

                    # Plot error range (filled)
                    ax.fill_betweenx(valid_heights,
                                     means - stds, means + stds,
                                     color=color, alpha=0.3)

                # Plot mixing layer height (horizontal line)
                blh_stats = phase_stats[phase][period_type]['blh']
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
        x_ticks = np.arange(0, xlim[1] + 1, 50)
        ax.set_xticks(x_ticks)

        # Set y-axis (Y-axis range, starting from 0.3 km (above 300 m))
        ax.set_ylim(0, 1.8)
        yticks = np.append(0.3, np.arange(0.5, 1.9, 0.5))
        ax.set_yticks(yticks)

        # Set ticks
        ax.tick_params(axis='both', which='major', direction='in',
                       length=6, labelsize=18)

        # Set font weight for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Set Y-axis label (only show for first subplot)
        if i == 0:
            ax.set_ylabel('Height (km, AGL)', fontsize=18,
                          labelpad=8, fontweight='bold')

        # Add legend (only show in last subplot)
        if i == 3:
            # Create legend with all categories
            legend_elements = []
            for period_type, color in COLORS.items():
                # Mixing layer height (dashed line)
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                                  label=f'{period_type} MLH', linestyle='--', alpha=0.8))

    # Add X-axis label at the bottom center of the entire figure
    fig.text(0.5, -0.05, r'Ozone concentration (μg/m$^\mathbf{3}$)',
             ha='center', va='bottom', fontsize=18, fontweight='bold')

    # Save figure
    output_file = output_folder_path / "ozone_profiles_with_mixing_layer_height_four_episodes_0.3_1.5.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Generated vertical ozone profiles for four episodes (with mixing layer height): {output_file}")

    return fig, axes


def main():
    # File path configuration
    input_folder_path = Path(
        r"")
    ozone_file_path = input_folder_path / ".csv"

    # Mixing layer height data path
    blh_folder_path = Path(
        r"")

    output_folder_path = input_folder_path

    # Read ozone concentration data
    print("Reading ozone concentration data...")
    ozone_data, heights = parse_ozone_data(ozone_file_path)
    if ozone_data is None:
        print("Failed to load ozone concentration data. Please check the file path and format.")
        return

    # Read mixing layer height data
    print("Reading mixing layer height data...")
    blh_data = process_mixing_layer_height_data(blh_folder_path)
    if blh_data.empty:
        print("Warning: Failed to read mixing layer height data. Only ozone profiles will be plotted.")
        blh_data = None

    # Classify ozone and mixing layer height data by time periods
    print("\nClassifying data by time periods...")
    phase_stats = calculate_phase_ozone_and_blh_stats(ozone_data, heights, blh_data)

    # Output statistical information
    print("\n========== Statistical Information of Ozone Concentration and Mixing Layer Height for Each Episode ==========")
    for phase, phase_data in phase_stats.items():
        print(f"\n{phase}:")
        for period_type, period_data in phase_data.items():
            # Ozone concentration statistics
            ozone_data = period_data['ozone']
            valid_means = [data['mean'] for data in ozone_data.values()
                           if not np.isnan(data['mean']) and data['count'] > 0]
            if len(valid_means) > 0:
                avg_ozone = np.mean(valid_means)
                ozone_count = sum([data['count'] for data in ozone_data.values()])
                print(f"  {period_type}: Average ozone concentration = {avg_ozone:.2f} μg/m³, Number of data points = {ozone_count}")
            else:
                print(f"  {period_type}: No valid ozone data")

            # Mixing layer height statistics
            blh_data = period_data['blh']
            if not np.isnan(blh_data['mean']) and blh_data['count'] > 0:
                print(
                    f"                Average mixing layer height = {blh_data['mean']:.2f} ± {blh_data['std']:.2f} m, Number of data points = {blh_data['count']}")
            else:
                print(f"                No valid mixing layer height data")

    print("=" * 50)

    # Plot vertical ozone profiles and mixing layer height
    print("\nGenerating vertical ozone profiles (with mixing layer height)...")
    fig, axes = plot_phase_ozone_profiles_with_blh(phase_stats, heights, output_folder_path)

    print("Vertical ozone profiles (with mixing layer height, classified by episode) generated successfully!")

    return phase_stats


if __name__ == "__main__":
    main()