import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def parse_data(input_file_path):
    """Read .csv file and convert 999 values to NaN"""

    data = pd.read_csv(input_file_path, encoding='utf-8')

    # Convert "observation_time" column to datetime format
    data['监测时间'] = pd.to_datetime(data['监测时间'])  # Keep original column name as it appears in data

    # Get height values (columns starting from the third column)
    heights = data.columns[2:].tolist()

    # Replace 999 values with NaN
    for col in data.columns[2:]:
        data[col] = data[col].replace(999, np.nan)

    return data, heights


def calculate_u_v_components(horizontal_speed, horizontal_direction):
    """Calculate u (zonal) and v (meridional) wind components from horizontal speed and direction"""

    # Convert meteorological wind direction to radians in mathematical coordinate system
    direction_rad = np.radians(horizontal_direction)
    direction_rad = np.pi / 2 - direction_rad  # Transform to mathematical coordinate system

    # North wind (0°) corresponds to positive v direction; East wind (90°) corresponds to positive u direction
    # u component = wind speed * cos(wind direction angle)
    # v component = wind speed * sin(wind direction angle)
    u_component = horizontal_speed * np.cos(direction_rad)
    v_component = horizontal_speed * np.sin(direction_rad)

    return u_component, v_component


def calculate_hourly_tke(times, u_data, v_data, w_data, heights):
    """Calculate hourly turbulent kinetic energy (TKE)"""

    # Determine time range
    min_time = times.min().floor('h')  # Floor to the nearest hour
    max_time = times.max().ceil('h')  # Ceil to the nearest hour

    # Create hourly time series
    hourly_times = pd.date_range(start=min_time, end=max_time, freq='h')

    # Create result DataFrame
    result_df = pd.DataFrame()
    result_df['observation_time'] = hourly_times[:-1]  # Use generic column name

    # Initialize TKE data storage
    tke_data = {height: [] for height in heights}

    # Calculate TKE for each hour
    for i in range(len(hourly_times) - 1):
        start_time = hourly_times[i]
        end_time = hourly_times[i + 1]

        # Get indices of data within the current hour
        hour_indices = (times >= start_time) & (times < end_time)

        # If no data for this hour, fill with NaN
        if not hour_indices.any():
            for height in heights:
                tke_data[height].append(np.nan)
            continue

        # Calculate TKE for each height within the hour
        for j, height in enumerate(heights):
            u_hour = u_data[hour_indices, j]
            v_hour = v_data[hour_indices, j]
            w_hour = w_data[hour_indices, j]

            # Remove NaN values
            valid_indices = ~np.isnan(u_hour) & ~np.isnan(v_hour) & ~np.isnan(w_hour)

            # Calculate TKE only if enough valid data points exist
            if np.sum(valid_indices) > 3:  # Require at least 4 valid data points
                u_var = np.var(u_hour[valid_indices])
                v_var = np.var(v_hour[valid_indices])
                w_var = np.var(w_hour[valid_indices])

                # Calculate TKE: 0.5*(u'^2 + v'^2 + w'^2)
                tke = 0.5 * (u_var + v_var + w_var)
                tke_data[height].append(tke)
            else:
                tke_data[height].append(np.nan)

    # Add TKE data to result DataFrame
    for height in heights:
        result_df[height] = tke_data[height]

    return result_df


def main(vertical_vel_file, horiz_speed_file, horiz_dir_file, output_file_path):
    """Main function to process wind data and calculate hourly TKE"""

    # Read input .csv files
    vertical_vel_data, heights = parse_data(vertical_vel_file)
    horiz_speed_data, _ = parse_data(horiz_speed_file)
    horiz_dir_data, _ = parse_data(horiz_dir_file)

    print(f"Successfully read data, number of height layers: {len(heights)}")

    # Extract time series
    times = vertical_vel_data['监测时间']  # Match original data column name

    # Convert data to numpy arrays for calculation
    w_data = vertical_vel_data[vertical_vel_data.columns[2:]].values
    horiz_speed = horiz_speed_data[horiz_speed_data.columns[2:]].values
    horiz_direction = horiz_dir_data[horiz_dir_data.columns[2:]].values

    # Calculate u and v components
    u_data = np.zeros_like(horiz_speed)
    v_data = np.zeros_like(horiz_speed)

    for i in range(horiz_speed.shape[0]):
        u_row, v_row = calculate_u_v_components(horiz_speed[i], horiz_direction[i])
        u_data[i] = u_row
        v_data[i] = v_row

    # Calculate hourly TKE
    tke_df = calculate_hourly_tke(times, u_data, v_data, w_data, heights)

    # Save results to new .csv file
    tke_df.to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Hourly TKE calculation results saved to: {output_file_path}")


if __name__ == "__main__":
    # Generic file paths (adjust these according to your actual directory structure)
    input_dir = Path("./input_data/wind_profiler/2025_03_21-27")
    vertical_vel_file = input_dir / "vertical_velocity_data.csv"
    horiz_speed_file = input_dir / "horizontal_speed_data.csv"
    horiz_dir_file = input_dir / "horizontal_direction_data.csv"
    output_file = input_dir / "hourly_turbulent_kinetic_energy.csv"

    # Create input directory if it doesn't exist
    input_dir.mkdir(parents=True, exist_ok=True)

    main(vertical_vel_file, horiz_speed_file, horiz_dir_file, output_file)