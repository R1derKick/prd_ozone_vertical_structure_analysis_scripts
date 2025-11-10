TKE Analysis Tool with Mixing Layer Height Integration
A Python tool for calculating hourly Turbulent Kinetic Energy (TKE) from wind profiler data, with integrated mixing layer height (MLH) analysis and phase-based visualization.
Overview
This tool processes wind profiler observations (vertical velocity, horizontal wind speed, and wind direction) to compute hourly TKE, then classifies results by four pollution episodes (Onset, Peak, Persistence, Dissipation) and environmental conditions (clean/polluted, day/night). It generates statistical summaries and vertical profile plots with overlaid MLH for comparative analysis.
Core Features
Data Processing
Reads wind profiler data (CSV format) and cleans missing values (converts 999 to NaN).
Processes ceilometer-derived MLH data (optional) to compute hourly averages.
TKE Calculation
Converts horizontal wind speed/direction to zonal (u) and meridional (v) components.
Computes hourly TKE using the formula:
TKE = 0.5 × (u'² + v'² + w'²)
where u', v', w' are turbulent fluctuations (variances) of wind components.
Supports flexible time windows (forward/backward) for hourly aggregation.
Phase Classification
Categorizes data into 4 pollution episodes (configurable time ranges in TIME_PERIODS).
Further classifies each episode into 4 types: Clean night, Clean daytime, Polluted night, Polluted daytime.
Computes statistics (mean, standard deviation, sample count) for TKE and MLH per category.
Visualization
Generates 1×4 subplot figures (one per episode) showing vertical TKE profiles.
Overlays mean MLH as dashed lines (color-matched to categories).
Standardized formatting for academic use (Arial font, consistent axes, error bars).
Dependencies
Install required packages first:
bash
pip install pandas numpy matplotlib pathlib
Input Data Requirements
Mandatory Wind Profiler Data (CSV files)
3 files with consistent time stamps and height columns:
Vertical velocity data
Column 1: Time stamps (column name: 监测时间, datetime format, e.g., 2025/3/21 1:00:00).
Columns 2+: Vertical wind speed values at different heights (column names as height labels, e.g., 100m, 200m).
Horizontal wind speed data
Same structure as vertical velocity data, with horizontal speed values.
Horizontal wind direction data
Same structure as vertical velocity data, with wind direction (meteorological convention: 0° = north, 90° = east).
Missing values must be marked as 999 (automatically converted to NaN).
Optional Mixing Layer Height (MLH) Data
Folder containing CSV files with ceilometer-derived MLH data.
Files must include a date_stamp column (time stamps) and columns with bl_height (e.g., bl_height_1 for primary MLH).
Usage
1. Configure File Paths
Modify the paths in the if __name__ == "__main__": section to match your local data:
python
运行
input_folder_path = Path("./input_data/wind_profiler")  # Wind profiler data
vertical_speed_input_file = input_folder_path / "vertical_velocity_data.csv"
horizontal_speed_input_file = input_folder_path / "horizontal_speed_data.csv"
horizontal_direction_input_file = input_folder_path / "horizontal_direction_data.csv"
output_file_path = input_folder_path / "hourly_turbulent_kinetic_energy.csv"

blh_folder_path = Path("./input_data")  # Optional MLH data
2. Adjust Parameters (Optional)
Time Periods: Modify TIME_PERIODS to redefine pollution episodes or their time ranges.
Time Window: Set time_window in main():
'backward': TKE for 16:00 uses data from 15:00–15:50.
'forward': TKE for 16:00 uses data from 16:00–16:50.
Visualization: Adjust figsize or xlim in plot_phase_tke_profiles_with_blh() to modify plot dimensions/ranges.
3. Run the Code
bash
python tke_analysis.py
Outputs
Hourly TKE Data
CSV file: hourly_turbulent_kinetic_energy.csv
Columns: observation_time (hourly time stamps) + TKE values at each height (units: m²/s²).
Statistical Summary (Console Output)
For each episode and category:
Average TKE (with sample count).
Average MLH (± standard deviation, with sample count).
Visualization
PNG file: hourly_turbulent_kinetic_energy_mlh_phase_vertical_profiles.png
1×4 subplots (one per episode) with:
Solid lines: Mean TKE profiles (color-coded by category).
Shaded areas: TKE standard deviation.
Dashed lines: Mean MLH (color-matched to TKE profiles).
Notes
Data Compatibility: Ensure all input files share the same time stamps and height labels.
MLH Data: If unavailable, the tool skips MLH analysis and plots only TKE profiles.
Missing Data: Hours with <3 valid data points are marked as NaN in TKE results.
Customization: Modify COLORS to adjust plot colors, or TIME_PERIODS to redefine episode classifications.
Troubleshooting
File Not Found: Verify paths in main() match your local directory structure.
Encoding Errors: Adjust encoding in pd.read_csv() (e.g., use encoding='gbk' for Chinese-language files).
Empty Results: Check if time ranges in TIME_PERIODS overlap with your data’s time stamps.
