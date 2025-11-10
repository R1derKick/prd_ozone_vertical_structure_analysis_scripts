Ozone Vertical Profile Analysis with Mixing Layer Height Overlay
Project Overview
This repository contains Python code for analyzing vertical ozone concentration profiles across four ozone pollution episode stages (Onset, Peak, Persistence, Dissipation). The code integrates ozone lidar data and ceilometer-derived mixing layer height (MLH) data, performs time-period classification, statistical analysis, and generates visualized vertical profile plots with overlaid MLH.
Key Features
Data Loading & Preprocessing
Read ozone concentration data from lidar observations (CSV format)
Process three-layer mixing layer height data from ceilometer (CSV format)
Filter heights below 300 m and handle missing values (-999 → NaN)
Calculate hourly averages for mixing layer height data
Classification & Statistical Analysis
Categorize data into 4 types: Clean night, Clean daytime, Polluted night, Polluted daytime
Classify observations by four pollution episode stages (configurable time ranges)
Compute mean and standard deviation of ozone concentration (per height) and MLH
Output detailed statistical summaries for each category
Visualization
Generate 1×4 subplot figures (one per episode stage)
Plot ozone vertical profiles with error bars (std dev)
Overlay mean mixing layer height as dashed lines (color-matched to data categories)
Standardized formatting for academic publication (Arial font, bold labels, consistent axes)
Dependencies
Install required Python packages before running the code:
bash
pip install pandas numpy matplotlib pathlib
pandas ≥ 1.0.0 (data handling)
numpy ≥ 1.18.0 (numerical computations)
matplotlib ≥ 3.2.0 (plotting)
pathlib (file path management, built-in in Python 3.4+)
datetime (time processing, built-in)
File Structure
plaintext
project/
├── Ozone_lidar_data_folder/                # Ozone lidar data folder
│   └── 2025_03_21-27/
│       └── Ozone_concentration_data.csv  # Ozone concentration data (CSV)
├── 21-27/              # Mixing layer height data folder
│           └── *.CSV           # Ceilometer MLH data (multiple CSV files)
├── ozone_profile_analysis.py   # Main analysis script
└── README.md                   # This documentation
Usage Instructions
1. Configure File Paths
Modify the input_folder_path, ozone_file_path, and blh_folder_path in the main() function to match your local file structure:
python
input_folder_path = Path(r"Your/local/path/to/ozone/lidar/data/folder")
ozone_file_path = input_folder_path / "Your_ozone_data.csv"  # Ozone data filename
blh_folder_path = Path(r"Your/local/path/to/ceilometer/MLH/folder")
2. Adjust Time Periods (Optional)
Modify the TIME_PERIODS dictionary to match your study's episode stages and time ranges. Each entry includes:
Episode stage (Onset/Peak/Persistence/Dissipation)
Data category (Clean night/Clean daytime/Polluted night/Polluted daytime)
Time ranges (start time, end time) in "YYYY/M/D H:M:S" format
3. Run the Code
Execute the main script:
bash
python ozone_profile_analysis.py
Parameter Explanations
Core Configurations
Parameter	Description
TIME_PERIODS	Defines episode stages and corresponding time ranges for each data category
COLORS	Color mapping for each data category (consistent for profiles and MLH lines)
height filter	Automatically filters heights below 300 m (modify in parse_ozone_data())
xlim	Ozone concentration axis range (default: 0–200 μg/m³, modify in plot_phase_ozone_profiles_with_blh())
ylim	Height axis range (0–1.8 km AGL, modify in plot_phase_ozone_profiles_with_blh())
Data Category Criteria
Clean/Polluted: Based on 1-hour average surface ozone concentration (<200 μg/m³ = Clean; ≥200 μg/m³ = Polluted)
Daytime/Nighttime: Daytime = 06:30–18:30 UTC+8; Nighttime = ceilometer-derived sunrise/sunset times
Outputs
1. Visualization
File name: ozone_profiles_with_mixing_layer_height_four_episodes_0.3_1.5.png
Format: PNG (300 DPI, tight layout)
Content: 1×4 subplots (one per episode stage) with:
Solid lines: Mean ozone concentration profiles (color-coded by category)
Shaded areas: Standard deviation of ozone concentration
Dashed lines: Mean mixing layer height (color-matched to corresponding profile)
Axes labels: Height (km AGL) and ozone concentration (μg/m³)
2. Statistical Summary (Printed to Console)
For each episode stage and data category:
Average ozone concentration (μg/m³) and number of valid data points
Average mixing layer height (m) ± standard deviation and number of valid data points
Notes & Limitations
Data Format Requirements
Ozone data CSV must include a "监测时间" column (time stamp) and height-specific concentration columns
Ceilometer MLH data must include "date_stamp" (time stamp) and columns containing "bl_height" (e.g., bl_height_1, bl_height_2)
Time stamps must be parseable to datetime format (adjust pd.to_datetime() parameters if needed)
Encoding Handling
Ozone data uses encoding='utf-8'; MLH data uses encoding='utf-8-sig' (adjust if encountering encoding errors)
Avoid special characters in file paths to prevent loading issues
MLH Data Priority
The code uses the first "bl_height" column (main mixing layer) for overlay; modify bl_height_cols[0] in calculate_phase_ozone_and_blh_stats() to use other layers
Missing Data
NaN values are automatically excluded from statistical calculations
Warnings are printed if no valid MLH or ozone data is found for a category
Troubleshooting
File not found error: Verify file paths in main() match local structure
Encoding errors: Change the encoding parameter in pd.read_csv() (try 'gbk' for Chinese-language files)
Empty statistical output: Check if time ranges in TIME_PERIODS match the time stamps in your data
Plotting issues: Ensure matplotlib backend is properly configured (use plt.switch_backend('TkAgg') if needed)
