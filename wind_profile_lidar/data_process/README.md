TKE Calculation Tool for Wind Profiler Data
A simple Python script to calculate hourly Turbulent Kinetic Energy (TKE) from wind profiler observations, including vertical velocity, horizontal wind speed, and wind direction data.
Overview
This tool processes three types of wind profiler data (vertical velocity, horizontal speed, and horizontal direction) to compute hourly-averaged Turbulent Kinetic Energy (TKE) at different height levels. TKE is calculated using the formula:TKE = 0.5 * (u'² + v'² + w'²)where u', v', and w' are the turbulent fluctuations (variances) of the zonal, meridional, and vertical wind components, respectively.
Input Data Requirements
The script requires 3 CSV files with the following structure:
Vertical velocity data
Column 1: Time stamp (column name: 监测时间, in datetime format)
Columns 2+: Vertical velocity values at different heights (column names as height labels)
Horizontal wind speed data
Same structure as vertical velocity data, with horizontal speed values
Horizontal wind direction data
Same structure as vertical velocity data, with wind direction values (meteorological convention: 0° = north, 90° = east)
Missing values should be marked as 999 (will be converted to NaN automatically)
Dependencies
Install required packages first:
bash
pip install pandas numpy
Usage
Prepare your input CSV files and place them in a directory (e.g., ./input_data/wind_profiler/).
Modify the file paths in the if __name__ == "__main__": section to match your data location:
python
运行
input_dir = Path("./your_input_directory")
vertical_vel_file = input_dir / "vertical_velocity_data.csv"
horiz_speed_file = input_dir / "horizontal_speed_data.csv"
horiz_dir_file = input_dir / "horizontal_direction_data.csv"
output_file = input_dir / "hourly_tke_results.csv"
Run the script:
bash
python tke_calculation.py
Output
A CSV file (hourly_tke_results.csv) with:
observation_time: Hourly time stamps (start of each hour)
Columns for each height: Calculated TKE values (units depend on input wind speed units, typically m²/s²)
Notes
The script filters out hours with fewer than 4 valid data points (to ensure reliable variance calculation).
Missing TKE values are marked as NaN.
Wind direction is converted from meteorological convention (0° = north) to mathematical coordinate system for component calculation.
