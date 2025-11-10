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

# 定义时间段配置（采用代码二的分类）
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

# 定义颜色配置（采用代码二的颜色）
COLORS = {
    'Clean night': '#1f77b4',  # 蓝色
    'Clean daytime': '#2ca02c',  # 绿色
    'Polluted night': '#ff7f0e',  # 橙色
    'Polluted daytime': '#d62728'  # 红色
}


def parse_ozone_data(input_file_path):
    """读取臭氧浓度数据文件"""
    try:
        # 读取数据
        data = pd.read_csv(input_file_path, encoding='utf-8')

        # 将 "监测时间" 列转换为 datetime 格式
        data['监测时间'] = pd.to_datetime(data['监测时间'])

        # 获取高度值（列名从第二列开始）
        all_heights = [float(h) for h in data.columns[1:].tolist()]

        # 过滤掉 300 m 以下的高度
        height_mask = [h >= 300 for h in all_heights]
        heights = [h for h in all_heights if h >= 300]

        # 保留时间列和 300 m 以上高度的数据列
        columns_to_keep = ['监测时间'] + [data.columns[i + 1] for i, keep in enumerate(height_mask) if keep]
        data = data[columns_to_keep]

        print(f"成功读取臭氧浓度数据，记录数量: {len(data)}")
        print(f"原始高度层数: {len(all_heights)}，范围: {min(all_heights)}-{max(all_heights)} 米")
        print(f"过滤后高度层数: {len(heights)}，范围: {min(heights)}-{max(heights)} 米")

        return data, heights
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None


def process_mixing_layer_height_data(input_mixing_layer_height_folder_path):
    """处理三层混合层高度数据"""
    blh_dfs = []
    for input_file in Path(input_mixing_layer_height_folder_path).glob('*.CSV'):
        try:
            df_mlh = pd.read_csv(input_file, skiprows=[1], encoding='utf-8-sig')
            df_mlh['date_stamp'] = df_mlh['date_stamp'].str.strip('\ufeff')
            df_mlh['date_stamp'] = df_mlh['date_stamp'].str.strip("b'")
            df_mlh['datetime'] = pd.to_datetime(df_mlh['date_stamp'])
            df_mlh.replace(-999, float('nan'), inplace=True)

            # 检查是否有多列bl_height
            bl_height_cols = [col for col in df_mlh.columns if 'bl_height' in col]
            print(f"找到的bl_height列: {bl_height_cols}")

            # 选择需要的列：datetime和所有bl_height列
            cols_to_keep = ['datetime'] + bl_height_cols
            blh_dfs.append(df_mlh[cols_to_keep])

        except Exception as e:
            print(f"处理混合层高度文件 {input_file.name} 时出错: {str(e)}")

    if not blh_dfs:
        print("警告: 未找到有效的混合层高度数据")
        return pd.DataFrame()

    combined_df = pd.concat(blh_dfs)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    combined_df['date'] = combined_df['datetime'].dt.date
    combined_df['hour'] = combined_df['datetime'].dt.hour

    # 获取所有bl_height列名
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

                # 对每一层混合层高度计算小时平均
                for col in bl_height_cols:
                    if col in hour_data.columns:
                        avg_height = hour_data[col].mean()
                        hourly_record[col] = avg_height

                hourly_data.append(hourly_record)

    hourly_df = pd.DataFrame(hourly_data)
    if not hourly_df.empty:
        hourly_df = hourly_df.sort_values('datetime').reset_index(drop=True)

        # 打印每层混合层高度的范围
        for col in bl_height_cols:
            if col in hourly_df.columns:
                min_val = hourly_df[col].min()
                max_val = hourly_df[col].max()
                print(f"{col}范围: {min_val:.1f} 至 {max_val:.1f} 米")

        return hourly_df
    else:
        print("警告: 小时平均后数据为空")
        return pd.DataFrame()


def calculate_phase_ozone_and_blh_stats(ozone_data, heights, blh_data=None):
    """
    按照时间段对臭氧数据和混合层高度数据进行分类和统计

    Returns:
    --------
    dict: 包含每个阶段和时间段的统计数据
    """
    results = {}

    for phase, time_periods in TIME_PERIODS.items():
        results[phase] = {}

        for period_type, time_ranges in time_periods.items():
            # 收集该时间段的所有臭氧数据
            all_ozone_data = [[] for _ in range(len(heights))]  # 每个高度一个列表
            all_blh_data = []  # 混合层高度数据

            for start_str, end_str in time_ranges:
                start_time = pd.to_datetime(start_str)
                end_time = pd.to_datetime(end_str)

                # 获取该时间段内的臭氧数据索引
                period_indices = (ozone_data['监测时间'] >= start_time) & (ozone_data['监测时间'] <= end_time)

                # 如果该时间段有臭氧数据，收集数据
                if period_indices.any():
                    # 提取该时间段的臭氧数据（排除时间列）
                    period_ozone_data = ozone_data.loc[period_indices, ozone_data.columns[1:]].values

                    # 将数据添加到各个高度的列表
                    for j in range(period_ozone_data.shape[1]):
                        # 去除 NaN 值
                        valid_data = period_ozone_data[:, j][~np.isnan(period_ozone_data[:, j])]
                        all_ozone_data[j].extend(valid_data)

                # 如果有混合层高度数据，也进行收集
                if blh_data is not None and not blh_data.empty:
                    blh_period_indices = (blh_data['datetime'] >= start_time) & (blh_data['datetime'] <= end_time)
                    if blh_period_indices.any():
                        # 获取第一层混合层高度数据（通常是主要的混合层）
                        bl_height_cols = [col for col in blh_data.columns if 'bl_height' in col]
                        if len(bl_height_cols) > 0:
                            period_blh_data = blh_data.loc[blh_period_indices, bl_height_cols[0]].dropna().values
                            all_blh_data.extend(period_blh_data)

            # 计算每个高度的臭氧平均值和标准差
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

            # 计算混合层高度的平均值和标准差
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
    绘制四个阶段的臭氧浓度垂直廓线图，并叠加混合层高度

    Parameters:
    -----------
    phase_stats : dict
        包含各阶段统计数据的字典
    heights : list
        高度列表
    output_folder_path : Path
        输出文件夹路径
    figsize : tuple
        图片大小
    xlim : tuple or None
        x轴范围
    """
    # 确保输出文件夹存在
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # 高度转换为千米
    heights_km = [h / 1000 for h in heights]

    # 创建1x4的子图
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    # 设置x轴范围
    if xlim is None:
        xlim = (0, 200)

    # 绘制每个阶段
    phase_names = list(TIME_PERIODS.keys())

    for i, phase in enumerate(phase_names):
        ax = axes[i]

        # 绘制每个时间段的臭氧廓线
        for period_type in TIME_PERIODS[phase].keys():
            if period_type in phase_stats[phase]:
                # 提取该时间段的臭氧平均值和标准差
                means = []
                stds = []
                valid_heights = []

                for height in heights:
                    height_stats = phase_stats[phase][period_type]['ozone'][str(height)]
                    if not np.isnan(height_stats['mean']) and height_stats['count'] > 0:
                        means.append(height_stats['mean'])
                        stds.append(height_stats['std'])
                        valid_heights.append(height / 1000)  # 转换为千米

                if len(means) > 0:
                    means = np.array(means)
                    stds = np.array(stds)
                    valid_heights = np.array(valid_heights)

                    # 绘制平均值线
                    color = COLORS[period_type]
                    ax.plot(means, valid_heights, color=color, linewidth=3,
                            label=period_type, linestyle='-')

                    # 绘制误差范围（填充）
                    ax.fill_betweenx(valid_heights,
                                     means - stds, means + stds,
                                     color=color, alpha=0.3)

                # 绘制混合层高度（水平线）
                blh_stats = phase_stats[phase][period_type]['blh']
                if not np.isnan(blh_stats['mean']) and blh_stats['count'] > 0:
                    blh_height_km = blh_stats['mean'] / 1000.0  # 转换为千米
                    blh_std_km = blh_stats['std'] / 1000.0

                    # 绘制混合层高度平均线
                    ax.axhline(y=blh_height_km, color=color, linewidth=2,
                               linestyle='--', alpha=0.8)

                    # # 绘制混合层高度误差范围（阴影）
                    # ax.axhspan(blh_height_km - blh_std_km,
                    #            blh_height_km + blh_std_km,
                    #            color=color, alpha=0.1)

        # 设置子图属性
        ax.set_title(phase, fontsize=18, fontweight='bold', pad=8)

        # 添加子图标签 (a), (b), (c), (d)
        ax.text(0.99, 0.99, f'({chr(97 + i)})', transform=ax.transAxes,
                fontsize=18, fontweight='bold', ha='right', va='top')

        # 设置x轴
        ax.set_xlim(xlim)
        x_ticks = np.arange(0, xlim[1] + 1, 50)
        ax.set_xticks(x_ticks)

        # 设置y轴（Y轴范围，从 0.3 km 开始（300 m 以上））
        ax.set_ylim(0, 1.8)
        yticks = np.append(0.3, np.arange(0.5, 1.9, 0.5))
        ax.set_yticks(yticks)

        # 设置刻度
        ax.tick_params(axis='both', which='major', direction='in',
                       length=6, labelsize=18)

        # 设置刻度标签的字体粗细
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # 设置Y轴标签（只有第一个子图显示）
        if i == 0:
            ax.set_ylabel('Height (km, AGL)', fontsize=18,
                          labelpad=8, fontweight='bold')

        # 添加图例（只在最后一个子图中显示）
        if i == 3:
            # 创建包含所有类别的图例
            legend_elements = []
            for period_type, color in COLORS.items():
                # # 臭氧廓线（实线）
                # legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3,
                #                                   label=f'{period_type} O₃', linestyle='-'))
                # 混合层高度（虚线）
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                                  label=f'{period_type} MLH', linestyle='--', alpha=0.8))

            # ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0),
            #           loc='upper left', fontsize=12, frameon=False, ncol=1)

    # 在整个图的中央底部添加X轴标签
    fig.text(0.5, -0.05, r'Ozone concentration (μg/m$^\mathbf{3}$)',
             ha='center', va='bottom', fontsize=18, fontweight='bold')

    # 保存图片
    output_file = output_folder_path / "臭氧浓度廓线_混合层高度_四阶段_混合层高度_0.3_1.5.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"已生成四个阶段的臭氧浓度垂直廓线图（含混合层高度）: {output_file}")

    return fig, axes


def main():
    # 文件路径配置
    input_folder_path = Path(
        r"C:\Users\房泽\Desktop\Analysis of Vertical Structural Characteristics of a Spring Ozone Pollution in the Pearl River Delta Based on Vertical Sounding Observations by Drones and LiDARs\数据\臭氧雷达数据\2025_03_21-27")
    ozone_file_path = input_folder_path / "番禺大学城-臭氧激光雷达-臭氧浓度数据.csv"

    # 混合层高度数据路径
    blh_folder_path = Path(
        r"C:\Users\房泽\Desktop\Analysis of Vertical Structural Characteristics of a Spring Ozone Pollution in the Pearl River Delta Based on Vertical Sounding Observations by Drones and LiDARs\数据\地表气象要素\三层\21-27")

    output_folder_path = input_folder_path

    # 读取臭氧浓度数据
    print("正在读取臭氧浓度数据...")
    ozone_data, heights = parse_ozone_data(ozone_file_path)
    if ozone_data is None:
        print("无法加载臭氧浓度数据，请检查文件路径和格式。")
        return

    # 读取混合层高度数据
    print("正在读取混合层高度数据...")
    blh_data = process_mixing_layer_height_data(blh_folder_path)
    if blh_data.empty:
        print("警告: 未能读取混合层高度数据，将只绘制臭氧廓线")
        blh_data = None

    # 按时间段分类臭氧和混合层高度数据
    print("\n正在按时间段分类数据...")
    phase_stats = calculate_phase_ozone_and_blh_stats(ozone_data, heights, blh_data)

    # 输出统计信息
    print("\n========== 各阶段臭氧浓度和混合层高度统计信息 ==========")
    for phase, phase_data in phase_stats.items():
        print(f"\n{phase}:")
        for period_type, period_data in phase_data.items():
            # 臭氧浓度统计
            ozone_data = period_data['ozone']
            valid_means = [data['mean'] for data in ozone_data.values()
                           if not np.isnan(data['mean']) and data['count'] > 0]
            if len(valid_means) > 0:
                avg_ozone = np.mean(valid_means)
                ozone_count = sum([data['count'] for data in ozone_data.values()])
                print(f"  {period_type}: 平均臭氧浓度 = {avg_ozone:.2f} μg/m³, 数据点数 = {ozone_count}")
            else:
                print(f"  {period_type}: 无有效臭氧数据")

            # 混合层高度统计
            blh_data = period_data['blh']
            if not np.isnan(blh_data['mean']) and blh_data['count'] > 0:
                print(
                    f"                平均混合层高度 = {blh_data['mean']:.2f} ± {blh_data['std']:.2f} m, 数据点数 = {blh_data['count']}")
            else:
                print(f"                无有效混合层高度数据")

    print("=" * 50)

    # 绘制臭氧浓度垂直廓线和混合层高度
    print("\n正在生成臭氧浓度垂直廓线图（含混合层高度）...")
    fig, axes = plot_phase_ozone_profiles_with_blh(phase_stats, heights, output_folder_path)

    print("臭氧浓度垂直廓线图（含混合层高度，按阶段分类）已生成完毕！")

    return phase_stats


if __name__ == "__main__":
    main()