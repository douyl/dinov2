import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置参数 =================
# 1. 输入文件路径
FOLDER_PATH = "/media/douyl/Disk4T/douyl/IDEA_Lab/Project_BCI/dinov2/samples"
FILE_NAME = "aaaaaaaa_s001_t000.edf"
INPUT_PATH = os.path.join(FOLDER_PATH, FILE_NAME)

# 2. 输出文件路径 (保存裁剪后的EDF)
OUTPUT_NAME = FILE_NAME.replace(".edf", "_artifact.edf")
OUTPUT_PATH = os.path.join(FOLDER_PATH, OUTPUT_NAME)

# 3. 目标时间段 (03:30 - 04:00)
# START_TIME = 210  # 3 * 60 + 30
# STOP_TIME = 240   # 4 * 60 + 00
# START_TIME = 570  # 9 * 60 + 30
# STOP_TIME = 600   # 10 * 60 + 00
# START_TIME = 840  # 14 * 60 + 00
# STOP_TIME = 870   # 14 * 60 + 30
START_TIME = 960  # 16 * 60 + 00
STOP_TIME = 990   # 16 * 60 + 30

# 4. 伪迹阈值 (微伏)
THRESHOLD_UV = 100.0

# 5. 目标通道
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]
# ===========================================

def process_and_visualize():
    if not os.path.exists(INPUT_PATH):
        print(f"错误: 找不到文件 {INPUT_PATH}")
        return

    print(f"正在加载: {INPUT_PATH}")
    # 1. 读取原始数据
    raw = mne.io.read_raw_edf(INPUT_PATH, preload=True, verbose=False)
    
    # 2. 筛选通道
    print(f"筛选 {len(TARGET_CHANNELS)} 个标准通道...")
    try:
        raw.pick_channels(TARGET_CHANNELS, ordered=True)
    except ValueError as e:
        print(f"通道筛选错误: {e}")
        return

    # 3. 裁切时间段 (Crop)
    print(f"裁切时间: {START_TIME}s - {STOP_TIME}s ...")
    raw.crop(tmin=START_TIME, tmax=STOP_TIME, include_tmax=False)
    
    # ================= 任务一：保存 EDF =================
    print(f"正在保存裁剪后的 EDF 到: {OUTPUT_PATH}")
    mne.export.export_raw(OUTPUT_PATH, raw, fmt='edf', overwrite=True)
    print("EDF 保存成功。")

    # ================= 任务二：可视化 (plt.show) =================
    print("-" * 50)
    print("正在准备可视化...")
    
    # 获取数据并转为微伏
    data, times = raw.get_data(return_times=True)
    data_uv = data * 1e6 
    
    # 调整时间轴显示为相对时间 (或者绝对时间，看你喜好，这里用相对时间 0-30s 方便看)
    # 如果想看绝对时间，就用 times + START_TIME
    plot_times = times + START_TIME

    # 找出包含伪迹的通道
    artifact_channels = []
    for i, ch_name in enumerate(TARGET_CHANNELS):
        if np.max(np.abs(data_uv[i])) > THRESHOLD_UV:
            artifact_channels.append(i)

    if not artifact_channels:
        print("注意: 该片段内没有发现 > 100uV 的数据点，将显示所有通道。")
        channels_to_plot = list(range(len(TARGET_CHANNELS)))
    else:
        print(f"发现 {len(artifact_channels)} 个通道存在伪迹，将只显示这些通道。")
        channels_to_plot = artifact_channels

    # 开始绘图
    num_plots = len(channels_to_plot)
    # 动态调整图表高度
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]

    for plot_idx, ch_idx in enumerate(channels_to_plot):
        ax = axes[plot_idx]
        ch_name = TARGET_CHANNELS[ch_idx]
        signal = data_uv[ch_idx]
        
        # 1. 画波形
        ax.plot(plot_times, signal, color='black', linewidth=0.8, label=ch_name)
        
        # 2. 标红超阈值点
        bad_mask = np.abs(signal) > THRESHOLD_UV
        if np.any(bad_mask):
            ax.scatter(plot_times[bad_mask], signal[bad_mask], color='red', s=15, zorder=5, label='>100uV')
            
            # 打印一下具体的最大值信息
            max_val = np.max(np.abs(signal))
            print(f"  --> 通道 {ch_name}: 最大幅值 {max_val:.2f} uV")

        # 3. 画阈值线
        ax.axhline(y=THRESHOLD_UV, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-THRESHOLD_UV, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(f"{ch_name} (uV)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.xlabel("Time (s)")
    plt.suptitle(f"Artifact Analysis ({START_TIME}s - {STOP_TIME}s)", fontsize=14)
    plt.tight_layout()
    
    print("正在显示图像窗口 (请检查弹窗)...")
    plt.show()

if __name__ == "__main__":
    process_and_visualize()