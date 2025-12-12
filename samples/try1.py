import numpy as np
import mne
import matplotlib.pyplot as plt

# ================= 配置 =================
# 你的 npy 文件路径
FILE_PATH = "/home/douyl/Downloads/aaaaaaaa_s001_t000.npy"

# 之前的标准 19 通道列表 (顺序必须对应)
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]
SFREQ = 250
# ========================================

def main():
    # 1. 读取 NPY
    print(f"Loading {FILE_PATH} ...")
    data = np.load(FILE_PATH)
    
    # 2. 查看维度
    print("-" * 30)
    print(f"原始数据维度 (Shape): {data.shape}")
    print("解释: (Segment数量, 通道数, Patch数量, Patch内点数)")
    
    # 3. 检查数值量级 (Volts vs uV)
    max_val = np.max(np.abs(data))
    mean_val = np.mean(np.abs(data))
    print(f"最大绝对值: {max_val:.2e}")
    print(f"平均绝对值: {mean_val:.2e}")
    
    is_volts = True
    if max_val > 1.0: 
        print(">> 判断: 数据单位看起来是 微伏 (μV)。(数值 > 1)")
        is_volts = False
    else:
        print(">> 判断: 数据单位看起来是 伏特 (Volts)。(数值 < 1)")
        is_volts = True

    # 4. 还原为 EEG 格式 (Epochs)
    # 输入维度: (N, 19, 30, 250)
    # 目标维度: (N, 19, 30 * 250) = (N, 19, 7500)
    n_segs, n_chs, n_patches, n_times = data.shape
    
    # Reshape: 合并最后两个维度
    data_reshaped = data.reshape(n_segs, n_chs, -1)
    print(f"重塑后维度: {data_reshaped.shape} (用于 MNE)")

    # 如果数据是 uV，MNE 需要 Volts 才能正确显示比例尺，所以转回 Volts
    if not is_volts:
        print("正在将数据从 uV 转换为 V 以适配 MNE...")
        data_reshaped = data_reshaped * 1e-6

    # 5. 创建 MNE 对象
    print("创建 MNE Epochs 对象...")
    info = mne.create_info(ch_names=TARGET_CHANNELS, sfreq=SFREQ, ch_types='eeg')
    
    # 创建 EpochsArray
    epochs = mne.EpochsArray(data_reshaped, info)
    
    # 设置标准蒙太奇 (可选，用于拓扑图)
    try:
        epochs.set_montage('standard_1020')
    except:
        pass

    # 6. 可视化
    print("正在打开可视化窗口...")
    # scalings='auto' 会自动调整缩放
    # n_epochs=3 只显示前3个 segment，你可以改大
    epochs.plot(
        scalings='auto', 
        n_epochs=5, 
        n_channels=19, 
        title=f"Visualization of {FILE_PATH}",
        show=True,
        block=True
    )

if __name__ == "__main__":
    main()