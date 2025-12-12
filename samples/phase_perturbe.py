import mne
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. 定义文件路径
base_path = '/media/douyl/Disk4T/douyl/IDEA_Lab/Project_BCI/dinov2/samples/'
input_file = os.path.join(base_path, 'aaaaaaaa_s001_t000.edf')
output_crop_file = os.path.join(base_path, 'aaaaaaaa_s001_t000_select.edf')
output_aug_file = os.path.join(base_path, 'aaaaaaaa_s001_t000_select_phase.edf')

# 定义目标通道
target_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

def apply_phase_perturbation(raw, perturbation_strength=0.1):
    """
    对 Raw 对象的数据进行相位扰动
    """
    data = raw.get_data()
    
    # FFT
    fft_data = np.fft.rfft(data, axis=-1)
    
    # 提取幅度与相位
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    # --- 关键点：添加随机扰动 ---
    # perturbation_strength 是标准差。
    # 2.0 意味着相位会在很大范围内波动，波形改变会非常明显。
    noise = np.random.normal(0, perturbation_strength, phase.shape)
    perturbed_phase = phase + noise
    
    # 重构
    perturbed_fft = amplitude * np.exp(1j * perturbed_phase)
    perturbed_data = np.fft.irfft(perturbed_fft, n=data.shape[-1], axis=-1)
    
    # 创建新对象
    raw_aug = mne.io.RawArray(perturbed_data, raw.info)
    raw_aug.set_annotations(raw.annotations)
    
    return raw_aug

def verify_and_plot_difference(raw_orig, raw_aug, channel_index=0):
    """
    数值验证并绘图对比
    """
    data_orig = raw_orig.get_data()
    data_aug = raw_aug.get_data()
    
    # 1. 数值验证
    diff = data_orig - data_aug
    max_diff = np.max(np.abs(diff))
    mse = np.mean(diff ** 2)
    
    print("\n" + "="*30)
    print("       数据差异验证报告")
    print("="*30)
    
    if np.allclose(data_orig, data_aug):
        print("!!! 警告：数据完全相同（扰动失败） !!!")
    else:
        print(f"✅ 数据不相等 (验证通过)")
        print(f"   最大绝对差异值: {max_diff:.2e}")
        print(f"   均方误差 (MSE): {mse:.2e}")
    
    print("="*30 + "\n")

    # 2. 绘图验证 (取第一个通道的前 5 秒)
    sfreq = raw_orig.info['sfreq']
    n_samples_plot = int(5 * sfreq) # 只画前5秒
    times = np.arange(n_samples_plot) / sfreq
    
    ch_name = raw_orig.ch_names[channel_index]
    y_orig = data_orig[channel_index, :n_samples_plot]
    y_aug = data_aug[channel_index, :n_samples_plot]
    y_diff = diff[channel_index, :n_samples_plot]
    
    plt.figure(figsize=(12, 6))
    
    # 子图1：叠加对比
    # plt.subplot(2, 1, 1)
    plt.plot(times, y_orig, label='Original', color='black', alpha=0.7, linewidth=1)
    plt.plot(times, y_aug, label='Perturbed (Augmented)', color='red', alpha=0.7, linewidth=1, linestyle='--')
    plt.title(f'Channel: {ch_name} (Overlay)')
    plt.legend()
    
    # 子图2：纯差异
    # plt.subplot(2, 1, 2)
    # plt.plot(times, y_diff, color='blue', linewidth=1)
    # plt.title(f'Difference (Original - Perturbed) | Max Diff: {np.max(np.abs(y_diff)):.2e}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude Diff')
    
    plt.tight_layout()
    plt.show()

def main():
    # --- 读取与裁剪 ---
    print(f"正在加载文件: {input_file}")
    raw = mne.io.read_raw_edf(input_file, preload=True, verbose=False)
    
    # 选择通道
    available_chans = [ch for ch in target_channels if ch in raw.ch_names]
    raw.pick(available_chans)
    
    # 裁剪 (60-90s)
    raw.crop(tmin=60, tmax=90, include_tmax=False)
    
    # 保存原始裁剪版
    mne.export.export_raw(output_crop_file, raw, fmt='edf', overwrite=True)
    
    # --- 数据增强 ---
    # 【修改点】：将强度调整为 2.0 (幅度很大，足以肉眼可见)
    # 之前的 0.2 太小，波形看起来几乎重合
    strength = 0.2
    print(f"正在进行相位扰动增强 (强度={strength})...")
    
    raw_aug = apply_phase_perturbation(raw, perturbation_strength=strength)
    
    # 保存增强版
    mne.export.export_raw(output_aug_file, raw_aug, fmt='edf', overwrite=True)
    
    # --- 验证与绘图 ---
    verify_and_plot_difference(raw, raw_aug, channel_index=0)
    
    print(f"处理完成！文件已保存:\n1. {output_crop_file}\n2. {output_aug_file}")

if __name__ == "__main__":
    main()