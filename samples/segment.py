import mne
import numpy as np
import matplotlib.pyplot as plt
import os

def process_eeg_for_dino(file_path, start_time=60, duration=30):
    """
    读取EDF，截取指定时间段，预处理为(C, N, T)格式保存为.npy。
    同时会生成临时的截取版EDF文件用于验证，随后删除。
    
    参数:
        file_path: EDF文件路径
        start_time: 截取开始时间 (秒)
        duration: 截取时长 (秒)
    """
    
    # ============================
    # 1. 设置参数
    # ============================
    target_channels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    sfreq = 250
    samples_per_patch = 250 
    n_patches = int(duration) # N = 时长(秒)，假设每秒1个Patch(或根据需求调整)
    # 注意：如果你的N固定为30，而duration变了，这里需要逻辑兼容。
    # 这里假设 N 随着 duration 变化 (即 1秒 = 1 patch)
    # 如果必须固定 N=30 但 duration不是30s，你需要修改这里的逻辑。
    # 根据之前的代码 N=30, T=250, Total=7500 => 30秒数据。这里保持一致。
    
    total_samples = samples_per_patch * n_patches 
    
    print(f"--- 处理文件: {file_path} ---")
    
    # ============================
    # 2. 读取与预处理
    # ============================
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # [需求] 打印总时长
    print(f"原始文件总时长: {raw.times[-1]:.2f} 秒")

    # 检查通道
    try:
        raw.pick_channels(target_channels)
        raw.reorder_channels(target_channels)
    except ValueError as e:
        print(f"通道选择错误: {e}")
        return

    # [需求] 截取指定时长
    print(f"正在截取数据: {start_time}s 到 {start_time + duration}s ...")
    raw.crop(tmin=start_time, tmax=start_time+duration, include_tmax=False)
    
    # 获取数据
    data = raw.get_data()
    
    # 检查长度
    if data.shape[1] < total_samples:
        print(f"错误: 数据长度不足。实际: {data.shape[1]} 点, 需要: {total_samples} 点")
        return
    data = data[:, :total_samples] # 裁剪多余的点

    # [需求] 打印 Max 和 Min
    print(f"截取后数据统计: Max = {np.max(data):.4e}, Min = {np.min(data):.4e}")

    # ============================
    # 3. 保存 NPY (保持不变)
    # ============================
    patched_data = data.reshape(len(target_channels), n_patches, samples_per_patch)
    
    base_name = os.path.splitext(file_path)[0]
    npy_save_name = base_name + '.npy'
    np.save(npy_save_name, patched_data)
    print(f"NPY数据已保存至: {npy_save_name}")

    # ============================
    # 4. 保存临时 EDF 并展示
    # ============================
    # [需求] 保存截取段到相同路径下的edf (改名字)
    temp_edf_name = base_name + f'_cropped_{start_time}_{duration}.edf'
    print(f"\n正在导出临时截取EDF: {temp_edf_name}")
    
    # 使用 mne.export 导出 (需要 mne >= 0.24)
    # 如果没有 export_raw，可以使用 mne.io.Raw.save (但在旧版本对EDF支持有限，通常export_raw更好)
    try:
        mne.export.export_raw(temp_edf_name, raw, fmt='edf', overwrite=True)
        print(f"临时EDF文件已创建: {temp_edf_name} (请检查文件夹)")
    except AttributeError:
         # 兼容旧版本MNE，或者无法导出EDF的情况
        print("警告: 当前MNE版本可能不支持直接导出EDF，跳过导出步骤。")

    # ============================
    # 5. 绘图验证 (展示)
    # ============================
    print("正在绘图验证...")
    # 使用刚保存的npy读取回来画图，确保npy文件是好的
    loaded_data = np.load(npy_save_name).reshape(19, -1)
    raw_recon = mne.io.RawArray(loaded_data, raw.info)
    
    raw_recon.plot(
        duration=duration, 
        n_channels=19, 
        # scalings='auto', 
        title=f"Cropped Data ({start_time}-{start_time+duration}s) | Min:{np.min(data):.2e} Max:{np.max(data):.2e}",
        show=True,
        block=True
    )

    # ============================
    # 6. 删除临时 EDF
    # ============================
    if os.path.exists(temp_edf_name):
        os.remove(temp_edf_name)
        print(f"\n[清理] 临时EDF文件已删除: {temp_edf_name}")
    else:
        print("\n[清理] 未找到临时文件，可能未生成。")

if __name__ == "__main__":
    # 请在这里修改为你的实际 .edf 文件路径
    edf_file = "/media/douyl/Disk4T/douyl/IDEA_Lab/Project_BCI/dinov2/samples/aaaaaaaa_s001_t000.edf" 
    
    if os.path.exists(edf_file):
        # [修改] 可以在这里指定开始时间和时长
        process_eeg_for_dino(edf_file, start_time=60, duration=30)
    else:
        print(f"未找到文件: {edf_file}")
        # 生成模拟数据测试代码逻辑
        print("生成模拟数据进行测试...")
        sim_data = np.random.randn(19, 250*100) * 1e-5 # 100秒数据
        info = mne.create_info(
            ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'], 
            250, 'eeg'
        )
        raw_sim = mne.io.RawArray(sim_data, info)
        mne.export.export_raw("test_mock.edf", raw_sim, fmt='edf', overwrite=True)
        
        # 测试：截取从第 10 秒开始，长 30 秒
        process_eeg_for_dino("test_mock.edf", start_time=10, duration=30)
        
        # 清理模拟的源文件
        if os.path.exists("test_mock.edf"): os.remove("test_mock.edf")