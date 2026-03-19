# -------------------------------------------------------
# @Author: vegesin
# @Date: 2026-03-19
# @FilePath: \radar_pc\src\datasets\mmradar_pc.py
# @Description: 毫米雷达点云数据集 包含数解析相关函数/训练评估使用dataset类
# -------------------------------------------------------


import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple


# --------------------------------------------------------
# 点云数据解析基础函数
# --------------------------------------------------------

def load_pkl(file_path: str) -> List[np.ndarray]:
    """
    ### @ 加载 pkl 雷达点云数据文件
    
  
    读取出来的数据集格式: 2维list: raw_data[num_frame][num_point] | [X,Y,Z,V,S]
    
    示例:
    raw_data[1].shape
    (59, 5)
    raw_data[2].shape
    (42, 5)

    Args:
        file_path: .pkl 文件路径

    Returns:
        包含每一帧点云数据的列表

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 读取到的数据不是列表时抛出。
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a list of frames, but got {type(raw_data)}.")
        return raw_data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")


def read_pkl_perframe(file_path: str, frame_index: int, plot_scatter: bool = False, color_by: str = 'V') -> np.ndarray:
    """
    ### @ 读取指定帧的雷达点云数据，并可选择进行可视化。

    Args:
        file_path: .pkl 文件路径。
        frame_index: 要提取的帧索引。
        plot_scatter: 为 True 时，绘制该帧的 3D 散点图。
        color_by: 颜色映射方式，'V' 表示速度，'S' 表示 SNR。

    Returns:
        指定帧的点云数据，形状为 (N_i, 5) 的 numpy 数组。

    Raises:
        IndexError: frame_index 超出范围时抛出。
    """
    raw_data = load_pkl(file_path)

    if frame_index < 0 or frame_index >= len(raw_data):
        raise IndexError(f"Frame index {frame_index} is out of bounds. "
                         f"The file contains {len(raw_data)} frames.")

    # 提取当前帧并确保为 numpy 数组
    frame_data = np.array(raw_data[frame_index])

    if frame_data.ndim != 2 or frame_data.shape[1] != 5:
        # 如果该帧为空或格式异常，尝试处理
        if frame_data.size == 0:
            frame_data = np.empty((0, 5))
        else:
            raise ValueError(f"Frame {frame_index} has an unexpected shape: {frame_data.shape}")

    # 可视化单帧
    if plot_scatter:
        if frame_data.shape[0] == 0:
            print(f"Warning: Frame {frame_index} is empty. Nothing to plot.")
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]

            if color_by.upper() == 'S':
                c_data, c_label, cmap = frame_data[:, 4], 'Signal Strength (SNR)', 'plasma'
            else:
                c_data, c_label, cmap = frame_data[:, 3], 'Doppler Velocity (m/s)', 'jet'

            scatter = ax.scatter(x, y, z, c=c_data, cmap=cmap, marker='o', s=20, alpha=0.8)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Radar Point Cloud - Frame {frame_index}')

            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(c_label)
            plt.show()

    return frame_data


def show_pkl_allframe(file_path: str, interval_ms: int = 60, color_by: str = 'V') -> None:
    """
    ### @ 读取一个pkl文件,播放所有帧,帧间采集间隔60ms

    Args:
        file_path: .pkl 文件路径。
        interval_ms: 帧间延迟（毫秒），默认值为 60。
        color_by: 颜色映射方式，'V' 表示速度，'S' 表示 SNR。
    """
    raw_data = load_pkl(file_path)

    # 过滤掉空帧并将所有帧转换为统一的 numpy 格式
    frames = [np.array(f) for f in raw_data if len(f) > 0 and np.array(f).shape[1] == 5]

    if not frames:
        print("No valid frames found to play.")
        return

    # 将所有点拼接起来，用于计算全局的坐标轴边界和颜色边界
    all_points = np.vstack(frames)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    if color_by.upper() == 'S':
        c_min, c_max = all_points[:, 4].min(), all_points[:, 4].max()
        c_label, cmap = 'Signal Strength (SNR)', 'plasma'
        c_idx = 4
    else:
        c_min, c_max = all_points[:, 3].min(), all_points[:, 3].max()
        c_label, cmap = 'Doppler Velocity (m/s)', 'jet'
        c_idx = 3

    # 初始化绘图窗口
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 固定坐标轴范围，防止播放时画面抖动缩放
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # 初始化第一帧数据
    first_frame = frames[0]
    scatter = ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], c=first_frame[:, c_idx], cmap=cmap, vmin=c_min, vmax=c_max, marker='o', s=20, alpha=0.8)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(c_label)

    title = ax.set_title(f'Playing Radar Frames (0/{len(frames)})')

    # 更新函数，用于 FuncAnimation
    def update(frame_idx: int):
        current_frame = frames[frame_idx]

        # 在 Matplotlib 中更新 3D 散点图的数据和颜色
        # _offsets3d 是底层的属性，要求传入一个元组 (x_array, y_array, z_array)
        scatter._offsets3d = (current_frame[:, 0], current_frame[:, 1], current_frame[:, 2])
        # 更新颜色
        scatter.set_array(current_frame[:, c_idx])

        title.set_text(f'Playing Radar Frames ({frame_idx + 1}/{len(frames)})')
        return scatter, title

    # 创建动画对象
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True                # 播放结束后是否循环播放
    )

    plt.show()



# --------------------------------------------------------
# TODO : 点云数据加载器
# --------------------------------------------------------