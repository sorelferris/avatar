import time
import numpy as np
import matplotlib.pyplot as plt
from rich.live import Live

from src.joycon_utils import JoyCon


def main():
    joycon = JoyCon()

    # 1. 初始化一个超简单的 2D 极坐标系（就像电影里的雷达扫描仪，长相非常高级）
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    ax.set_facecolor("#111115")  # 深色背景
    fig.patch.set_facecolor("#111115")

    # 初始化雷达指针（L蓝色，R红色）
    (line_L,) = ax.plot([0, 0], [0, 1], color="#00d2ff", linewidth=4, label="L")
    (line_R,) = ax.plot([0, 0], [0, 1], color="#ff3b30", linewidth=4, label="R")
    ax.set_rmax(1.0)  # 半径最大为1（单位向量）
    ax.grid(True, color="#333338")
    ax.tick_params(colors="white")

    while True:
        try:
            imu = joycon.get_imu()
            dir_L = imu["L"]["direction"]
            dir_R = imu["R"]["direction"]

            # 2. 关键数学转换：把 3D 向量映射到 2D 罗盘平面
            # 计算水平面上的夹角（弧度）
            angle_L = np.arctan2(dir_L.y, dir_L.x)
            angle_R = np.arctan2(dir_R.y, dir_R.x)

            # 计算倾斜投影长度（如果手柄完全垂直朝上，指针就会缩短到中心）
            len_L = np.sqrt(dir_L.x**2 + dir_L.y**2)
            len_R = np.sqrt(dir_R.x**2 + dir_R.y**2)

            # 3. 极速更新 2D 雷达指针，零延迟
            line_L.set_data([angle_L, angle_L], [0, len_L])
            line_R.set_data([angle_R, angle_R], [0, len_R])

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.03)  # ~30Hz，极其丝滑

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
