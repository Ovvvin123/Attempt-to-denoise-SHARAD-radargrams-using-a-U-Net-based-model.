from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 根据轨迹名称寻找 GEOM.TAB，并读取经纬度
# =========================


def get_geom_latlon(track_name, geom_root, lat_col=2, lon_col=3):
    """
    根据轨迹名称找到对应的 GEOM.TAB 文件，并读取经纬度。

    适用于这种结构：
        geom_data_total/
            s_0016xx/
                s_00168901_geom.tab
                s_00168901_geom.lbl
                s_00168901_geom.xml

    参数：
        track_name:
            可以是：
            's_00168901_tiff.tif'
            's_00168901_rgram.img'
            's_00168901_geom.tab'
            '00168901'

        geom_root:
            GEOM 总文件夹，例如：
            r"E:\\python\\pytorchProject1\\geom_data_total"

        lat_col:
            纬度所在列，默认 2，表示第 3 列。

        lon_col:
            经度所在列，默认 3，表示第 4 列。

    返回：
        geom_path, lat, lon
    """
    geom_root = Path(geom_root)

    # 提取 8 位轨道编号
    m = re.search(r"(\d{8})", str(track_name))
    if m is None:
        raise ValueError(f"轨迹名称中没有找到 8 位编号：{track_name}")

    track_id = m.group(1)

    # 例如 00168901 -> s_0016xx
    subfolder_name = f"s_{track_id[:4]}xx"

    # 例如 s_00168901_geom.tab
    geom_filename = f"s_{track_id}_geom.tab"

    geom_path = geom_root / subfolder_name / geom_filename

    if geom_path.exists():
        tab_path = geom_path
    else:
        # 如果直接构造失败，再用递归搜索兜底
        candidates = list(geom_root.rglob(f"s_{track_id}_geom.tab"))

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"没有找到 {track_id} 对应的 GEOM.TAB。\n"
                f"尝试路径为：{geom_path}"
            )

        tab_path = candidates[0]

    df = pd.read_csv(
        tab_path,
        header=None,
        sep=r"\s*,\s*|\s+",
        engine="python",
        comment="#"
    )

    lat = pd.to_numeric(df.iloc[:, lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(df.iloc[:, lon_col], errors="coerce").to_numpy()

    mask = np.isfinite(lat) & np.isfinite(lon)
    return tab_path, lat[mask], lon[mask]


# =========================
# 2. 火星北极极射赤面投影
# =========================
def mars_north_polar_stereo(lat_deg, lon_deg, lon0_deg=0):
    """
    把火星经纬度转换到北极平面坐标。

    lat_deg: 纬度，单位 degree
    lon_deg: 经度，单位 degree，0-360 或 -180-180 都可以
    lon0_deg: 中央经线，默认 0°E

    输出：
        x, y，单位 km
    """
    R = 3396190.0  # 火星近似半径，单位 m

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lon0 = np.deg2rad(lon0_deg)

    rho = 2 * R * np.tan(np.pi / 4 - lat / 2)

    x = rho * np.sin(lon - lon0)
    y = -rho * np.cos(lon - lon0)

    return x / 1000, y / 1000


# =========================
# 3. 画北极经纬网
# =========================
def draw_mars_polar_grid(ax, lat_min=70, lon0_deg=0):
    # 纬度圈
    for lat in [70, 75, 80, 85]:
        lon = np.linspace(0, 360, 720)
        lat_arr = np.full_like(lon, lat)
        x, y = mars_north_polar_stereo(lat_arr, lon, lon0_deg)
        ax.plot(x, y, color="gray", linewidth=0.7, linestyle="--")

        # 标注纬度
        x_text, y_text = mars_north_polar_stereo(np.array([lat]), np.array([45]), lon0_deg)
        ax.text(x_text[0], y_text[0], f"{lat}°N", fontsize=9)

    # 经度线
    for lon in range(0, 360, 30):
        lat = np.linspace(lat_min, 90, 200)
        lon_arr = np.full_like(lat, lon)
        x, y = mars_north_polar_stereo(lat, lon_arr, lon0_deg)
        ax.plot(x, y, color="gray", linewidth=0.7, linestyle="--")

        # 标注经度
        x_text, y_text = mars_north_polar_stereo(np.array([lat_min]), np.array([lon]), lon0_deg)
        ax.text(x_text[0], y_text[0], f"{lon}°E", fontsize=8, ha="center", va="center")


# =========================
# 4. 主函数：画某条轨迹
# =========================
def plot_sharad_track(track_name, geom_dir, lat_col=2, lon_col=3, lat_min=70):
    geom_file, lat, lon = get_geom_latlon(
        track_name,
        geom_dir,
        lat_col=lat_col,
        lon_col=lon_col
    )
    print("读取文件：", geom_file)

    # 只保留北纬 lat_min 以上
    mask = lat >= lat_min
    lat_plot = lat[mask]
    lon_plot = lon[mask]

    if len(lat_plot) == 0:
        raise ValueError(f"这条轨迹没有经过 {lat_min}°N 以上区域。")

    x, y = mars_north_polar_stereo(lat_plot, lon_plot)

    fig, ax = plt.subplots(figsize=(8, 8))

    draw_mars_polar_grid(ax, lat_min=lat_min)

    ax.plot(x, y, linewidth=2, label=track_name)
    ax.scatter(x[0], y[0], s=40, label="start")
    ax.scatter(x[-1], y[-1], s=40, label="end")

    # 设置显示范围
    r_lim = 2 * 3396190.0 * np.tan(np.pi / 4 - np.deg2rad(lat_min) / 2) / 1000
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)

    ax.set_aspect("equal")
    ax.set_title(f"SHARAD Ground Track on Mars North Pole\n{track_name}")
    ax.legend()
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")

    plt.show()



if __name__ == "__main__":
    geom_dir = r"E:\python\pytorchProject1\GEOM"
    track_name = input("请输入轨迹名，例如 00172101 或 s_00172101_geom：").strip()

    plot_sharad_track(
        track_name,
        geom_dir,
        lat_col=2,
        lon_col=3,
        lat_min=70
    )
