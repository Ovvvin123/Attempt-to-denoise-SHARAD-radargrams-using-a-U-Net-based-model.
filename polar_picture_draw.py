from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import rasterio


# =========================
# 基本参数：按你下载 MOLA DEM 时的设置
# =========================

dem_path = r"MOLA_polar_60-90.tif"
geom_dir = Path(r".\geom_data_total")

MIN_LAT = 60
MAX_LAT = 90

LON0 = 0              # Center Longitude = 0
MARS_RADIUS = 3396190 # m

# 你说 geom.tab 中：
# 第 3 个数字 = 经度
# 第 4 个数字 = 纬度
# Python 从 0 开始，所以是 2 和 3
LON_COL = 3
LAT_COL = 2


def find_geom_file(track_name, geom_dir):
    """
    根据输入的轨迹名寻找对应的 geom.tab 文件。

    支持输入：
        00172101
        s_00172101
        s_00172101_geom
        s_00172101_geom.tab
        S_00172101_RGRAM.IMG
    """

    geom_dir = Path(geom_dir)
    track_name = str(track_name).strip()

    m = re.search(r"(\d{8})", track_name)

    if m is None:
        raise ValueError(
            f"输入的名字里没有找到 8 位轨迹编号。\n"
            f"你当前输入的是：{repr(track_name)}\n"
            f"请直接输入类似：00172101"
        )

    track_id = m.group(1)

    # 例如 00172101 -> s_0017xx
    subfolder_name = f"s_{track_id[:4]}xx"
    subfolder = geom_dir / subfolder_name

    if not subfolder.exists():
        raise FileNotFoundError(
            f"没有找到对应子文件夹：{subfolder}\n"
            f"请检查 geom_dir 是否设对。"
        )

    candidates = list(subfolder.glob(f"*{track_id}*_geom.tab"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"在子文件夹 {subfolder} 中没有找到编号 {track_id} 对应的 geom.tab 文件"
        )

    return candidates[0]


def read_geom_tab(tab_file):
    """
    读取 geom.tab 文件中的经纬度。

    文件里包含时间字符串，所以不能整表读取。
    这里只读取第 3、4 列：
        第 3 列 = 经度
        第 4 列 = 纬度
    """

    lon_lat = np.loadtxt(
        tab_file,
        delimiter=",",
        usecols=(LON_COL, LAT_COL)
    )

    lon = lon_lat[:, 0]
    lat = lon_lat[:, 1]

    return lon, lat


def mars_north_polar_stereo(lon, lat, lon0=LON0, radius=MARS_RADIUS):
    """
    火星北极立体投影，球体近似。

    对应你下载 DEM 时的设置：
        Projection: Polar Stereographic
        Center Latitude: 90
        Center Longitude: 0
        Longitude Domain: 0 to 360
        Longitude Direction: Positive East

    lat = 90 时，x = 0, y = 0。
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_rad = np.deg2rad(lon0)

    dlon = lon_rad - lon0_rad

    rho = 2 * radius * np.tan(np.pi / 4 - lat_rad / 2)

    x = rho * np.sin(dlon)
    y = -rho * np.cos(dlon)

    return x, y


def draw_graticule(ax, x0, y0):
    """
    在北极投影图上画经纬网。

    纬线：60N, 70N, 80N
    经线：每 30° 一条
    经度标注：每 60° 标一次
    """

    grid_color = "0.35"

    # 纬线圈
    lon_circle = np.linspace(0, 360, 721)

    for lat_ring in [60, 70, 80]:
        lat_arr = np.full_like(lon_circle, lat_ring, dtype=float)

        gx, gy = mars_north_polar_stereo(
            lon_circle,
            lat_arr,
            lon0=LON0,
            radius=MARS_RADIUS
        )

        gx = gx + x0
        gy = gy + y0

        ax.plot(
            gx,
            gy,
            color=grid_color,
            linewidth=0.6,
            alpha=0.65,
            zorder=3
        )

        # 纬度标签放在 45E 附近，避免挡住中央经线
        lx, ly = mars_north_polar_stereo(
            np.array([45]),
            np.array([lat_ring]),
            lon0=LON0,
            radius=MARS_RADIUS
        )

        ax.text(
            lx[0] + x0,
            ly[0] + y0,
            f"{lat_ring}°N",
            fontsize=9,
            color=grid_color,
            ha="center",
            va="center",
            zorder=4
        )

    # 经线
    lat_line = np.linspace(MIN_LAT, MAX_LAT, 300)

    for lon_line in range(0, 360, 30):
        lon_arr = np.full_like(lat_line, lon_line, dtype=float)

        gx, gy = mars_north_polar_stereo(
            lon_arr,
            lat_line,
            lon0=LON0,
            radius=MARS_RADIUS
        )

        gx = gx + x0
        gy = gy + y0

        ax.plot(
            gx,
            gy,
            color=grid_color,
            linewidth=0.5,
            alpha=0.55,
            zorder=3
        )

    # 经度标签，每 60° 标一次，放在 60N 外圈附近
    for lon_label in range(0, 360, 60):
        lx, ly = mars_north_polar_stereo(
            np.array([lon_label]),
            np.array([MIN_LAT]),
            lon0=LON0,
            radius=MARS_RADIUS
        )

        ax.text(
            lx[0] + x0,
            ly[0] + y0,
            f"{lon_label}°E",
            fontsize=9,
            color=grid_color,
            ha="center",
            va="center",
            zorder=4
        )


def main(track_name=None, show=True, save_path=None):
    """
    画单条 SHARAD 轨迹在 MOLA 北极 DEM 上的位置。

    当前版本：
        1. 只画北纬 60° 以上的轨迹；
        2. 雷达轨迹只用黑线；
        3. 标出轨迹进入图中区域后的起始点；
        4. 画经纬网。
    """

    if track_name is None:
        track_name = input("请输入轨迹名，例如 00172101 或 s_00172101_geom：").strip()

    track_name = str(track_name).strip()

    # =========================
    # 1. 找 geom 文件
    # =========================

    tab_file = find_geom_file(track_name, geom_dir)
    print(f"使用 geom 文件：{tab_file}")

    # =========================
    # 2. 读取经纬度
    # =========================

    lon, lat = read_geom_tab(tab_file)

    # 你的 DEM 下载时 Longitude Domain 是 0 to 360
    lon = lon % 360

    print("经度范围:", np.nanmin(lon), np.nanmax(lon))
    print("纬度范围:", np.nanmin(lat), np.nanmax(lat))

    if np.nanmax(lat) < MIN_LAT:
        print(f"这条轨迹最高纬度只有 {np.nanmax(lat):.2f}°N，没有进入北纬 {MIN_LAT}° 以上区域。")
        return None, None

    # =========================
    # 3. 读取 DEM
    # =========================

    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype("float32").filled(np.nan)
        bounds = src.bounds
        dem_crs = src.crs

    extent = [
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ]

    print("DEM CRS:")
    print(dem_crs)
    print("DEM extent:", extent)

    # DEM 图像中心
    x0 = (extent[0] + extent[1]) / 2
    y0 = (extent[2] + extent[3]) / 2

    print("DEM center x:", x0)
    print("DEM center y:", y0)

    # =========================
    # 4. 经纬度转北极投影坐标
    # =========================

    x, y = mars_north_polar_stereo(
        lon,
        lat,
        lon0=LON0,
        radius=MARS_RADIUS
    )

    # 让 90N 对齐到 DEM 图像中心
    x = x + x0
    y = y + y0

    # 只保留北纬 60° 以上点
    valid = (lat >= MIN_LAT) & np.isfinite(x) & np.isfinite(y)

    if not np.any(valid):
        print(f"这条轨迹没有北纬 {MIN_LAT}° 以上的点。")
        return None, None

    # 用 NaN 断开 60N 以下的部分
    # 这样不用 split_valid_segments，也不会把不连续段错误连起来
    x_plot = np.where(valid, x, np.nan)
    y_plot = np.where(valid, y, np.nan)

    valid_idx = np.where(valid)[0]
    start_idx = valid_idx[0]

    print(f"北纬 {MIN_LAT}° 以上点数:", len(valid_idx))
    print("进入图中区域的起始点：")
    print("  lon =", lon[start_idx])
    print("  lat =", lat[start_idx])

    # =========================
    # 5. 绘图
    # =========================

    fig, ax = plt.subplots(figsize=(10, 10))

    vmin, vmax = np.nanpercentile(dem, [2, 98])

    img = ax.imshow(
        dem,
        extent=extent,
        origin="upper",
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
        zorder=1
    )

    cbar = plt.colorbar(img, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Elevation / m")

    # 经纬网
    draw_graticule(ax, x0, y0)

    # 北极点
    ax.scatter(
        x0,
        y0,
        s=70,
        c="cyan",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
        label="North Pole"
    )

    # 雷达轨迹：只用黑线
    track_id = re.search(r"(\d{8})", track_name).group(1)

    ax.plot(
        x_plot,
        y_plot,
        color="black",
        linewidth=1.3,
        alpha=0.95,
        zorder=7,
        label=f"S_{track_id}"
    )

    # 起始点：轨迹进入北纬 60° 以上区域后的第一个点
    ax.scatter(
        x[start_idx],
        y[start_idx],
        s=80,
        marker="*",
        c="red",
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
        label="Track start above 60°N"
    )

    track_id = re.search(r"(\d{8})", track_name).group(1)

    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")

    ax.set_title(
        f"SHARAD Track S_{track_id} on MOLA North Polar DEM\n"
        f"Polar Stereographic, 60°N–90°N, 0°–360°E"
    )

    ax.legend(loc="upper right")

    plt.tight_layout()

    

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    main()