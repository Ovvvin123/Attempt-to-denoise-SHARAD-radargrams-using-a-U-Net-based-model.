import numpy as np
from PIL import Image

def top_correct(dat, fline, top, ep, dtt):
    """
    dat: 2D numpy array, 雷达矩阵
    fline: array_like, 每列起始点
    top: array_like, 每列顶部位置
    ep: float, 校正参数
    dtt: float, 平移像素数
    """
    vc = 300
    dt = 0.0375
    sdat = np.zeros_like(dat)
    tdat = np.zeros_like(dat)
    v1 = vc / np.sqrt(ep)
    dh0 = dt * v1 / 2
    ndat = dat.shape[1]
    maxline = np.max(fline)
    
    # 第一轮列平移
    for i in range(ndat):
        shift_val = int(round(maxline - fline[i] - dtt))
        sdat[:, i] = np.roll(dat[:, i], shift_val)
    
    # 顶部校正平移
    dtop = np.max(top) - top
    shftop = np.round(dtop / dh0).astype(int)
    
    for i in range(ndat):
        tdat[:, i] = np.roll(sdat[:, i], shftop[i])
    
    return tdat

def fig_id_track_holt2010(ID, par_path, rgram_path, browse_path):
    # 读取参数
    par = np.loadtxt(par_path)
    
    # 读取图像
    dat = np.array(Image.open(rgram_path))
    smdat = np.array(Image.open(browse_path))
    
    # top_correct
    tdat = top_correct(dat, par[:, 3], par[:, 2], 3.15, 400)
    smtdat = top_correct(smdat, par[:, 3], par[:, 2], 3.15, 400)
    
    # 计算距离（使用球面公式近似）
    R = 3376  # 火星半径 km
    lat = par[:, 0]
    lon = par[:, 1]
    dis1 = np.zeros(lon.shape[0])
    for i in range(1, len(dis1)):
        # 球面距离公式近似
        dlat = np.radians(lat[i] - lat[i-1])
        dlon = np.radians(lon[i] - lon[i-1])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat[i])) * np.cos(np.radians(lat[i-1])) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dis1[i] = dis1[i-1] + R * c
    
    # 纵轴坐标
    ytop = np.arange(dat.shape[0])
    ytop = ytop * 0.0375 * 300 / np.sqrt(3.15) / 2
    
    return tdat, smtdat, dis1, ytop

# 返回函数和主程序作为可调用对象
fig_id_track_holt2010, top_correct