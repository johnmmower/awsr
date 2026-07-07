import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.interpolate import RegularGridInterpolator
from get_legacy_radar_data import *

import cv2

FPS = 25000/16/64

def data_to_bgr_frame(arr, cmap=cv2.COLORMAP_VIRIDIS, vmin=None, vmax=None):
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    normalized = np.clip((arr - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(normalized, cmap)  # returns BGR directly, shape (H, W, 3)
    return bgr

def toMagAll(data):
    return np.abs(np.sum(data, axis=0))

def toLogMagAll(data):
    return 20*np.log10(toMagAll(data))

class ViewRadar(QtWidgets.QMainWindow):

    N_img_pts = 1000
    
    def __init__(self):
        super().__init__()
        self.resize(self.N_img_pts, self.N_img_pts)
        self.imv = pg.ImageView()
        cmap = pg.colormap.get('jet', source='matplotlib')
        self.imv.setColorMap(cmap)
        self.imv.ui.histogram.hide()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        self.setCentralWidget(self.imv)
                
        self.lr = GetLegacyRadarData('20260519-082349_data_dabob_0.bin',
                                     '20260519-082349_meta_dabob_0.bin',
                                     '20260519-082349_theta_dabob_0.bin')
        
        self.rngs = np.arange(1024) * 3e8 / 31.25e6 / 2
        self.ang_res = 1.
        self.angs = np.arange(0, 360 + self.ang_res, self.ang_res)
        self.plr = np.zeros((len(self.angs), len(self.rngs)))

        self.cntr = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.cc = cv2.VideoWriter('output.mp4', fourcc, FPS, (len(self.rngs), len(self.angs)))
            
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)                                                                                                                     
        self.timer.timeout.connect(self.update_heatmap)
        self.timer.start()

    def closeEvent(self, event: QCloseEvent):
        self.cc.release()
        
    def update_heatmap(self):

        [angle, tm, en, ant, data] = self.lr.get_chunk()        

        if ant is not None and ant != -1 and en:

            if True: #(ant == 0):
                datal = toLogMagAll(data)
                if ant == 0:
                    anglea = (angle + 180) % 360.
                elif ant == 1:
                    anglea = (angle + 90) % 360.
                elif ant == 2:
                    anglea = angle
                else:
                    anglea = (angle - 90) % 360.
                idx = np.abs(self.angs - anglea).argmin()

                datal[~np.isfinite(datal)] = 0.
                self.plr[idx,:] = datal
                
                ptr = np.array(self.plr, copy=True)
                ptr[idx,:] *= 0.
                ptr[ptr > 100] = 65
                ptr[ptr < 50] = 55
                '''
                ptr -= 55
                ptr /= 10
                ptr *= 255
                ptr = np.clip(ptr, 0, 255).astype(np.uint8)
                '''
                
                self.imv.setImage(ptr.T)

                self.cc.write(data_to_bgr_frame(ptr))
                
                self.cntr += 1

                
            
                
            '''
            height = len(self.rngs)
            width = len(self.rngs)
            center = (width // 2, height // 2)
            plr_img = cv2.warpPolar(
                self.plr,
                (width, height),
                center,
                height,
                cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            self.imv.setImage(plr_img.T)
            '''
            
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ViewRadar()
    window.show()
    sys.exit(app.exec())


