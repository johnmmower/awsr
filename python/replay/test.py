import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.interpolate import RegularGridInterpolator
from get_legacy_radar_data import *

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
        '''
        self.T, self.R = np.meshgrid(self.angs*np.pi/180, self.rngs)

        self.x = np.linspace(-self.rngs[-1], self.rngs[1], self.N_img_pts)
        self.y = np.linspace(-self.rngs[-1], self.rngs[1], self.N_img_pts)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        '''
        self.plr = np.zeros((len(self.angs), len(self.rngs)))
        
        self.cart = np.zeros((self.N_img_pts, self.N_img_pts))

        self.imv.setImage(self.cart)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)  # ~33 FPS                                                                                                                   
        self.timer.timeout.connect(self.update_heatmap)
        self.timer.start()

    def update_heatmap(self):

        [angle, tm, en, ant, data] = self.lr.get_chunk()        
        
        if ant is not None and ant != -1 and en:
            
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
            self.imv.setImage(ptr)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ViewRadar()
    window.show()
    sys.exit(app.exec())


