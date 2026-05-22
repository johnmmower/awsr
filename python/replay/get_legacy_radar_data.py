
from struct import unpack
import numpy as np

def raw2cplx(vals):                                                                                                                  
    return np.complex64(np.array(vals[0:1024]) + 1j * np.array(vals[1024:2048]))

class GetLegacyRadarData(object):

    N_chunk = 64
    
    def __init__(self, data_fname, meta_fname, theta_fname):
        self.data_fid = open(data_fname, 'rb')
        self.meta_fid = open(meta_fname, 'rb')
        self.theta_fid = open(theta_fname, 'rb')

        self.theta_fid.seek(0, 2)
        self.num_smps = self.theta_fid.tell() // 4
        self.theta_fid.seek(0, 0)

    def get_chunk(self):
        angles = np.zeros(self.N_chunk)
        tms = np.zeros(self.N_chunk)
        ens = np.zeros(self.N_chunk)
        ants = np.zeros(self.N_chunk)
        datas = np.zeros((self.N_chunk, 1024), dtype=np.complex64)
        for i in range(self.N_chunk):
            angle, tm, en, ant, data = self.get_frame()
            if angle is None:
                break
            if i > 0 and ants[i] != ants[0]:
                ant = -1
                self.data_fid.seek(-2*1024*4, 2)
                self.meta_fid.seek(-32, 2)
                self.theta_fid.seek(-4, 2)
                break
            angles[i] = angle
            tms[i] = tm
            datas[i] = raw2cplx(data)
        if angle is None:
            return [None, None, None, None, None]
        else:
            return [angles[self.N_chunk//2], np.mean(tms), en, ant, datas]

    def get_frame(self):
        try:
            data = unpack('<2048i', self.data_fid.read(2*1024*4))
            [tic, epoch] = unpack('<II', self.meta_fid.read(8))
            tm = epoch + tic / 125e6
            self.meta_fid.read(2)
            enant = unpack('b', self.meta_fid.read(1))[0]
            en = enant >> 2
            ant = enant & 3
            self.meta_fid.read(32-11)
            angle = (unpack('<i', self.theta_fid.read(4))[0] * 180.0 / 2**23 + 180.0) % 360.
            return [angle, tm, en, ant, data]
        except:
            return [None, None, None, None, None]
