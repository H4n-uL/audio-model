import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import itertools
 
spt = []
ins = []
n = 0
insts = 128

for instrument, note in itertools.product(range(insts), range(50)):
    print(instrument, note)
    y, sr = librosa.load('output.wav', sr=None, offset=n, duration=2.0) # n초지점부터 2초까지만 데이터를 읽어옵니다.
    n += 2
    # 데이터를 늘리기 위해 white 노이즈를 섞은 버전도 함께 변환합니다
    # 시간 대역 데이터를 옥타브당 24단계로, 총 7옥타브로 변환할 겁니다.
    for r in (0, 1e-4, 1e-3):
        ret = librosa.cqt(y + ((np.random.rand(*y.shape) - 0.5) * r if r else 0), sr=sr, 
            hop_length=1024, n_bins=24*7, bins_per_octave=24)
        # 주파수의 위상은 관심없고, 세기만 보겠으니 절대값을 취해줍니다
        ret = np.abs(ret)
        spt.append(ret) # 스펙토그램을 저장합니다
        ins.append((instrument, 38 + note)) # 악기 번호와 음 높이를 저장합니다
 
for note in range(46):
    y, sr = librosa.load('output.wav', sr=None, offset=n, duration=2.0)
    n += 2
 
    for r, s in itertools.product([0, 1e-5, 1e-4, 1e-3], range(7)):
        ret = librosa.cqt(y + ((np.random.rand(*y.shape) - 0.5) * r * s if r else 0), sr=sr, 
            hop_length=1024, n_bins=24 * 7, bins_per_octave=24)
        ret = np.abs(ret)
        spt.append(ret)
        ins.append((note + insts, 0))
 
    # 아래의 코드는 변환된 주파수 대역의 스펙토그램을 보여줍니다.
    #librosa.display.specshow(librosa.amplitude_to_db(np.abs(ret), ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Constant-Q power spectrum')
    #plt.tight_layout()
    #plt.show()
 
spt = np.array(spt, np.float32)
ins = np.array(ins, np.int16)
np.savez('cqt.npz', spec=spt, instr=ins)