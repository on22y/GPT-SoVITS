import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tools.slicer2 import Slicer
from tools.my_utils import load_audio
from scipy.io import wavfile


def main():
    inp = 'data/resampled/iy_live_32k.wav'
    outdir = 'data/sliced'
    os.makedirs(outdir, exist_ok=True)
    audio = load_audio(inp, 32000)
    slicer = Slicer(sr=32000, threshold=-34, min_length=3000, min_interval=300, hop_size=20, max_sil_kept=500)
    cnt = 0
    for item in slicer.slice(audio):
        try:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                chunk, start, end = item
            else:
                chunk = item
                start, end = 0, chunk.shape[0]
            tmp_max = np.abs(chunk).max()
            if tmp_max > 1:
                chunk = chunk / tmp_max
            wavfile.write(os.path.join(outdir, f"{os.path.basename(inp)}_{start}_{end}.wav"), 32000, (chunk * 32767).astype(np.int16))
            cnt += 1
        except Exception as e:
            print('slice error', e)
    print('sliced_count', cnt)


if __name__ == '__main__':
    main()
