"""Convert PTB-XL records100 to 12-channel WAV without changing a single sample.

    python -m ecgpit.convert

PTB-XL stores 16-bit integers at 1000 per mV. The same integers are written to a 16-bit PCM WAV, so 1 WAV unit = 1 microvolt
for every recording and the true voltage is kept.
"""
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import load_config, resolve


def convert(dat, out):
    hea = dat.with_suffix(".hea").read_text().split("\n")
    assert hea[0].split()[1:4] == ["12", "100", "1000"], dat
    assert all(h.split()[1] == "16" and h.split()[2].startswith("1000.0(0)/mV") for h in hea[1:13]), dat
    x = np.fromfile(dat, "<i2"); assert x.size == 12000, dat
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(12); w.setsampwidth(2); w.setframerate(100); w.writeframes(x.tobytes())
    with wave.open(str(out), "rb") as w:
        assert w.readframes(1000) == x.tobytes(), out


def main():
    root = resolve(load_config()["data"]["root"])
    files = sorted((root / "records100").glob("*/*.dat"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda d: convert(d, root / "wav" / d.parent.name / (d.stem + ".wav")), files))
    print(f"{len(files)} recordings converted and verified sample by sample")


if __name__ == "__main__":
    main()
