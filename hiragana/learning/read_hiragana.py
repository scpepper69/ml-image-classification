import struct
import numpy as np
from PIL import Image

sz_record = 8199

shift_jis = []
jisx0208 = []
unicode = []
with open("./jis/JIS0208.TXT", "r") as f:
    for line in f:
        if line[0] == "#":
            pass
        else:
            sjis, jisx, unic, _ = line.strip().split("\t")
            shift_jis.append(int(sjis,16))
            jisx0208.append( int(jisx,16))
            unicode.append(  int(unic,16))

def jis2uni(n):
    return unicode[jisx0208.index(n)]

def read_record_ETL8G(f):
    s = f.read(sz_record)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)


def read_hiragana():
    # Character type = 72, person = 160, y = 127, x = 128
    ary = np.zeros([71, 160, 127, 128], dtype=np.uint8)  #hiragana
#    ary = np.zeros([956, 160, 127, 128], dtype=np.uint8) #all

    for j in range(1, 33):
        filename = './datasets/ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                moji = 0
                for i in range(956):
                    r = read_record_ETL8G(f)
                    if b'.HIR' in r[2] and r[1] < 10000 :
                        ary[moji, (j - 1) * 5 + id_dataset] = np.array(r[-1])
                        moji += 1
#                        print(moji, r[0:-2], hex(r[1]))
                        print(moji-1, r[0:-2], hex(r[1]),r[1],chr(jis2uni(r[1])))
    np.savez_compressed("hiragana.npz", ary)

read_hiragana()

