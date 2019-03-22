# coding: utf-8

shift_jis = []
jisx0208 = []
unicode = []
with open("JIS0208.TXT", "r") as f:
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
    
if __name__ == '__main__':
#    print("{:x}".format(jis2uni(0x2422)))
    print(chr(jis2uni(0x2422)))
