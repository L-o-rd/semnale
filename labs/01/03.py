import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    '''
        a) Intervalul de timp intre doua esantioane este ts, unde
        ts = 1 / fs, unde fs este frecventa de esantionare.
        Deci ts = 1 / 2000 = 0.0005 s = 0.5 ms.

        b) Esantion - 4 bits
            2 Esantioane = 1 byte
            1 ora = 60 minute = 60 * 60 secunde
            Fie sample rate-ul / frecventa de esantionare V:
                V * 60 * 60 esantioane in total
                V * 60 * 60 * 4 bits
                V * 60 * 60 * 4 / 8 bytes
                V * 60 * 60 / 2 bytes
                V * 60 * 30 bytes
                V * 1800 bytes
            Deci 2000 * 1800 = 3'600'000 bytes
                             = 3.43 mebibytes
                             = 3.60 megabytes
    '''
    pass
    
