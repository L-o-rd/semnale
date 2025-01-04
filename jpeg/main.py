from scipy.fft import dctn, idctn
import matplotlib.pyplot as plt
from bitstring import BitStream
from scipy.ndimage import zoom
from typing import Final
from PIL import Image
import numpy as np
import scipy as sp
import argparse
import huffman
import cv2

# Step I:
#   ~ Convert RGB to YCbCr.
#   ~ Using ITU-T T.871 values.

YCbCrMatrix: Final = np.array([[0.299, 0.587, 0.114],
                               [-0.168736, -0.331264, 0.5],
                               [0.5, -0.418688, -0.081312]])

RGBMatrix: Final = np.array([
    [1.0,  0.0,       1.402],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.772,     0.0]
])

def rgb_to_ycbcr(image):
    shift = np.array([0, 128, 128])
    ycbcr = np.dot(image, YCbCrMatrix.T) + shift
    return ycbcr.clip(0, 255).astype(np.int32)

def ycbcr_to_rgb(image):
    rgb = np.dot(image, RGBMatrix.T)
    return rgb.clip(0, 255).astype(np.int32)

# Step II:
#   ~ Subsample chrominance channels.
#   ~ 4 : 2 : 2 (YCbCr scheme).

def chroma_subsample(ycbcr):
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    Cb = zoom(Cb, (0.5, 0.5), order = 0)
    Cr = zoom(Cr, (0.5, 0.5), order = 0)
    return Y, Cb, Cr

# Step III:
#   ~ Divide into 8x8 blocks.

def mcu_divide_into_blocks(channel, block_size = 8):
    assert len(channel.shape) == 2, 'Channel should be (width, height).'

    # Align to block size

    height, width = channel.shape
    aligned = (
        ((height + block_size - 1) // block_size) * block_size,
        ((width + block_size - 1) // block_size) * block_size
    )

    # Pad everything with 0s
    
    padded = np.pad(
        channel,
        ((0, aligned[0] - height), (0, aligned[1] - width)),
        mode = 'constant',
        constant_values = 0
    )
    
    blocks = []
    for y in range(0, aligned[0], block_size):
        for x in range(0, aligned[1], block_size):
            blocks.append(padded[y : y + block_size, x : x + block_size])

    return np.array(blocks)

# Step IV
#   ~ Apply DCT over inputs.

def apply_dct(blocks):
    applied = [
        dctn(block, norm = 'ortho') for block in blocks
        #dct(dct(block.T, norm = 'ortho').T, norm = 'ortho') for block in blocks
    ]

    return np.array(applied)

# Step V
#   ~ Quantization.

QLum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 28, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]], dtype = np.int32)

QChr = np.array([ 17,  18,  24,  47,  99,  99,  99,  99,
                    18,  21,  26,  66,  99,  99,  99,  99,
                    24,  26,  56,  99,  99,  99,  99,  99,
                    47,  66,  99,  99,  99,  99,  99,  99,
                    99,  99,  99,  99,  99,  99,  99,  99,
                    99,  99,  99,  99,  99,  99,  99,  99,
                    99,  99,  99,  99,  99,  99,  99,  99,
                    99,  99,  99,  99,  99,  99,  99,  99], dtype = np.int32).reshape((8, 8))

def quantize(blocks, matrix):
    return np.round(blocks / matrix).astype(np.int32)

def qquality(Q, quality):
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    S = 5000 / quality if quality < 50 else 200 - 2 * quality
    Q = np.floor((S * Q + 50) / 100).astype(np.int32)
    Q[Q == 0] = 1
    Q[Q > 255] = 255
    return Q

# Step VI
#   ~ Vectorize zig-zag order.

ZigZagOrder = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])

def zigzag(block):
    return block.reshape((64,))[ZigZagOrder].astype(np.int32)

# Step VII
#   ~ Run-length encoding.
#   ~ Compute no. of zeros between non-zero coefficients.

def rle_encode(blocks):
    def one_block(block):
        result = []
        zs = 0
        
        for value in block:
            if value == 0:
                zs += 1
            else:
                result.append((zs, value))
                zs = 0

        return result
    
    return np.array(
        [one_block(block) for block in blocks]
    )

def jpeg_encode(image, output, quality = 50, block_size = 8, stop_early = False, scaling = False, scale = 1.0):
    assert len(image.shape) == 3, 'error: Image has only one channel.'
    assert image.shape[-1] == 3, f'error: Image has {image.shape[-1]} channels.'
    sheight, swidth, _ = image.shape

    if not scaling:
        qlum = QLum.copy()
        qlum = qquality(qlum, quality)

        qchr = QChr.copy()
        qchr = qquality(qchr, quality)
    else:
        qlum = QLum.copy() * scale
        qchr = QChr.copy() * scale

    # Step I
    image = rgb_to_ycbcr(image)

    # Step II
    #   ~ No subsampling.
    # Y, Cb, Cr = chroma_subsample(image)
    Y, Cb, Cr = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    #   ~ Convert to Y'UV from YCbCr.
    Y = Y - 127
    Cb = Cb - 127
    Cr = Cr - 127

    # Step III
    Y = mcu_divide_into_blocks(Y, block_size)
    Cb = mcu_divide_into_blocks(Cb, block_size)
    Cr = mcu_divide_into_blocks(Cr, block_size)

    # Step IV
    Y = apply_dct(Y)
    Cb = apply_dct(Cb)
    Cr = apply_dct(Cr)

    # Step V
    Y = quantize(Y, qlum)
    Cb = quantize(Cb, qchr)
    Cr = quantize(Cr, qchr)

    if stop_early:
        return Y, Cb, Cr
    
    # Step VI
    Y = np.array([
        zigzag(block) for block in Y
    ])

    Cb = np.array([
        zigzag(block) for block in Cb
    ])

    Cr = np.array([
        zigzag(block) for block in Cr
    ])

    # Step VII
    #   ~ Not used.

    # Y = rle_encode(Y)
    # Cb = rle_encode(Cb)
    # Cr = rle_encode(Cr)

    # Step VIII
    #   ~ Huffman Coding.

    sblocks = (swidth // block_size) * (sheight // block_size)
    yDC = np.zeros((sblocks,), dtype = np.int32)
    uDC = np.zeros((sblocks,), dtype = np.int32)
    vDC = np.zeros((sblocks,), dtype = np.int32)
    dyDC = np.zeros((sblocks,), dtype = np.int32)
    duDC = np.zeros((sblocks,), dtype = np.int32)
    dvDC = np.zeros((sblocks,), dtype = np.int32)
    bs = BitStream()

    for nblock in range(sblocks):
        yDC[nblock] = Y[nblock][0]
        uDC[nblock] = Cb[nblock][0]
        vDC[nblock] = Cr[nblock][0]

        if nblock == 0:
            dyDC[nblock] = yDC[nblock]
            duDC[nblock] = uDC[nblock]
            dvDC[nblock] = vDC[nblock]
        else:
            dyDC[nblock] = yDC[nblock] - yDC[nblock - 1]
            duDC[nblock] = uDC[nblock] - uDC[nblock - 1]
            dvDC[nblock] = vDC[nblock] - vDC[nblock - 1]

        bs.append(huffman.coded(huffman.encodeDCToBoolList(dyDC[nblock], 1)))
        huffman.encodeACBlock(bs, Y[nblock][1:], 1)

        bs.append(huffman.coded(huffman.encodeDCToBoolList(duDC[nblock], 0)))
        huffman.encodeACBlock(bs, Cb[nblock][1:], 0)

        bs.append(huffman.coded(huffman.encodeDCToBoolList(dvDC[nblock], 0)))
        huffman.encodeACBlock(bs, Cr[nblock][1:], 0)

    with open(output, 'wb+') as f:
        # Write Start of Image (SOI) marker
        f.write(b'\xFF\xD8')
        
        # Write JFIF-APP0 marker
        f.write(b'\xFF\xE0')
        f.write(b'\x00\x10\x4A\x46\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00')

        # Write Quantization Tables (DQT)
        # Write Luminance Table
        f.write(b'\xFF\xDB')
        f.write(b'\x00\x43\x00')
        table = qlum.reshape((64,))
        f.write(bytes(table.tolist()))
        
        # Write Chrominance Table
        f.write(b'\xFF\xDB')
        f.write(b'\x00\x43\x01')
        table = qchr.reshape((64,))
        f.write(bytes(table.tolist()))

        # Write Start of Frame (SOF) marker
        f.write(b'\xFF\xC0')
        f.write(b'\x00\x11\x08')
        f.write(sheight.to_bytes(2, byteorder = 'big'))
        f.write(swidth.to_bytes(2, byteorder = 'big'))
        f.write(b'\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01')
        
        # Write Huffman Table (DHT)
        f.write(b'\xFF\xC4')
        f.write(huffman.HexTable)
        
        # Write Start of Scan (SOS)
        f.write(b'\xFF\xDA')
        f.write(b'\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00')
        padding = huffman.coded(np.ones((8 - (bs.len & 7),), dtype = np.int32).tolist())
        bs.append(padding)
        
        # Write Encoded Data
        bbytes = bs.bytes
        for i in range(len(bbytes)):
            f.write(bytes([bbytes[i]]))
            if bbytes[i] == 0xff:
                f.write(bytes([0]))
        
        # Write End of Image (EOI) marker
        f.write(b'\xFF\xD9')

def comp_to_rgb(shape, quality, Y, Cb, Cr, scaling = False, scale = 1.0):
    sheight, swidth, _ = shape
    yblocks, xblocks = sheight // 8, swidth // 8

    if not scaling:
        qlum = qquality(QLum, quality)
        qchr = qquality(QChr, quality)
    else:
        qlum = QLum.copy() * scale
        qchr = QChr.copy() * scale

    YY = Y.copy()
    CCb = Cb.copy()
    CCr = Cr.copy()

    YY = np.array([
        idctn(qlum * block, norm = 'ortho') for block in YY
    ])

    CCb = np.array([
        idctn(qchr * block, norm = 'ortho') for block in CCb
    ])

    CCr = np.array([
        idctn(qchr * block, norm = 'ortho') for block in CCr
    ])

    YY = YY + 127
    CCb = CCb + 127
    CCr = CCr + 127

    YY = YY.reshape(yblocks, xblocks, 8, 8)
    YY = YY.transpose(0, 2, 1, 3).reshape(sheight, swidth)
    
    CCb = CCb.reshape(yblocks, xblocks, 8, 8)
    CCb = CCb.transpose(0, 2, 1, 3).reshape(sheight, swidth)

    CCr = CCr.reshape(yblocks, xblocks, 8, 8)
    CCr = CCr.transpose(0, 2, 1, 3).reshape(sheight, swidth)
    CCb = CCb - 128
    CCr = CCr - 128

    inv_ycbcr_image = np.stack((YY, CCb, CCr), axis = -1, dtype = np.float32)
    image = ycbcr_to_rgb(inv_ycbcr_image).astype(np.int32)
    return image

def jpeg_to_mse(image, threshold, scaling = False, factor = 1.0):
    def mse(original, compressed):
        original = np.array(original, dtype = np.float32)
        compressed = np.array(compressed, dtype = np.float32)
        m = np.mean((original - compressed) ** 2)
        return m
    
    step = 5
    scale = 1
    quality = 100
    attained = False
    compressed_image = None
    compressed_mse = float('-inf')

    if not scaling:
        while quality > 0:
            compressed_image = image.copy()
            Y, Cb, Cr = jpeg_encode(compressed_image, None, quality = quality, stop_early = True)
            compressed_image = comp_to_rgb(image.shape, quality, Y, Cb, Cr)

            compressed_mse = mse(image, compressed_image)
            print(f"Quality: {quality}, MSE: {compressed_mse}")

            if compressed_mse >= threshold:
                attained = True
                break

            quality -= step
    else:
        while compressed_mse < threshold:
            compressed_image = image.copy()
            Y, Cb, Cr = jpeg_encode(compressed_image, None, quality = quality, stop_early = True, scaling = scaling, scale = scale)
            compressed_image = comp_to_rgb(image.shape, quality, Y, Cb, Cr, scaling = scaling, scale = scale)

            compressed_mse = mse(image, compressed_image)
            print(f"Scale: {scale}, MSE: {compressed_mse}")

            if compressed_mse >= threshold:
                attained = True
                break

            scale += factor

    if attained == False:
        print('MSE Threshold was not attained.')
    
    plt.figure(figsize = (8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    if not scaling:
        plt.title(f'$mse = {compressed_mse}$, $quality = {quality}$')
    else:
        plt.title(f'$mse = {compressed_mse}$, $scale = {scale}$')
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.tight_layout()
    if not scaling:
        plt.savefig(f'./result/03/person-{quality}.pdf')
    else:
        plt.savefig(f'./result/03/person-s{scale:.2f}.pdf')
    plt.show()

def ex2(quality = 100):
    image = sp.misc.face()
    ycbcr_image = rgb_to_ycbcr(image)

    Y = ycbcr_image[:, :, 0]
    Cb = ycbcr_image[:, :, 1]
    Cr = ycbcr_image[:, :, 2]

    plt.figure(figsize = (8, 6))

    # Original image

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original RGB Image")
    plt.axis("off")

    # Y' channel

    plt.subplot(2, 2, 2)
    plt.imshow(Y, cmap='gray')
    plt.title("Y' Channel (Luminance)")
    plt.axis("off")

    # Cb channel

    plt.subplot(2, 2, 3)
    plt.imshow(Cb, cmap='gray')
    plt.title("Cb Channel (Chrominance)")
    plt.axis("off")

    # Cr channel

    plt.subplot(2, 2, 4)
    plt.imshow(Cr, cmap='gray')
    plt.title("Cr Channel (Chrominance)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('./result/02/misc-a.pdf')
    plt.show()

    plt.figure(figsize = (8, 6))

    # Encode it

    Y = mcu_divide_into_blocks(Y, 8)
    Cb = mcu_divide_into_blocks(Cb, 8)
    Cr = mcu_divide_into_blocks(Cr, 8)

    # Step IV
    Y = np.array([
        dctn(block) for block in Y
    ])

    Cb = np.array([
        dctn(block) for block in Cb
    ])

    Cr = np.array([
        dctn(block) for block in Cr
    ])

    qlum = qquality(QLum, quality)
    qchr = qquality(QChr, quality)

    # Step V
    Y = quantize(Y, qlum)
    Cb = quantize(Cb, qchr)
    Cr = quantize(Cr, qchr)

    sheight, swidth, _ = image.shape
    yblocks, xblocks = sheight // 8, swidth // 8
    
    YY = Y.copy()
    YY = YY.reshape(yblocks, xblocks, 8, 8)
    YY = YY.transpose(0, 2, 1, 3).reshape(sheight, swidth)
    
    CCb = Cb.copy()
    CCb = CCb.reshape(yblocks, xblocks, 8, 8)
    CCb = CCb.transpose(0, 2, 1, 3).reshape(sheight, swidth)

    CCr = Cr.copy()
    CCr = CCr.reshape(yblocks, xblocks, 8, 8)
    CCr = CCr.transpose(0, 2, 1, 3).reshape(sheight, swidth)

    # Original image

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original RGB Image")
    plt.axis("off")

    # Y' channel

    plt.subplot(2, 2, 2)
    plt.imshow(YY, cmap='gray')
    plt.title("Y' Blocks (Luminance)")
    plt.axis("off")

    # Cb channel

    plt.subplot(2, 2, 3)
    plt.imshow(CCb, cmap='gray')
    plt.title("Cb Blocks (Chrominance)")
    plt.axis("off")

    # Cr channel

    plt.subplot(2, 2, 4)
    plt.imshow(CCr, cmap='gray')
    plt.title("Cr Blocks (Chrominance)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('./result/02/misc-b.pdf')
    plt.show()

    Y = np.array([
        idctn(qlum * block) for block in Y
    ])

    Cb = np.array([
        idctn(qchr * block) for block in Cb
    ])

    Cr = np.array([
        idctn(qchr * block) for block in Cr
    ])

    YY = Y.copy()
    YY = YY.reshape(yblocks, xblocks, 8, 8)
    YY = YY.transpose(0, 2, 1, 3).reshape(sheight, swidth)
    
    CCb = Cb.copy()
    CCb = CCb.reshape(yblocks, xblocks, 8, 8)
    CCb = CCb.transpose(0, 2, 1, 3).reshape(sheight, swidth)

    CCr = Cr.copy()
    CCr = CCr.reshape(yblocks, xblocks, 8, 8)
    CCr = CCr.transpose(0, 2, 1, 3).reshape(sheight, swidth)

    CCb = CCb - 128
    CCr = CCr - 128
    inv_ycbcr_image = np.stack((YY, CCb, CCr), axis = -1, dtype = np.float32)
    inv_image = ycbcr_to_rgb(inv_ycbcr_image).astype(np.int32)

    plt.figure(figsize = (8, 6))

    # Original image

    plt.subplot(2, 2, 1)
    plt.imshow(inv_image)
    plt.title("Inverse RGB Image")
    plt.axis("off")

    # Y' channel

    plt.subplot(2, 2, 2)
    plt.imshow(YY, cmap='gray')
    plt.title("Y' Blocks (Luminance)")
    plt.axis("off")

    # Cb channel

    plt.subplot(2, 2, 3)
    plt.imshow(CCb, cmap='gray')
    plt.title("Cb Blocks (Chrominance)")
    plt.axis("off")

    # Cr channel

    plt.subplot(2, 2, 4)
    plt.imshow(CCr, cmap='gray')
    plt.title("Cr Blocks (Chrominance)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('./result/02/misc-c.pdf')
    plt.show()

    jpeg_encode(image, f'./result/02/misc-{quality}.jpg', quality)

def ex4(video, output, quality = 100):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("error: could not open video.")
        exit()

    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = ''.join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Y, Cb, Cr = jpeg_encode(frame, None, quality, stop_early = True)
        nimage = comp_to_rgb(frame.shape, quality, Y, Cb, Cr).astype(np.float32)
        plt.title(f'Frame {frame_count}')
        plt.imshow(nimage.astype(np.uint8))
        plt.savefig(f'./result/04/frames/frame-{frame_count}.pdf')
        frame = cv2.cvtColor(nimage, cv2.COLOR_RGB2BGR)
        out.write(frame.astype(np.uint8))
        frame_count += 1

    cap.release()
    out.release()

def ex1(quality):
    X = sp.misc.ascent()
    qlum = qquality(QLum, quality)
    for i in range(0, X.shape[0], 8):
        for j in range(0, X.shape[1], 8):
            x = X[i:i + 8, j:j + 8]
            y = dctn(x)
            y_jpeg = qlum * np.round(y / qlum)
            plt.subplot(8, 8, (j // 8) + 1).imshow(x, cmap = 'gray')
            plt.subplot(8, 8, (j // 8) + 1).axis('off')
        
        plt.savefig(f'./result/01/oblocks/oblock-{i // 8}.pdf')

    for i in range(0, X.shape[0], 8):
        for j in range(0, X.shape[1], 8):
            x = X[i:i + 8, j:j + 8]
            y = dctn(x)
            y_jpeg = qlum * np.round(y / qlum)
            x_jpeg = idctn(y_jpeg)
            plt.subplot(8, 8, (j // 8) + 1).imshow(x_jpeg, cmap = 'gray')
            plt.subplot(8, 8, (j // 8) + 1).axis('off')
        
        plt.savefig(f'./result/01/jblocks/jblock-{i // 8}.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'ps-jpeg', description = 'JPEG compression.')
    parser.add_argument('source')
    parser.add_argument('-o', '--out', default = './output.jpg')
    parser.add_argument('-q', '--quality', default = 100)
    parser.add_argument('-b', '--block', default = 8) # (not used)
    parser.add_argument('-e', '--exercises', action = 'store_true')
    parser.add_argument('-m', '--mse', default = 1.0)
    parser.add_argument('-opt', '--optional')
    args = parser.parse_args()

    if args.exercises == False:
        # Open image and print useful information.
        ini = Image.open(args.source)
        print(f'Input: {args.source}')
        print(f'Output: {args.out}')
        print(f'Width: {ini.size[0]}')
        print(f'Height: {ini.size[1]}')

        # Compress given image to JPEG.
        image = np.array(ini.convert('RGB'), dtype = np.uint8)
        jpeg_encode(image, quality = int(args.quality), output = args.out)
    else:
        match int(args.source):
            case 1:
                ex1(int(args.quality))
            case 2:
                ex2(int(args.quality))
            case 3:
                ini = Image.open(args.optional)
                print(f'Input: {args.optional}')
                print(f'Width: {ini.size[0]}')
                print(f'Height: {ini.size[1]}')

                # Compress given image to JPEG.
                image = np.array(ini.convert('RGB'), dtype = np.uint8)
                jpeg_to_mse(image, float(args.mse), True, 10.0)
            case 4:
                print(f'Input: {args.optional}')

                # Compress given video (frames) to JPEG.
                ex4(args.optional, args.out, int(args.quality))
            case _:
                print(f'No such exercise: {int(args.source)}.')
                exit(1)