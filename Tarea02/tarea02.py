import os
import sys
sys.path.append('./pai_basis')
import basis
import pai_io
import orientation_histograms as oh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.ndimage.filters as nd_filters
import math
import skimage.feature as feature
import skimage

# from scipy.ndimage import gaussian_filter

def compare(h1, h2):
    d = 0
    for i in range(len(h1)):
        d = d + np.power(h1[i] - h2[i], 2)
    return math.sqrt(d)


def first(p):
    return p[0]


def mAP(d1, d2):
    AP = []
    counter = 0
    for key in d1:
        counter = counter + 1
        ranks = []
        test = []
        for obj in d2:
            for hist in d2[obj]:
                ranks.append((compare(d1[key], hist), key==obj))
        ranks.sort(key=first, reverse=False)
        precision = []
        hit = 0
        for i in range(20):
            if ranks[i][1]:
                hit = hit + 1
                precision.append(hit/(i + 1))
            else:
                precision.append(0)
        AP.append((sum(precision) + ((len(d2[key]) - hit)/50))/len(d2[key]))
    return sum(AP)/counter


def histogram_ho(image, k):
    h = np.zeros(k , np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    ang = np.arctan2(gy,gx)
    ang[ang < 0] = ang[ang < 0] + np.pi #sin signo
    mag = np.sqrt(np.square(gy) + np.square(gx))
    indx = np.round(k * ang / np.pi)
    indx[indx ==  k] = 0
    for i in range(k):
        rows, cols = np.where(indx  == i)
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)  #vector unitario
    return h


def histogram_helo(image, B, k):
    # image = gaussian_filter(image, sigma=1.5)
    h = np.zeros(k , np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    Lx = np.zeros((B, B) , np.float32)
    Ly = np.zeros((B, B) , np.float32)

    t1 = np.power(gx, 2) - np.power(gy, 2)
    t2 = 2*gx*gy

    dx = int(image.shape[0]/B)
    dy = int(image.shape[1]/B)

    mask = np.ones((dx, dy))

    t1 = nd_filters.convolve(t1, mask)
    t2 = nd_filters.convolve(t2, mask)

    for i in range(B):
        for j in range(B):
            Lx[i][j] = t1[int(dx/2) + dx*i][int(dy/2) + dy*j]
            Ly[i][j] = t2[int(dx/2) + dx*i][int(dy/2) + dy*j]
    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)

    angles = np.arctan2(Ly, Lx)/2
    mag = np.sqrt(np.square(Lx) + np.square(Ly))

    angles[angles < 0] = angles[angles < 0] + np.pi
    indx = np.round(k*angles/np.pi)
    indx[indx == k] = 0

    for i in range(k):
        rows, cols = np.where(indx  == i)
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)

    return h


def histogram_shelo(image, B, k):
    # image = gaussian_filter(image, sigma=1.5)
    h = np.zeros(k , np.float32)
    dx = int(image.shape[0]/B)
    dy = int(image.shape[1]/B)

    maskx = np.zeros((2*dx, 2*dy))
    masky = np.zeros((2*dx, 2*dy))

    for i in range(2*dx):
        for j in range(2*dy):
            if (i + 0.5) <= dx:
                maskx[i][j] = (i + 0.5)/dx
            else:
                maskx[i][j] = 1 - ((i + 0.5 - dx)/dx)

    for i in range(2*dx):
        for j in range(2*dy):
            if (j + 0.5) <= dy:
                masky[i][j] = (j + 0.5)/dy
            else:
                masky[i][j] = 1 - ((j + 0.5 - dy)/dy)

    mask = maskx*masky

    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)

    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    A = np.zeros((B, B) , np.float32)
    D = np.zeros((B, B) , np.float32)

    t1 = np.power(gx, 2) - np.power(gy, 2)
    t2 = 2*gx*gy

    t1 = nd_filters.convolve(t1, mask)
    t2 = nd_filters.convolve(t2, mask)

    for i in range(B):
        for j in range(B):
            A[i][j] = t1[int(dx/2) + dx*i][int(dy/2) + dy*j]
            D[i][j] = t2[int(dx/2) + dx*i][int(dy/2) + dy*j]

    angles = np.arctan2(D, A)/2
    mag = np.sqrt(np.square(D) + np.square(A))

    angles[angles < 0] = angles[angles < 0] + np.pi
    indx = np.round(k*angles/np.pi)
    indx[indx == k] = 0

    for i in range(k):
        rows, cols = np.where(indx  == i)
        h[i] = np.sum(mag[rows, cols])
    h =  h / np.linalg.norm(h,2)

    return h


def overlapped_helo(image, B, k):
    # image = gaussian_filter(image, sigma=1.5)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    Lx = np.zeros((B, B) , np.float32)
    Ly = np.zeros((B, B) , np.float32)

    t1 = np.power(gx, 2) - np.power(gy, 2)
    t2 = 2*gx*gy

    dx = int(image.shape[0]/B)
    dy = int(image.shape[1]/B)

    mask = np.ones((dx, dy))

    t1 = nd_filters.convolve(t1, mask)
    t2 = nd_filters.convolve(t2, mask)

    for i in range(B):
        for j in range(B):
            Lx[i][j] = t1[int(dx/2) + dx*i][int(dy/2) + dy*j]
            Ly[i][j] = t2[int(dx/2) + dx*i][int(dy/2) + dy*j]
    
    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)

    angles = np.arctan2(Ly, Lx)/2
    mag = np.sqrt(np.square(Lx) + np.square(Ly))

    max_mag = np.amax(mag)

    fig, xs = plt.subplots(1,2)
    for i in range(2):
        xs[i].set_axis_off()
    xs[0].imshow(image, cmap = 'gray')
    xs[0].set_title('Image')
    xs[1].imshow(image, cmap = 'gray')
    xs[1].set_title('Orientations')
    dx = image.shape[0]/B
    dy = image.shape[1]/B
    for i in range(B):
        for j in range(B):
            if mag[i][j] > max_mag/16:
                x1 = dx*(i + 1/2 + math.cos(-angles[i][j])/2)
                x2 = dx*(i + 1/2 - math.cos(-angles[i][j])/2)
                y1 = dy*(j + 1/2 + math.sin(-angles[i][j])/2)
                y2 = dy*(j + 1/2 - math.sin(-angles[i][j])/2)
                xs[1].plot([y1, y2], [x1, x2], 'y')
    plt.show()


def overlapped_shelo(image, B, k):
    dx = int(image.shape[0]/B)
    dy = int(image.shape[1]/B)

    maskx = np.zeros((2*dx, 2*dy))
    masky = np.zeros((2*dx, 2*dy))

    for i in range(2*dx):
        for j in range(2*dy):
            if (i + 0.5) <= dx:
                maskx[i][j] = (i + 0.5)/dx
            else:
                maskx[i][j] = 1 - ((i + 0.5 - dx)/dx)

    for i in range(2*dx):
        for j in range(2*dy):
            if (j + 0.5) <= dy:
                masky[i][j] = (j + 0.5)/dy
            else:
                masky[i][j] = 1 - ((j + 0.5 - dy)/dy)

    mask = maskx*masky

    # Lx = gaussian_filter(Lx, sigma=0.5)
    # Ly = gaussian_filter(Ly, sigma=0.5)

    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    A = np.zeros((B, B) , np.float32)
    D = np.zeros((B, B) , np.float32)

    t1 = np.power(gx, 2) - np.power(gy, 2)
    t2 = 2*gx*gy

    t1 = nd_filters.convolve(t1, mask)
    t2 = nd_filters.convolve(t2, mask)

    for i in range(B):
        for j in range(B):
            A[i][j] = t1[int(dx/2) + dx*i][int(dy/2) + dy*j]
            D[i][j] = t2[int(dx/2) + dx*i][int(dy/2) + dy*j]

    angles = np.arctan2(D, A)/2
    mag = np.sqrt(np.square(D) + np.square(A))

    max_mag = np.amax(mag)

    fig, xs = plt.subplots(1,2)
    for i in range(2):
        xs[i].set_axis_off()
    xs[0].imshow(image, cmap = 'gray')
    xs[0].set_title('Image')
    xs[1].imshow(image, cmap = 'gray')
    xs[1].set_title('Orientations')
    dx = image.shape[0]/B
    dy = image.shape[1]/B
    for i in range(B):
        for j in range(B):
            if mag[i][j] > max_mag/16:
                x1 = dx*(i + 1/2 + math.cos(-angles[i][j])/2)
                x2 = dx*(i + 1/2 - math.cos(-angles[i][j])/2)
                y1 = dy*(j + 1/2 + math.sin(-angles[i][j])/2)
                y2 = dy*(j + 1/2 - math.sin(-angles[i][j])/2)
                xs[1].plot([y1, y2], [x1, x2], 'y')
    plt.show()


if __name__ == '__main__' : 
    if len(sys.argv) == 2:
        if sys.argv[1] == "setup":
            path = './dataset_1/BD_2/'

            folders = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(path):
                for directory in d:
                    if directory != '':
                        folders.append(directory)

            objects = {}

            for obj in folders:
                files = []
                path2 = path + obj
                for r, d, f in os.walk(path2):
                    for file in f:
                        if '.jpg' in file:
                            files.append(os.path.join(r, file))
                objects[obj] = files

            K = [36, 72, 96]
            B = [25, 12]

            for bins in K:
                for obj in objects:
                    results_nc = []
                    results_c = []
                    for path in objects[obj]:
                        image = pai_io.imread(path, as_gray = True)
                        canny_image = 1 - feature.canny(image/255, sigma=1.5)
                        results_nc.append(histogram_ho(image, bins))
                        results_c.append(histogram_ho(canny_image, bins))
                    results_nc = np.array(results_nc)
                    results_c = np.array(results_c)
                    np.save("ho_nc_" + obj + "_" + str(bins), results_nc)
                    np.save("ho_c_" + obj + "_" + str(bins), results_c)
                for blocks in B:
                    for obj in objects:
                        results_helo_nc = []
                        results_helo_c = []
                        results_shelo_nc = []
                        results_shelo_c = []
                        for path in objects[obj]:
                            image = pai_io.imread(path, as_gray = True)
                            canny_image = 1 - feature.canny(image/255, sigma=1.5)
                            results_helo_nc.append(histogram_helo(image, blocks, bins))
                            results_helo_c.append(histogram_helo(canny_image, blocks, bins))
                            results_shelo_nc.append(histogram_shelo(image, blocks, bins))
                            results_shelo_c.append(histogram_shelo(canny_image, blocks, bins))
                        results_helo_nc = np.array(results_helo_nc)
                        results_helo_c = np.array(results_helo_c)
                        results_shelo_nc = np.array(results_shelo_nc)
                        results_shelo_c = np.array(results_shelo_c)
                        np.save("helo_nc_" + obj + "_" + str(bins) + "_" + str(blocks), results_helo_nc)
                        np.save("helo_c_" + obj + "_" + str(bins) + "_" + str(blocks), results_helo_c)
                        np.save("shelo_nc_" + obj + "_" + str(bins) + "_" + str(blocks), results_shelo_nc)
                        np.save("shelo_c_" + obj + "_" + str(bins) + "_" + str(blocks), results_shelo_c)
        elif sys.argv[1] == "setup_sketches":
            path = './dataset_1/queries/'
            data = {}
            for r, d, f in os.walk(path):
                for file in f:
                    if '.jpg' in file:
                        data[file.replace(".jpg", "")] = os.path.join(r, file)

            K = [36, 72, 96]
            B = [25, 12]

            for bins in K:
                for obj in data:
                    image = pai_io.imread(data[obj], as_gray = True)
                    canny_image = 1 - feature.canny(image/255, sigma=1.5)
                    np.save("sketch_ho_nc_" + obj + "_" + str(bins), histogram_ho(image, bins))
                    np.save("sketch_ho_c_" + obj + "_" + str(bins), histogram_ho(canny_image, bins))
                for blocks in B:
                    for obj in data:
                        image = pai_io.imread(data[obj], as_gray = True)
                        canny_image = 1 - feature.canny(image/255, sigma=1.5)
                        np.save("sketch_helo_nc_" + obj + "_" + str(bins) + "_" + str(blocks), histogram_helo(image, blocks, bins))
                        np.save("sketch_helo_c_" + obj + "_" + str(bins) + "_" + str(blocks), histogram_helo(canny_image, blocks, bins))
                        np.save("sketch_shelo_nc_" + obj + "_" + str(bins) + "_" + str(blocks), histogram_shelo(image, blocks, bins))
                        np.save("sketch_shelo_c_" + obj + "_" + str(bins) + "_" + str(blocks), histogram_shelo(canny_image, blocks, bins))
        elif sys.argv[1] == "compare":
            K = [36, 72, 96]
            B = [25, 12]

            path_helo = './HIST/HELO/'
            path_shelo = './HIST/SHELO/'
            path_ho = './HIST/HO/'
            path_sketch_helo = './SKETCH/HELO/'
            path_sketch_shelo = './SKETCH/SHELO/'
            path_sketch_ho = './SKETCH/HO/'

            ho_sketch = {}
            ho_hist = {}
            helo_sketch = {}
            helo_hist = {}
            shelo_sketch = {}
            shelo_hist = {}

            canny_ho_sketch = {}
            canny_ho_hist = {}
            canny_helo_sketch = {}
            canny_helo_hist = {}
            canny_shelo_sketch = {}
            canny_shelo_hist = {}

            translation = {}
            translation['000001'] = "accordion"
            translation['000002'] = "airplane"
            translation['000003'] = "airplane"
            translation['000004'] = "airplane"
            translation['000005'] = "anchor"
            translation['000006'] = "anchor"
            translation['000007'] = "ant"
            translation['000008'] = "cup"
            translation['000009'] = "soccer_ball"
            translation['000010'] = "barrel"
            translation['000011'] = "barrel"
            translation['000012'] = "bass"
            translation['000013'] = "chair"
            translation['000014'] = "ant"
            translation['000015'] = "ant"
            translation['000016'] = "airplane"
            translation['000017'] = "beaver"
            translation['000018'] = "anchor"
            translation['000019'] = "anchor"
            translation['000020'] = "binocular"
            translation['000021'] = "binocular"
            translation['000022'] = "butterfly"
            translation['000023'] = "butterfly"
            translation['000024'] = "cannon"
            translation['000025'] = "cannon"
            translation['000026'] = "cup"
            translation['000027'] = "cup"
            translation['000028'] = "cup"
            translation['000029'] = "brontosaurus"
            translation['000030'] = "lamp"
            translation['000031'] = "nautilus"
            translation['000032'] = "lamp"
            translation['000033'] = "flamingo"
            translation['000034'] = "pyramid"
            translation['000035'] = "pyramid"
            translation['000036'] = "revolver"
            translation['000037'] = "saxophone"
            translation['000038'] = "scissors"
            translation['000039'] = "scissors"
            translation['000040'] = "soccer_ball"
            translation['000041'] = "strawberry"
            translation['000042'] = "strawberry"
            translation['000043'] = "tick"
            translation['000044'] = "lobster"
            translation['000045'] = "laptop"
            translation['000046'] = "umbrella"
            translation['000047'] = "watch"
            translation['000048'] = "palace"
            translation['000049'] = "palace"
            translation['000050'] = "brain"
            translation['000051'] = "castle"
            translation['000052'] = "cellphone"
            translation['000053'] = "dalmatian"

            for bins in K:
                for blocks in B:
                    ho_hist[str(bins)] = {}
                    helo_hist[str(bins) + "_" + str(blocks)] = {}
                    shelo_hist[str(bins) + "_" + str(blocks)] = {}
                    ho_sketch[str(bins)] = {}
                    helo_sketch[str(bins) + "_" + str(blocks)] = {}
                    shelo_sketch[str(bins) + "_" + str(blocks)] = {}

                    canny_ho_hist[str(bins)] = {}
                    canny_helo_hist[str(bins) + "_" + str(blocks)] = {}
                    canny_shelo_hist[str(bins) + "_" + str(blocks)] = {}
                    canny_ho_sketch[str(bins)] = {}
                    canny_helo_sketch[str(bins) + "_" + str(blocks)] = {}
                    canny_shelo_sketch[str(bins) + "_" + str(blocks)] = {}

            print("Finished initial setup")

            print("Starting HO")

            for r, d, f in os.walk(path_ho):
                for file in f:
                    name = file.replace(".npy", "")
                    name = name.replace("car_side", "car-side")
                    name = name.replace("cougar_body", "cougar-body")
                    name = name.replace("faces_easy", "faces-easy")
                    name = name.replace("grand_piano", "grand-piano")
                    name = name.replace("soccer_ball", "soccer-ball")
                    name = name.split("_")
                    if name[2] == "soccer-ball":
                        name[2] = "soccer_ball"
                    if name[1] == "nc":
                        ho_hist[name[3]][name[2]] = np.load(os.path.join(r, file))
                    else:
                        canny_ho_hist[name[3]][name[2]] = np.load(os.path.join(r, file))

            print("Starting HELO")

            for r, d, f in os.walk(path_helo):
                for file in f:
                    name = file.replace(".npy", "")
                    name = name.replace("car_side", "car-side")
                    name = name.replace("cougar_body", "cougar-body")
                    name = name.replace("faces_easy", "faces-easy")
                    name = name.replace("grand_piano", "grand-piano")
                    name = name.replace("soccer_ball", "soccer-ball")
                    name = name.split("_")
                    if name[2] == "soccer-ball":
                        name[2] = "soccer_ball"
                    if name[1] == "nc":
                        helo_hist[name[3] + "_" + name[4]][name[2]] = np.load(os.path.join(r, file))
                    else:
                        canny_helo_hist[name[3] + "_" + name[4]][name[2]] = np.load(os.path.join(r, file))

            print("Starting SHELO")
            
            for r, d, f in os.walk(path_shelo):
                for file in f:
                    name = file.replace(".npy", "")
                    name = name.replace("car_side", "car-side")
                    name = name.replace("cougar_body", "cougar-body")
                    name = name.replace("faces_easy", "faces-easy")
                    name = name.replace("grand_piano", "grand-piano")
                    name = name.replace("soccer_ball", "soccer-ball")
                    name = name.split("_")
                    if name[2] == "soccer-ball":
                        name[2] = "soccer_ball"
                    if name[1] == "nc":
                        shelo_hist[name[3] + "_" + name[4]][name[2]] = np.load(os.path.join(r, file))
                    else:
                        canny_shelo_hist[name[3] + "_" + name[4]][name[2]] = np.load(os.path.join(r, file))

            print("Finished loading histograms. Moving to sketches")

            print("Starting HO")

            for r, d, f in os.walk(path_sketch_ho):
                for file in f:
                    name = file.replace(".npy", "")
                    name = name.split("_")
                    if name[2] == "nc":
                        if not translation[name[4]] in ho_sketch[name[5]]:
                            ho_sketch[name[5]][translation[name[4]]] = {}
                        ho_sketch[name[5]][translation[name[4]]] = np.load(os.path.join(r, file))
                    else:
                        if not translation[name[4]] in canny_ho_sketch[name[5]]:
                            canny_ho_sketch[name[5]][translation[name[4]]] = {}
                        canny_ho_sketch[name[5]][translation[name[4]]] = np.load(os.path.join(r, file))

            print("Starting HELO")

            for r, d, f in os.walk(path_sketch_helo):
                for file in f:
                    name = file.replace(".npy", "")
                    name = name.split("_")
                    if name[2] == "nc":
                        if not translation[name[4]] in helo_sketch[name[5] + "_" + name[6]]:
                            helo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = {}
                        helo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = np.load(os.path.join(r, file))
                    else:
                        if not translation[name[4]] in canny_helo_sketch[name[5] + "_" + name[6]]:
                            canny_helo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = {}
                        canny_helo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = np.load(os.path.join(r, file))

            print("Starting SHELO")
            
            for r, d, f in os.walk(path_sketch_shelo):
                for file in f:
                    name = file.replace(".npy", "")
                    name = name.split("_")
                    if name[2] == "nc":
                        if not translation[name[4]] in shelo_sketch[name[5] + "_" + name[6]]:
                            shelo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = {}
                        shelo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = np.load(os.path.join(r, file))
                    else:
                        if not translation[name[4]] in canny_shelo_sketch[name[5] + "_" + name[6]]:
                            canny_shelo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = {}
                        canny_shelo_sketch[name[5] + "_" + name[6]][translation[name[4]]] = np.load(os.path.join(r, file))

            print("No Canny")

            for bins in K:
                print("HO K:" + str(bins) + " " + str(mAP(ho_sketch[str(bins)], ho_hist[str(bins)])))
                for blocks in B:
                    print("HELO K:" + str(bins) + " B:" + str(blocks) + " " + str(mAP(helo_sketch[str(bins) + "_" + str(blocks)], helo_hist[str(bins) + "_" + str(blocks)])))
                    print("SHELO K:" + str(bins) + " B:" + str(blocks) + " " + str(mAP(shelo_sketch[str(bins) + "_" + str(blocks)], shelo_hist[str(bins) + "_" + str(blocks)])))

            print("Canny")

            for bins in K:
                print("HO K:" + str(bins) + " " + str(mAP(canny_ho_sketch[str(bins)], canny_ho_hist[str(bins)])))
                for blocks in B:
                    print("HELO K:" + str(bins) + " B:" + str(blocks) + " " + str(mAP(canny_helo_sketch[str(bins) + "_" + str(blocks)], canny_helo_hist[str(bins) + "_" + str(blocks)])))
                    print("SHELO K:" + str(bins) + " B:" + str(blocks) + " " + str(mAP(canny_shelo_sketch[str(bins) + "_" + str(blocks)], canny_shelo_hist[str(bins) + "_" + str(blocks)])))

        elif sys.argv[1] == "test_helo":
            image = pai_io.imread("dataset_1/BD_2/chair/180016.jpg", as_gray = True)
            overlapped_shelo(image, 25, 36)
        elif sys.argv[1] == "test_shelo":
            image = pai_io.imread("dataset_1/BD_2/chair/180016.jpg", as_gray = True)
            overlapped_helo(image, 25, 36)

