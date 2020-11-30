import math
from  skimage import io,color
import numpy as np
from tqdm import trange

class Cluster():
    cluster_index = 1
    def __init__(self, h, w, l=0, a=0, b=0):
        self.updata(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        self.cluster_index += 1

    def updata(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return f'{self.h},{self.w}:{self.l} {self.a} {self.b}'

    def __repr__(self):
        return self.__str__()

class SLICProcessor():
    @staticmethod
    def open_image(path):
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr
    @staticmethod
    def save_lab_image(path, lab_arr):
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)
    
    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w, self.data[h][w][0], self.data[h][w][1], self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K
        self.M = M
        self.filename = filename.split('.')[0]

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))
        print(f'height:{self.image_height}, width:{self.image_width}')
        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2
        gradient = self.data[h+1][w+1][0] - self.data[h][w][0] + \
                    self.data[h+1][w+1][1] - self.data[h][w][1] + \
                    self.data[h+1][w+1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    h_ = cluster.h + dh
                    w_ = cluster.w + dw
                    new_gradient = self.get_gradient(h_, w_)
                    if new_gradient < cluster_gradient:
                        cluster.updata(h_, w_, self.data[h_][w_][0], self.data[h_][w_][1], self.data[h_][w_][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: 
                    continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: 
                        continue
                    # print(f'h :{h}, w:{w}')
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l,2) + 
                        math.pow(A - cluster.a,2) + 
                        math.pow(B - cluster.b,2) 
                    )
                    Ds = math.sqrt(
                        math.pow(h - cluster.h,2) + 
                        math.pow(w - cluster.w,2)
                    )
                    D = math.sqrt( math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2) )
                    if D < self.dis[h][w]:
                        if (h, w) in self.label:
                            self.label[(h, w)].pixels.remove((h, w))
                        self.label[(h, w)] = cluster
                        cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def updata_cluster(self):
        for cluter in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluter.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            h_ = int(sum_h / number)
            w_ = int(sum_w / number)
            cluter.updata(h_, w_, self.data[h_][w_][0], self.data[h_][w_][1], self.data[h_][w_][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    def iterate_times(self, times=10):
        self.init_clusters()
        self.move_clusters()
        for _ in range(times):
            self.assignment()
            self.updata_cluster()
            name = f'{self.filename}_M{self.M}_K{self.K}_loop{_}.png'
            self.save_current_image(name)
    
if __name__ == "__main__":
    slic = SLICProcessor('pic01.jpeg', 200, 40)
    slic.iterate_times()
