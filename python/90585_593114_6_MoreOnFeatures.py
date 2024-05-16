import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../resources/messi.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

try:
    sift = cv2.SIFT()
except:
    sift = cv2.xfeatures2d.SIFT_create()

kp, descriptors = sift.detectAndCompute(gray, mask=None)
# print descriptors
# print len(kp), descriptors.shape

flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT
try:
    image = cv2.drawKeypoints(gray, kp, flags=flags)
except:
    cv2.drawKeypoints(gray, kp, image, flags=flags)

fig = plt.figure()
fig.set_size_inches(10, 10)
fig.add_subplot(1,1,1)
plt.imshow(image)
plt.show()

# Let's get our hands dirty :D
# Now we'll try to implement our own SIFT
import math
import random


class Descriptor(object):
    def __init__(self, x, y, feature_vector):
        self.x = x
        self.y = y
        self.feature_vector = feature_vector


class Keypoint(object):
    def __init__(self, x, y, magnitude, orientation, scale):
        self.x = x
        self.y = y
        self.magnitude = magnitude
        self.orientation = orientation
        self.scale = scale


class SIFT(object):
    def __init__(self, image, octaves, intervals):
        # CONSTANTS
        self.sigma_antialias = 0.5
        self.sigma_preblur = 1.0
        self.edge_threshold = 7.2
        self.intensity_threshold = 0.05
        self.pi = 3.1415926535897932384626433832795
        self.bins = 36
        self.max_kernel_dim = 20
        self.feature_win_dim = 16
        self.descriptor_bins = 8
        self.feature_vector_size = 128
        self.feature_vector_threshold = 0.2
        self.num_keypoints = 0
        self.keypoints = []
        self.descriptors = []
        
        self.image = image.copy()  # colored image (BGR format)
        self.octaves = octaves
        self.intervals = intervals
        
        self.gaussian = [[None for interval in range(self.intervals + 3)] for octave in range(self.octaves)]
        self.dog = [[None for interval in range(self.intervals + 2)] for octave in range(self.octaves)]
        self.extrema = [[None for interval in range(self.intervals)] for octave in range(self.octaves)]
        self.sigma_level =[[None for interval in range(self.intervals + 3)] for octave in range(self.octaves)]
        
    def build_scale_space(self):
        gray = np.float32(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                gray[i, j] /= 255.0
        
        ksize = int(3 * self.sigma_antialias)
        if ksize % 2 == 0: ksize += 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), self.sigma_antialias)
        
        self.gaussian[0][0] = np.float32(cv2.pyrUp(gray))
        ksize = int(3 * self.sigma_preblur)
        if ksize % 2 == 0: ksize += 1
        self.gaussian[0][0] = np.float32(
            cv2.GaussianBlur(self.gaussian[0][0], (ksize, ksize), self.sigma_preblur)
        )
        
        sigma_init = math.sqrt(2.0)
        self.sigma_level[0][0] = sigma_init * 0.5
        
        for i in range(self.octaves):
            sigma = sigma_init
            
            for j in range(1, self.intervals + 3):
                sigma_next = math.sqrt((2 ** (2.0 / self.intervals)) - 1) * sigma
                sigma *= 2 ** (1.0 / self.intervals)
                
                self.sigma_level[i][j] = sigma * 0.5 * (2 ** i)
                ksize = int(3 * sigma_next)
                if ksize % 2 == 0: ksize += 1
                
                self.gaussian[i][j] = np.float32(
                    cv2.GaussianBlur(self.gaussian[i][j - 1], (ksize, ksize), sigma_next)
                )
                
                self.dog[i][j - 1] = np.float32(cv2.subtract(self.gaussian[i][j - 1], self.gaussian[i][j]))
            
            if i != self.octaves - 1:
                self.gaussian[i + 1][0] = np.float32(cv2.pyrDown(self.gaussian[i][0]))
                self.sigma_level[i + 1][0] = self.sigma_level[i][self.intervals]
    
    def detect_extrema(self):
        num_keypoints = 0
        
        for i in range(self.octaves):
            for j in range(1, self.intervals + 1):
                self.extrema[i][j - 1] = np.float32(np.zeros_like(self.dog[i][0]))
                
                middle = self.dog[i][j]
                above = self.dog[i][j + 1]
                below = self.dog[i][j - 1]
                
                for x in range(1, self.dog[i][j].shape[0] - 1):
                    for y in range(1, self.dog[i][j].shape[1] - 1):
                        flag = False
                        
                        pixel = middle[x, y]
                        
                        # Sigh! Have to check against 26 pixel if you remember (9 + 9 + 8)
                        values = [
                            middle[x - 1, y - 1], middle[x, y - 1], middle[x + 1, y - 1],
                            middle[x - 1, y], middle[x, y], middle[x + 1, y],
                            middle[x - 1, y + 1], middle[x, y + 1], middle[x + 1, y + 1],
                            
                            above[x - 1, y - 1], above[x, y - 1], above[x + 1, y - 1],
                            above[x - 1, y], above[x, y], above[x + 1, y],
                            above[x - 1, y + 1], above[x, y + 1], above[x + 1, y + 1],
                            
                            below[x - 1, y - 1], below[x, y - 1], below[x + 1, y - 1],
                            below[x - 1, y], below[x, y], below[x + 1, y],
                            below[x - 1, y + 1], below[x, y + 1], below[x + 1, y + 1]
                        ]
                        values.sort()
                        
                        # Check for maxmima
                        if values[-1] == pixel and values[-1] != values[-2]:
                            flag = True
                            self.extrema[i][j - 1][x, y] = 255
                            num_keypoints += 1
                        # Check for minima
                        elif values[0] == pixel and values[0] != values[1]:
                            flag = True
                            self.extrema[i][j - 1][x, y] = 255
                            num_keypoints += 1
                        
                        # Intensity check
                        if flag and math.fabs(middle[x, y]) < self.intensity_threshold:
                            self.extrema[i][j - 1][x, y] = 0
                            num_keypoints -= 1
                            flag = False
                        
                        # Edge check
                        if flag:
                            # Using Hessian Matrix
                            dx2 = middle[x, y - 1] + middle[x, y + 1] - 2 * middle[x, y]
                            dy2 = middle[x - 1, y] + middle[x + 1, y] - 2 * middle[x, y]
                            dxy = (
                                middle[x - 1, y - 1] + 
                                middle[x - 1, y + 1] + 
                                middle[x + 1, y - 1] + 
                                middle[x + 1, y + 1]
                            )
                            dxy /= 4.0
                            
                            tr = dx2 + dy2
                            det = dx2 * dy2 + dxy ** 2
                            
                            curvature_ratio = (tr ** 2) / det
                            if det < 0 or curvature_ratio > self.edge_threshold:
                                self.extrema[i][j - 1][x, y] = 0
                                num_keypoints -= 1

        self.num_keypoints = num_keypoints
    
    def assign_orientation(self):
        magnitudes = [[None for interval in range(self.intervals)] for octave in range(self.octaves)]
        orientations = [[None for interval in range(self.intervals)] for octave in range(self.octaves)]
        
        for i in range(self.octaves):
            for j in range(1, self.intervals):
                magnitudes[i][j - 1] = np.float32(np.zeros_like(self.gaussian[i][j]))
                orientations[i][j - 1] = np.float32(np.zeros_like(self.gaussian[i][j]))
                
                for x in range(1, self.gaussian[i][j].shape[0] - 1):
                    for y in range(1, self.gaussian[i][j].shape[1] - 1):
                        # Calculate gradient
                        dx = self.gaussian[i][j][x + 1, y] - self.gaussian[i][j][x - 1, y]
                        dy = self.gaussian[i][j][x, y + 1] - self.gaussian[i][j][x, y - 1]
                        
                        magnitudes[i][j - 1][x, y] = math.sqrt(dx ** 2 + dy ** 2)
                        orientations[i][j - 1][x, y] = math.atan2(dy, dx)
        
        for i in range(self.octaves):
            scale = 2.0 ** i
            
            for j in range(1, self.intervals):
                sigma = self.sigma_level[i][j]
                
                ksize = int(3 * 1.5 * sigma)
                if ksize % 2 == 0: ksize += 1
                
                weighted = np.float32(cv2.GaussianBlur(magnitudes[i][j - 1], (ksize, ksize), 1.5 * sigma))
                approx_gaussian_kernel_size = ksize / 2
                
                mask = np.float32(np.zeros_like(self.gaussian[i][0]))
                
                for x in range(self.gaussian[i][0].shape[0]):
                    for y in range(self.gaussian[i][0].shape[1]):
                        if self.extrema[i][j - 1][x, y] != 0:
                            orientation_hist = [0.0 for b in range(self.bins)]
                            
                            for ii in range(-approx_gaussian_kernel_size, approx_gaussian_kernel_size + 1):
                                for jj in range(-approx_gaussian_kernel_size, approx_gaussian_kernel_size + 1):
                                    if x + ii < 0 or x + ii >= self.gaussian[i][0].shape[0]: continue
                                    if y + jj < 0 or y + jj >= self.gaussian[i][0].shape[1]: continue
                                    
                                    sampled_orientation = orientations[i][j - 1][x + ii, y + jj]
                                    sampled_orientation += self.pi
                                    
                                    degrees = sampled_orientation * 180 / self.pi
                                    orientation_hist[int(degrees * self.bins / 360)] += weighted[x + ii, y + jj]
                                    
                                    mask[x + ii, y + jj] = 255
                            
                            max_peak = max(orientation_hist)
                            max_peak_index = orientation_hist.index(max_peak)
                            
                            o = []
                            m = []
                            
                            for k in range(self.bins):
                                if orientation_hist[k] > 0.8 * max_peak:
                                    x1 = k - 1
                                    x2 = k
                                    y2 = orientation_hist[k]
                                    x3 = k + 1
                                    
                                    if k == 0:
                                        y1 = orientation_hist[self.bins - 1]
                                        y3 = orientation_hist[k + 1]
                                    elif k == self.bins - 1:
                                        y1 = orientation_hist[k - 1]
                                        y3 = orientation_hist[0]
                                    else:
                                        y1 = orientation_hist[k - 1]
                                        y3 = orientation_hist[k + 1]
                                    
                                    # Fit a down facing parabola to the above points (x1, y1), (x2, y2), (x3, y3)
                                    # y = a*x*x + b*x + c
                                    # y1 = a*x1*x1 + b*x1 + c
                                    # y1 = a*x2*x2 + b*x2 + c
                                    # y1 = a*x3*x3 + b*x3 + c
                                    # Y = X * Transpose([a, b, c]) (= L, say)
                                    # L = inverse(X) * Y
                                    
                                    X = np.array([
                                        [x1*x1, x1, 1],
                                        [x2*x2, x2, 1],
                                        [x3*x3, x3, 1]
                                    ])
                                    
                                    Y = np.array([
                                        [y1],
                                        [y2],
                                        [y3]
                                    ])
                                    
                                    L = np.dot(np.invert(X), Y)
                                    
                                    # So, now we have a,b,c for our parabola equation,let's find vertex
                                    x0 = -L[1] / (2 * L[0])
                                    if math.fabs(x0) > 2 * self.bins: x0 = x2
                                    while(x0 < 0): x0 += self.bins
                                    while(x0 > self.bins): x0 -= self.bins
                                    
                                    x0_normalized = x0 * (2 * self.pi / self.bins)
                                    x0_normalized -= self.pi
                                    
                                    o.append(x0_normalized)
                                    m.append(orientation_hist[k])
                            
                            self.keypoints.append(Keypoint(
                                x * scale / 2.0, y * scale / 2.0, m, o, i * self.intervals + j - 1
                            ))

    def generate_features(self):
        magnitudes_interpolated = [
            [
                None for interval in range(self.intervals)
            ] for octave in range(self.octaves)
        ]
        orientations_interpolated = [
            [
                None for interval in range(self.intervals)
            ] for octave in range(self.octaves)
        ]
        
        for i in range(self.octaves):
            for j in range(1, self.intervals + 1):
                temp = np.float32(cv2.pyrUp(self.gaussian[i][j]))
                
                magnitudes_interpolated[i][j - 1] = np.float32(
                    np.zeros_like(
                        cv2.resize(
                            self.gaussian[i][j],
                            (self.gaussian[i][j].shape[1] + 1, self.gaussian[i][j].shape[0] + 1)
                        )
                    )
                )
                orientations_interpolated[i][j - 1] = np.float32(
                    np.zeros_like(
                        cv2.resize(
                            self.gaussian[i][j],
                            (self.gaussian[i][j].shape[1] + 1, self.gaussian[i][j].shape[0] + 1)
                        )
                    )
                )
                
                ii = 1.5
                while(ii < self.gaussian[i][j].shape[0] - 1.5):
                    jj = 1.5
                    
                    while(jj < self.gaussian[i][j].shape[1] - 1.5):
                        ii1 = int(ii + 1.5)
                        ii2 = int(ii + 0.5)
                        ii3 = int(ii)
                        ii4 = int(ii - 0.5)
                        ii5 = int(ii - 1.5)
                        
                        jj1 = int(jj + 1.5)
                        jj2 = int(jj + 0.5)
                        jj3 = int(jj)
                        jj4 = int(jj - 0.5)
                        jj5 = int(jj - 1.5)
                        
                        dx = (
                            (self.gaussian[i][j][ii1, jj3] + self.gaussian[i][j][ii2, jj3]) / 2.0 -
                            (self.gaussian[i][j][ii5, jj3] + self.gaussian[i][j][ii4, jj3]) / 2.0
                        )
                        dy = (
                            (self.gaussian[i][j][ii3, jj1] + self.gaussian[i][j][ii3, jj2]) / 2.0 -
                            (self.gaussian[i][j][ii3, jj5] + self.gaussian[i][j][ii3, jj4]) / 2.0
                        )
                        
                        x_ = int(ii + 1)
                        y_ = int(jj + 1)
                        
                        magnitudes_interpolated[i][j - 1][x_, y_] = math.sqrt(dx ** 2 + dy ** 2)
                        if math.atan2(dy, dx) == self.pi: orientations_interpolated[i][j - 1][x_, y_] = -self.pi
                        else: orientations_interpolated[i][j - 1][x_, y_] = math.atan2(dy, dx)
                        
                        jj += 1
                    ii += 1
                
                for ii in range(self.gaussian[i][j].shape[0] + 1):
                    magnitudes_interpolated[i][j - 1][ii, 0] = 0
                    magnitudes_interpolated[i][j - 1][ii, self.gaussian[i][j].shape[1] - 1] = 0
                    orientations_interpolated[i][j - 1][ii, 0] = 0
                    orientations_interpolated[i][j - 1][ii, self.gaussian[i][j].shape[1] - 1] = 0
                
                for jj in range(self.gaussian[i][j].shape[1] + 1):
                    magnitudes_interpolated[i][j - 1][0, jj] = 0
                    magnitudes_interpolated[i][j - 1][self.gaussian[i][j].shape[0] - 1, jj] = 0
                    orientations_interpolated[i][j - 1][0, jj] = 0
                    orientations_interpolated[i][j - 1][self.gaussian[i][j].shape[0] - 1, jj] = 0
        
        G = self.interpolated_gaussian(self.feature_win_dim, 0.5 * self.feature_win_dim)
        
        buggy_keypoints = []

        for keypoint in self.keypoints:
            scale = keypoint.scale
            x = keypoint.x
            y = keypoint.y
            
            ii = int(x * 2) / int(2.0 ** (scale / self.intervals))
            jj = int(y * 2) / int(2.0 ** (scale / self.intervals))
            
            orientations = keypoint.orientation
            magnitudes = keypoint.magnitude
            
            main_orientation = orientations[0]
            main_magnitude = magnitudes[0]
            
            for i in range(len(magnitudes)):
                if magnitudes[i] > main_magnitude:
                    main_orientation = orientations[i]
                    main_magnitude = magnitudes[i]
            
            half_kernel_size = self.feature_win_dim / 2
            weights = np.float32(np.zeros((self.feature_win_dim, self.feature_win_dim)))
            
            for i in range(self.feature_win_dim):
                for j in range(self.feature_win_dim):
                    if ii + i + 1 < half_kernel_size:
                        weights[i, j] = 0
                    elif ii + i + 1 > half_kernel_size + self.gaussian[scale / self.intervals][0].shape[1]:
                        weights[i, j] = 0
                    elif jj + j + 1 < half_kernel_size:
                        weights[i, j] = 0
                    elif jj + j + 1 > half_kernel_size + self.gaussian[scale / self.intervals][0].shape[0]:
                        weights[i, j] = 0
                    else:
                        val = magnitudes_interpolated[scale / self.intervals][scale % self.intervals]
                        val = val[ii + i + 1 - half_kernel_size, jj + j + 1 - half_kernel_size]
                        
                        weights[i, j] = (G[i, j] * val)
            
            # 16 4x4 blocks
            feature_vector = [0.0 for s in range(self.feature_vector_size)]
            for i in range(self.feature_win_dim/4):
                for j in range(self.feature_win_dim/4):
                    hist = [0.0 for b in range(self.descriptor_bins)]
                    
                    start_i = int(ii - half_kernel_size) + 1 + int(half_kernel_size / 2 * i)
                    start_j = int(jj - half_kernel_size) + 1 + int(half_kernel_size / 2 * j)
                    
                    limit_i = int(ii) + int(half_kernel_size / 2) * (i - 1)
                    limit_j = int(jj) + int(half_kernel_size / 2) * (j - 1)
                    
                    for iii in range(start_i, limit_i + 1):
                        for jjj in range(start_j, limit_j + 1):
                            if iii < 0 or iii >= self.gaussian[scale / self.intervals][0].shape[1]: continue
                            if jjj < 0 or jjj >= self.gaussian[scale / self.intervals][0].shape[0]: continue
                            
                            # Rotation invariance
                            sampled = orientations_interpolated[scale / self.intervals][scale % self.intervals]
                            
                            sampled_orientation = sampled[iii, jjj] - main_orientation
                            while sampled_orientation < 0: sampled_orientation += (2 * self.pi)
                            while sampled_orientation > 2 * self.pi: sampled_orientation -= (2 * self.pi)
                            
                            degrees = sampled_orientation * 180 / self.pi
                            bin = degrees * self.descriptor_bins / 360.0
                            
                            w = weights[iii + half_kernel_size - ii - 1, jjj + half_kernel_size - jj - 1]
                            hist[int(bin)] += (1 - math.fabs(bin - int(bin) - 0.5)) * w
                    
                    for k in range(self.descriptor_bins):
                        feature_vector[(i * self.feature_win_dim / 4 + j) * self.descriptor_bins + k] += hist[k]
            
            # Illumination invariance
            try:
                norm = 0
                for i in range(self.feature_vector_size):
                    norm += (feature_vector[i] ** 2)
                norm = math.sqrt(norm)

                for i in range(self.feature_vector_size):
                    feature_vector[i] /= norm
                    if feature_vector[i] > self.feature_vector_threshold:
                        feature_vector[i] = self.feature_vector_threshold

                norm = 0
                for i in range(self.feature_vector_size):
                    norm += feature_vector[i] ** 2
                norm = math.sqrt(norm)

                for i in range(self.feature_vector_size):
                    feature_vector[i] /= norm

                self.descriptors.append(Descriptor(x, y, feature_vector))
            except:
                buggy_keypoints.append(keypoint)
                self.num_keypoints -= 1
        
        for bug in buggy_keypoints:
            self.keypoints.remove(bug)
                
    
    def draw_keypoints(self):
        img = self.image.copy()
        
        for kp in self.keypoints:
            r = int(random.random() * 500)
            while r > 255: r -= 50
            g = int(random.random() * 500)
            while g > 255: g -= 50
            b = int(random.random() * 500)
            while b > 255: b -= 50
            
            color = (b, g, r)
            
            cv2.line(
                img,
                (int(kp.x), int(kp.y)),
                (int(kp.x + 10 * math.cos(kp.orientation[0])), int(kp.y + 10 * math.sin(kp.orientation[0]))),
                color,
                2
            )
            cv2.circle(
                img,
                (int(kp.x), int(kp.y)),
                10,
                color,
                2
            )
        
        return img
                
    def interpolated_gaussian(self, size, sigma):
        half_size = size / 2 - 0.5
        sog = 0
        ret = np.float32(np.zeros((size, size), dtype=np.float32))
        
        for i in range(size):
            for j in range(size):
                x, y = i - half_size, j - half_size
                temp = 1.0 / (2 * self.pi * (sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))
                ret[i, j] = temp
                sog += temp
        
        for i in range(size):
            for j in range(size):
                ret[i, j] *= (1.0 / sog)
        
        return ret

image = cv2.imread("../resources/messi.jpg")

sift = SIFT(image, 4, 2)

sift.build_scale_space()
sift.detect_extrema()
sift.assign_orientation()
sift.generate_features()

out = sift.draw_keypoints()

fig = plt.figure()
fig.set_size_inches(10, 10)
fig.add_subplot(1,1,1)
plt.imshow(out)
plt.show()



