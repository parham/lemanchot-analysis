

import cv2
import random
import numpy as np
from scipy.special import binom

from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([
            self.r*np.cos(self.angle1),
            self.r*np.sin(self.angle1)
        ])
        self.p[2,:] = self.p2 + np.array([
            self.r*np.cos(self.angle2+np.pi),
            self.r*np.sin(self.angle2+np.pi)
        ])
        self.curve = bezier(self.p,self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2], points[i+1,2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ 
    Given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of control points.
    *edgy* is a parameter which controls how "edgy" the curve is, edgy=0 is smoothest.
    """
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    num_points = len(c)
    return [tuple(c[index,:]) for index in range(num_points)]

def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

def generate_random_mask(
    img : np.ndarray,
    target,
    texture,
    texture_target,
    area, 
    rad = 0.2, 
    edgy = 0.05, 
    num_points=11
):
    bx = area[0]
    by = area[1]
    ex = area[2]
    ey = area[3]
    width = abs(ex - bx)
    height = abs(ey - by)
    # Generate random points
    rand_points = get_random_points(n=num_points, scale=1)
    # Form the polygon using the random points
    points = get_bezier_curve(rand_points, rad=rad, edgy=edgy)
    points = list(map(lambda p : (int(p[0] * width) + bx, int(p[1] * height) + by), points))
    # Create the mask
    mask_pil = Image.new("L", (img.shape[1], img.shape[0]), "black")
    draw = ImageDraw.Draw(mask_pil)
    draw.polygon(points, fill='white')
    mask = np.asarray(mask_pil)
    idx = (mask != 0)
    # Write the texture in the image
    channel = img.shape[-1]
    for cind in range(channel):
        tmp = img[:,:,cind]
        tmp_tex = texture[:,:,cind]
        tmp[idx] = tmp_tex[idx]
        img[:,:,cind] = tmp
    # Write the target in the image
    target[idx] = texture_target[idx]

    return img, target

def generate_augmented_texture(
    background : np.ndarray,
    dataset : Dataset,
    num_texture : int = 10,
    num_rand_points : int = 5
):
    img_target = np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8)
    shape = background.shape[:-1]

    row_count = 20
    col_count = 20
    row = np.append(np.array(range(row_count)) * (shape[0] // row_count), shape[0]-1)
    column = np.append(np.array(range(col_count)) * (shape[1] // col_count), shape[1]-1)

    data_loader = DataLoader(dataset, batch_size=num_texture, shuffle=True)
    batch = next(iter(data_loader))
    textures = batch[0]
    targets = batch[1]

    for index in range(num_texture):
        texture = np.resize(textures[index].cpu().detach().numpy(), background.shape)
        target = np.resize(targets[index].cpu().detach().numpy(), img_target.shape)
        row_select = sorted(random.choices(row, k=2))
        col_select = sorted(random.choices(column, k=2))
        if abs(row_select[1] - row_select[0]) < 10 or \
            abs(col_select[1] - col_select[0]) < 10:
            continue
        background, img_target = generate_random_mask(
            background, 
            img_target,
            texture, 
            target,
            (col_select[0], row_select[0], col_select[1], row_select[1]), 
            0.1, 0.05,
            num_points = num_rand_points
        )
    
    return background, img_target