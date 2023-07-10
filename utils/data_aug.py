import math
import torch

def get_affine(params):
    # construct affine operator
    affine = torch.zeros(len(params), 2, 3)

    aspect_ratio = 1
    for i, (dy,dx,alpha,scale,flip) in enumerate(params):
        # R inverse
        sin = math.sin(alpha * math.pi / 180.)
        cos = math.cos(alpha * math.pi / 180.)

        # inverse, note how flipping is incorporated
        affine[i,0,0], affine[i,0,1] = flip * cos, sin * aspect_ratio
        affine[i,1,0], affine[i,1,1] = -sin / aspect_ratio, cos

        # T inverse Rinv * t == R^T * t
        affine[i,0,2] = -1. * (cos * dx + sin * dy)
        affine[i,1,2] = -1. * (-sin * dx + cos * dy)

        # T
        affine[i,0,2] /= float(768 // 2)
        affine[i,1,2] /= float(768 // 2)

        # scaling
        affine[i] *= scale

    return affine

def get_affine_inv(affine, params):

    aspect_ratio = 1

    affine_inv = affine.clone()
    affine_inv[:,0,1] = affine[:,1,0] * aspect_ratio**2
    affine_inv[:,1,0] = affine[:,0,1] / aspect_ratio**2
    affine_inv[:,0,2] = -1 * (affine_inv[:,0,0] * affine[:,0,2] + affine_inv[:,0,1] * affine[:,1,2])
    affine_inv[:,1,2] = -1 * (affine_inv[:,1,0] * affine[:,0,2] + affine_inv[:,1,1] * affine[:,1,2])

    # scaling
    affine_inv /= torch.Tensor(params)[:, 3].view(-1,1,1)**2

    return affine_inv
