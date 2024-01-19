"""
Background stitching Problem
The goal of this task is to stitch two images and deliminate the foreground.

Do NOT modify the code provided to you.
Only add your code inside the function (including newly imported packages)
You can design a new function and call the new function in the given functions. 
Please complete all the functions that are labeled with '#to do'. 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    # TODO: implement this function.
    def U(p_1 , p_2):
        return np.linalg.norm(p_1 - p_2)
    
    def W(d_1 , d_2 , k =2):
        if len(d_1) > len(d_2): 
            T, Q = d_1 ,d_2
        else:
            Q, T = d_1 , d_2

        dist = np.linalg.norm(T[: , np.newaxis] - Q, axis=2)
        k_i = np.argpartition(dist , k)[:,:k]
        g_p = []
        for Idx , ind in enumerate(k_i):
            g_1 = [(dist[Idx, idx],idx ) for idx in ind]
            g_p.append((sorted(g_1, key = lambda x:x[0] ), Idx))
        return g_p
    
    img1_g = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kpts_1,feats_1 = sift.detectAndCompute(img1_g, None)
    kpts_2,feats_2 = sift.detectAndCompute(img2_g, None)
    m_1 = W(feats_1 , feats_2)
    g_p = []
    for ut in m_1: 
        m_2 = ut[0]
        if m_2[0][0] < 0.6 * m_2[1][0]:
            g_p.append(ut)
    
    def c_b(img): 
        g_1 = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        _ , T = cv2.threshold(g_1 ,1 , 255 , cv2.THRESH_BINARY)
        c , _ = cv2.findContours(T , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        x , y, w ,h = cv2.boundingRect(c[0])
        for it in c:
            (x_new , y_new , w_new , h_new) = cv2.boundingRect(it)
            x = min(x , x_new)
            y = min(y , y_new)
            w = max(x + w , x_new + w_new) - x
            h = max(y + h , y_new + h_new) - y

        c_m = img[y:y+h , x:x+w]

        return c_m

    i_d_s = len(feats_1) < len(feats_2)
    s_1 = [kpts_1[m[0][0][1] if i_d_s else m[1]].pt for m in g_p]
    d_1 = [kpts_2[m[1] if i_d_s else m[0][0][1]].pt for m in g_p]
    s_2 = np.float32(s_1).reshape(-1,1,2)
    d_2 = np.float32(d_1).reshape(-1,1,2)
    H, mask = cv2.findHomography(s_2 , d_2 , cv2.RANSAC, 5.0)
    P = np.dot(H, [0,0,1]) / np.dot(H, [0,0,1])[-1]
    offset_x = int(np.ceil(-P[0])) if P[0] < 0 else 0
    offset_y = 100
    H[0][-1] = P[0] + offset_x
    H[1][-1] = P[1] + offset_y
    image_w = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img1.shape[0] + img2.shape[0]))
    output_img = image_w.copy()
    output_img[offset_y:img2.shape[0] + offset_y, offset_x:img2.shape[1] + offset_x] = img2
    for b in range(250, 700):
        for n in range(350 , 700):
            if image_w[b, n, 0] != 0 and image_w[b ,n ,1] != 0 and image_w[b ,n ,2] != 0:
                output_img[b ,n,:] = image_w[b ,n, :]

    final_img = c_b(output_img)
    if savepath: 
        cv2.imwrite(savepath , final_img)

if __name__ == "__main__":
    img1 = cv2.imread('./images/t3_1.png')
    img2 = cv2.imread('./images/t3_2.png')
    savepath = 't3.png'
    stitch_background(img1, img2, savepath=savepath)

