from PIL import ImageFont, ImageDraw, Image
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean

from dataProcessLib import doPCAWeight
from videoLib import collectData


def visualize_dtw(features1, features2, distance, path, ax, studentName):
    # 計算x軸和y軸的實際距離
    x_distance = len(features2)
    y_distance = len(features1)
    
    im = ax.imshow(np.ones((len(features1), len(features2))), cmap='gray_r', origin='lower')
    
    for i, j in path:
        ax.plot(j, i, 'bo')  # 'bo' indicates blue dots
    
    ax.set_title(f"{studentName} - DTW Alignment Path with Distance: {distance:.2f}")
    ax.set_xlabel(f"Student Kata Sequence (Distance: {x_distance})")
    ax.set_ylabel(f"Instructor Kata Sequence (Distance: {y_distance})")
    
    plt.colorbar(im, ax=ax)    
    
def cv2AddTwText(img,text,position,textColor=(0,255,0),textSize=30):
    if(isinstance(img,np.ndarray)):
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(img)
    fontStyle=ImageFont.truetype('msjh.ttc',textSize,encoding="utf-8")
    draw.text(position,text,textColor,font=fontStyle)
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)


def run():
    show_demo = 'y' #input("是否顯示教練影片？(y/n): ")

    bShowDemo = True if show_demo.lower() == 'y' else False

    if bShowDemo:
        videoFile='c:\\video\\kata\TT1.mp4'
        teacherData,teacherFrameTimestamps=collectData(videoFile)
        np.save('c:\\video\\kata\\teacher_data.npy',teacherData )   

    teacherData=np.load('c:\\video\\kata\\teacher_data.npy')    

    videoFile='c:\\video\\kata\\SD.mp4'
    studentData,studentFrameTimestamps =collectData(videoFile)


    fig, (ax1) = plt.subplots(1, figsize=(10, 6))

    feature_weights=doPCAWeight(teacherData)
    weighted_teacherData = teacherData * feature_weights
    weighted_studentData = studentData * feature_weights

    distance, path = fastdtw(weighted_teacherData, weighted_studentData, dist=scipy_euclidean)

    visualize_dtw(weighted_teacherData, weighted_studentData, distance, path,ax1,'Student F')

    plt.show()


if __name__ == "__main__":
    run()

