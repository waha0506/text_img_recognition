import cv2
import numpy as np
import sys
import argparse
import pytesseract
from PIL import Image
import threading

text_list = { }

ap=argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str)
ap.add_argument('-o', '--output', type = str)
ap.add_argument('-c', '--coordinate', type=str)
ap.add_argument('-p', '--padding', type=str)
ap.add_argument('-m', '--contrast', nargs='?', default=False)
ap.add_argument('-a', '--alpha', nargs='?', type=float)
ap.add_argument('-b', '--beta', nargs='?', type=float)
ap.add_argument('-r', '--ocr', nargs='?', default=True)
args=vars(ap.parse_args())

def text_ocr(image_name, coordinate_output, ocr_output, padding, f):
    global text_list
    orig_img = cv2.imread(image_name)
    if args['contrast']:
        orig_img=cv2.convertScaleAbs(orig_img, alpha=args['alpha'], beta=args['beta'])
    else:
        pass
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('0', gray)

    ret, bw_gray = cv2.threshold(gray.copy(),127,255,cv2.THRESH_BINARY)
    #cv2.imshow('1', bw_gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    #cv2.imshow('2', grad)

    _, bw = cv2.threshold(grad, 0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #_, bw = cv2.threshold(grad, 64, 255.0, cv2.THRESH_BINARY)
    #_, bw = cv2.threshold(grad, 50, 255.0, cv2.THRESH_BINARY)

    #cv2.imshow('3', bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('4', connected)

    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(orig_img, contours, -1, (0,255,0), 3)
    #cv2.imshow('contours', orig_img)
    #print(len(contours))
    mask = np.zeros(bw.shape, dtype=np.uint8)

    #cv2.imshow('5', mask)

    #f = open(coordinate_output, 'a+')
    #f.truncate(0)

    padding_cf = float(padding)
    threads = []

    for idx in range(len(contours)):
        t=threading.Thread(target=imagetostring, args=(orig_img, bw_gray, contours, mask, idx, padding_cf, f, ))
        threads.append(t)
        t.start()
        t.join()
            
    #cv2.imshow('mask', mask)
    #f.close()
    #cv2.imshow('rects', orig_img)
    cv2.imwrite(ocr_output, orig_img)
    
def imagetostring(orig_img, bw_gray, contours, mask, idx, padding_cf, f):
    x, y, w, h = cv2.boundingRect(contours[idx])
    #mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    #cv2.imshow('%d'%idx, mask)
    #r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    cv2.rectangle(orig_img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
    #print(x,y,w,h,idx)
    #if w/h > 1.5 and h > 20:
    #if w/h >=1 and h>20:
    #print(args['ocr'])
    if args['ocr']==True:
        if h>15:
            #cv2.rectangle(small_rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            #ocr_candidate = cv2.bitwise_not(bw_gray[y:y+h, x:x+w])
            ocr_candidate = cv2.bitwise_not(bw_gray[int(y*(1-padding_cf)):int((y+h)*(1+padding_cf)), int(x*(1-padding_cf)):int((x+w)*(1+padding_cf))])
            #cv2.imshow('%d'%idx, ocr_candidate)
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(ocr_candidate, config=config)
            text=text.encode('ascii', 'ignore').decode('ascii')
            text_list={'idx':idx, 'center_x':int(x+w/2),'center_y':int(y+h/2), 'top_left_x':int(x), 'top_left_y':int(y), 'width':int(w), 'height':int(h), 'ocr_result':text}
            f.write('index--%s--center_x--%s--center_y--%s--top_left_x--%s--top_left_y--%s--width--%s--height--%s--ocr_result--%s\n'%(text_list['idx'],text_list['center_x'],text_list['center_y'],text_list['top_left_x'],text_list['top_left_y'],text_list['width'],text_list['height'],text_list['ocr_result']))
            #print(text_list)
            #cv2.imshow('%d'%idx, ocr_candidate)
    else:
        if h>15:
            text='xxx'
            text=text.encode('ascii', 'ignore').decode('ascii')
            #print(idx)
            #print(text)
            #print(x,y,w,h)
            text_list={'idx':idx, 'center_x':int(x+w/2),'center_y':int(y+h/2), 'top_left_x':int(x), 'top_left_y':int(y), 'width':int(w), 'height':int(h), 'ocr_result':text}
            #print(text_list)
            f.write('index--%s--center_x--%s--center_y--%s--top_left_x--%s--top_left_y--%s--width--%s--height--%s--ocr_result--%s\n'%(text_list['idx'],text_list['center_x'],text_list['center_y'],text_list['top_left_x'],text_list['top_left_y'],text_list['width'],text_list['height'],text_list['ocr_result']))
            #print(text_list)
            #cv2.imshow('%d'%idx, ocr_candidate)

def main():
    f = open(args['coordinate'], 'a+')
    f.truncate(0)
    text_ocr(args['image'], args['coordinate'], args['output'], args['padding'], f)
    #with open(args['coordinate'], 'w') as f:
    #    for key in coordinate.keys():
    #        f.write('%s %s\n'%(key, coordinate[key]))
    f.close()
    cv2.waitKey(0)
    
main()