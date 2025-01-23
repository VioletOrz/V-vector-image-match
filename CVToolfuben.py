import cv2
from cnocr import CnOcr
import numpy as np
class CVTool:
    def __init__(self):
        self.ocrV2 = CnOcr(det_model_fp='./models/det_1.onnx',
                           rec_model_fp='./models/cnocr-v2.3-densenet_lite_666-gru_large-epoch=004-ft-model.onnx',
                           context='cpu')
        # self.ocrV2 = CnOcr()
        # self.ocrV2 = CnOcr(rec_model_name='densenet_lite_136-fc',
        #                    det_model_name='ch_PP-OCRv3_det',
        #                    rec_root=r'cnocr', det_root=r'cnstd')
        # # self.ocr = CnOcr(det_model_name='naive_det')
        # self.ocr = CnOcr(rec_model_name='densenet_lite_136-fc',det_model_name='naive_det', rec_root='cnocr')

    def ocr_det(self, img):
        [x,y] = img.shape[0:2]
        txt = self.ocrV2.ocr(img, resized_shape=[x,y],)
        reslut = ''
        for line in txt:
            if line['score'] > 0.1:
                reslut = reslut + line['text']
        return reslut


    def ocr_all(self,img,resized_shape = (768, 768)):
        [x, y] = img.shape[0:2]
        txt=self.ocrV2.ocr(img,resized_shape = [x,y])
        reslut =''
        for line in txt:
            reslut = reslut+line['text']

        return reslut

    def ocr_all_lol(self, img, resized_shape=(400, 400)):
        txt = self.ocrV2.ocr(img, resized_shape=img.shape[0:2])

        reslut = ''
        for line in txt:
            if line['score'] > 0.1:
                reslut = reslut + line['text'] + '____'

        return reslut
    #  Detect_area(s_x,s_y,e_x,e_y)
    #  ispercent 是不是百分比截图.
    def cut_img(self,img,Detect_area=(0,0,0,0),ispercent=True):
        if ispercent:
            [h, w, _] = img.shape
            img = img[int(h * Detect_area[1]):int(h * Detect_area[3]),
                  int(w * Detect_area[0]): int(w * Detect_area[2]), :]
            return img
        else:
            s_x, s_y, e_x, e_y = Detect_area
            img = img[s_y:e_y, s_x:e_x, :]
            return img
    def cut_img1(self,img,Detect_area=(0,0,0,0),ispercent=True):
        if ispercent:
            [h, w, _] = img.shape
            img = img[int(h * Detect_area[1]):int(h * Detect_area[3]),
                  int(w * Detect_area[0]): int(w * Detect_area[2]), :]
            return img
        else:
            s_x, s_y, e_x, e_y = Detect_area
            img = img[s_y:e_y, s_x:e_x, :]
            return img
    def cutmixup(self,mat):
        # 从mat图像中剪出来4个子图像 在concat
        [h, w, _] = mat.shape
        part1 = mat[int(h * 0.2):int(h * 0.4), int(w * 0.2):int(w * 0.4), :]
        part2 = mat[int(h * 0.6):int(h * 0.8), int(w * 0.2):int(w * 0.4), :]
        part3 = mat[int(h * 0.2):int(h * 0.4), int(w * 0.6):int(w * 0.8), :]
        part4 = mat[int(h * 0.6):int(h * 0.8), int(w * 0.6):int(w * 0.8), :]
        mat_concat = cv2.hconcat([part1, part2, part3, part4])
        return mat_concat

    # def screenshot(self, region=(0, 0, pyautogui.size()[0], pyautogui.size()[1])):
    #     img = pyautogui.screenshot(region=region)
    #     img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    #     # try:
    #     #     img = pyautogui.screenshot(region=region)
    #     #     img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    #     # except:
    #     #     print(" screen grab failed")
    #     return img

    def get_hist(self, img, channel=0):
        hist = cv2.calcHist([img], [channel], mask=None, histSize=[256], ranges=[0, 256])
        return hist

    def compareHist(self, hist1, hist2):
        k = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return 1 - k
    
    def compareHist2(self, hist1, hist2):
        k = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        return 1 - k
    

    def compareHist3(self, hist1, hist2):
        k = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 1 - k
    
    def compareHist4(self, hist1, hist2):
        k = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        return 1 - k
    
    def compareHist1(self, hists1, hists2):
        similarities = []
        for hist1, hist2 in zip(hists1, hists2):
            k = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarities.append(1 - k)
        return np.mean(similarities)  # 返回平均相似度
    
    def get_hist1(self, img, channels=[0,1,2]):
        if isinstance(channels, int):
            channels = [channels]
        hists = []
        for channel in channels:
            hist = cv2.calcHist([img], [channel], mask=None, histSize=[256], ranges=[0, 256])
            hists.append(hist)
        return hists

    def step_down_sample(self,img,step):
        return img[::step,::step,:]
    