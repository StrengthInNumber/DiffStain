import extract
import extract_utils
import os

crf_params = {
    "w1": 600, 
    "alpha": 10000, 
    "beta": 13, 
    "w2": 3, 
    "gamma": 3, 
    "it": 5.0
}
os.remove('./output/03-15_21-15C01/crfs/r03c01f01p01_C01.png')
extract.extract_crf('./output/03-15_21-15C01/ori_images/', './output/03-15_21-15C01/ori_segs/', './output/03-15_21-15C01/crfs/',
                    crf_params=crf_params)