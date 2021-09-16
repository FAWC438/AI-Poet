# coding: utf-8
import jieba.analyse
import numpy as np
from os import path
from scipy.misc import imread
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

if __name__ == "__main__":

    mpl.rcParams['font.sans-serif'] = ['FangSong']
    #mpl.rcParams['axes.unicode_minus'] = False

    file = open("./dataset/poetrySong/piggy.txt", "rb").read()

    # tags extraction based on TF-IDF algorithm
    tags = jieba.analyse.extract_tags(file, topK=100, withWeight=False)
    text = " ".join(tags)
    # read the mask
    d = path.dirname(__file__)
    trump_coloring = imread(path.join(d, "./dataset/img.jpg"))
    background_image = np.array(Image.open("./dataset/img.jpg"))

    wc = WordCloud(font_path='/data/simsun.ttc',  # date文件夹在词频作诗中
                   background_color="white", mask=background_image,
                   )
    # background_color="black", max_words=3000, max_font_size=200, random_state=42

    # generate word cloud
    wc.generate(text)

    # generate color from image
    image_colors = ImageColorGenerator(background_image)

    plt.imshow(wc)
    plt.axis("off")
    plt.show()
