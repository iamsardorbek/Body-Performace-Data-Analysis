from builtins import print

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
import cv2
import PyQt5
from pandas.conftest import axis

if __name__ == '__main__':
    data = pd.read_csv("bodyPerformanceDataset.csv")
    # 1. understand the data
    # print(data.head())
    # print(data.tail())
    # print(data.shape)
    # print(data.describe())
    # print(data.columns)
    # print(data.nunique())
    # print(data['class'].unique())
    # print(data['class'].unique())

    # 2. clean the data
    # rm a column
    # modified_data = data.drop(['class'], axis=1)
    # print(modified_data.head())

    # 3. relationship analysis
    correlation = data.corr()
    # print(correlation)
    # sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
    # plt.savefig("heatmap.png") # save to a file
    # sns.pairplot(data)
    # plt.savefig("pairplot.png") # save to a file
    # sns.relplot(x='height_cm', y = 'weight_kg', hue='gender', data=data)
    # plt.savefig("relplot.png")  # save to a file
    sns.histplot(data['sit-ups counts'])
    plt.savefig("histplot.png")  # save to a file
    sns.catplot(x = 'sit-ups counts', kind='box', data=data)
    plt.savefig("catplot.png")  # save to a file
    # plt.show()
    # image = cv2.imread('plot.png')
    # cv2.imshow('plot', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
