import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/TQC/ML/ex/AEY042500Python3.x 機器學習範例檔/ch0Python與機器學習/範例/Pokemon_894_12.csv",header=0)
# print(df.head())
# print(df.info())
# print(df.loc[:,"HP":"SpecialDef"].describe())
# print("HP平均值:",df.loc[:,"HP"].mean())
# print("HP中位數:",df.loc[:,"HP"].median())
# print("HP眾樹:",df.loc[:,"HP"].mode())
# print("HP最大值:",df.loc[:,"HP"].max())
# print("HP最小值:",df.loc[:,"HP"].min())
# print("HP第一四分位:",df.loc[:,"HP"].quantile(q=0.25))
# print("HP第三四分位:",df.loc[:,"HP"].describe()[6])
# print("相關係數:\n",np.corrcoef(df["HP"],df["Defense"]))

# plt.scatter(df["HP"], df["Defense"])
# sns.scatterplot(df["HP"], df["Defense"])

# cmap=sns.color_palette("muted",n_colors=7)
# sns.scatterplot(x=df['HP'],y=df["Defense"],
#                 data="df",
#                 hue=df['Generation'],style=df['Generation'],palette=cmap)

# g=sns.pairplot(df.loc[:,"HP":"SpecialDef"])
# g.map_lower(sns.kdeplot,levels=4,color='.2')


