import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Fake Bills Data.csv', sep = ';')
X = dataset.iloc[:, :-1].values
y = dataset['is_genuine'].map({True:1, False:0}).values

