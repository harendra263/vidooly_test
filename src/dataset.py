import pandas as pd
import numpy as np


import pandas as pd

train = pd.read_csv("https://raw.githubusercontent.com/harendra263/hiringtask/master/machine_learning/ad_org/data/mn/ad_org_train.csv")

test = pd.read_csv("https://raw.githubusercontent.com/harendra263/hiringtask/master/machine_learning/ad_org/data/mn/ad_org_test.csv")

train.to_csv("input/train.csv", index=False)
test.to_csv("input/test.csv", index=False)
