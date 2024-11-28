from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

import pandas as pd
def minmaxnormalize(dataframe):
    normalizer = MinMaxScaler()
    dataframe_normalized = pd.DataFrame(normalizer.fit_transform(dataframe), columns=dataframe.columns, index=dataframe.index)
    return dataframe_normalized

def l2normalize(dataframe):
    dataframe_normalized = pd.DataFrame(normalize(dataframe, norm='l2'), columns=dataframe.columns, index=dataframe.index)
    return dataframe_normalized