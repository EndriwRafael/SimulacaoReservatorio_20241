import pandas as pd
from pandas import DataFrame as df
from pandas import MultiIndex

di = {
    't1': [1, 2, 3],
    'e1': [2, 3, 1],
    't2': [1, 2, 3],
    'e2': [2, 3, 1]
}
ddd = [[2, 3, 1, 2], [2, 3, 1, 2], [2, 3, 1, 2], [2, 3, 1, 2]]
lista = [
    ['Explicit', 'Implicit'],
    ['Tempo', 'Erro']
]

columns = pd.MultiIndex.from_product(lista)
df1 = df(ddd, columns=columns)
df1.to_excel('oi.xlsx')
