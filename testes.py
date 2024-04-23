import pandas as pd

# Criando os dados para as planilhas
data1 = {'Nome': ['Alice', 'Bob', 'Charlie'],
         'Idade': [25, 30, 35]}
data2 = {'Produto': ['Maçã', 'Banana', 'Laranja'],
         'Quantidade': [10, 20, 15]}

# Criando um DataFrame para cada conjunto de dados
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Criando um objeto ExcelWriter
writer = pd.ExcelWriter('dados.xlsx', engine='xlsxwriter')

# Escrevendo os DataFrames em planilhas diferentes
df1.to_excel(writer, sheet_name='Pessoas', index=False)
df2.to_excel(writer, sheet_name='Produtos', index=False)

# Salvando o arquivo Excel
writer.save()

print("Arquivo Excel criado com sucesso!")
