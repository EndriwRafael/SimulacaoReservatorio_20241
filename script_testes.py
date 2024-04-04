class Objeto:
    def __init__(self, valor1, valor2):
        self.valor1 = valor1
        self.valor2 = valor2

def minha_funcao(objeto):
    # Acessando os valores do objeto
    print("Valor 1:", objeto.valor1)
    print("Valor 2:", objeto.valor2)

# Criando um objeto
meu_objeto = Objeto(10, 20)

# Chamando a função e passando o objeto como parâmetro
minha_funcao(meu_objeto)
