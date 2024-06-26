import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variáveis de entrada
valor_imovel = ctrl.Antecedent(np.arange(0, 1000001, 1), 'valor_imovel')
receita_cliente = ctrl.Antecedent(np.arange(0, 20001, 1), 'receita_cliente')
localizacao = ctrl.Antecedent(np.arange(0, 11, 1), 'localizacao')

# Variável de saída
classificacao = ctrl.Consequent(np.arange(0, 11, 1), 'classificacao')

# Funções de pertinência para valor do imóvel
valor_imovel['baixo'] = fuzz.trimf(valor_imovel.universe, [0, 0, 300000])
valor_imovel['medio'] = fuzz.trapmf(valor_imovel.universe, [200000, 400000, 600000, 800000])
valor_imovel['alto'] = fuzz.trimf(valor_imovel.universe, [600000, 1000000, 1000000])

# Funções de pertinência para receita do cliente
receita_cliente['baixa'] = fuzz.trimf(receita_cliente.universe, [0, 0, 5000])
receita_cliente['media'] = fuzz.trapmf(receita_cliente.universe, [3000, 6000, 10000, 15000])
receita_cliente['alta'] = fuzz.trimf(receita_cliente.universe, [10000, 20000, 20000])

# Funções de pertinência para localização
localizacao['ruim'] = fuzz.trimf(localizacao.universe, [0, 0, 3])
localizacao['regular'] = fuzz.trapmf(localizacao.universe, [2, 4, 6, 8])
localizacao['boa'] = fuzz.trimf(localizacao.universe, [6, 10, 10])

# Funções de pertinência para classificação
classificacao['baixa'] = fuzz.trimf(classificacao.universe, [0, 0, 4])
classificacao['media'] = fuzz.trapmf(classificacao.universe, [2, 4, 6, 8])
classificacao['alta'] = fuzz.trimf(classificacao.universe, [6, 10, 10])

# Regras
rule1 = ctrl.Rule(valor_imovel['baixo'] & receita_cliente['baixa'] & localizacao['ruim'], classificacao['baixa'])
rule2 = ctrl.Rule(valor_imovel['medio'] & receita_cliente['media'] & localizacao['regular'], classificacao['media'])
rule3 = ctrl.Rule(valor_imovel['alto'] & receita_cliente['alta'] & localizacao['boa'], classificacao['alta'])
rule4 = ctrl.Rule(valor_imovel['baixo'] & receita_cliente['media'] & localizacao['regular'], classificacao['media'])
rule5 = ctrl.Rule(valor_imovel['medio'] & receita_cliente['alta'] & localizacao['boa'], classificacao['alta'])

# Sistema de controle
classificacao_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
classificacao_simulador = ctrl.ControlSystemSimulation(classificacao_ctrl)

# Entrada de dados
classificacao_simulador.input['valor_imovel'] = 450000
classificacao_simulador.input['receita_cliente'] = 12000
classificacao_simulador.input['localizacao'] = 7

# Computa a saída
classificacao_simulador.compute()

print(classificacao_simulador.output['classificacao'])

# Plotando os gráficos (opcional)
valor_imovel.view()
receita_cliente.view()
localizacao.view()
classificacao.view(sim=classificacao_simulador)

enter = input("Enter to exit:")