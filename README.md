# Trabalho de Métodos Numéricos - Cabo Flexível Suspenso

## Descrição do Problema

Este projeto resolve numericamente o problema de um cabo flexível suspenso entre dois pontos, descrito pela equação diferencial:

$$\frac{d^{2}y}{dx^{2}}=C\sqrt{1+(\frac{dy}{dx})^{2}}$$

**Parâmetros:**
- Constante C = 0.041 m⁻¹
- Condições de contorno: y(0) = 15 m, y(20) = 10 m
- Método: Runge-Kutta de 4ª ordem com h = 0.01
- Tolerância: 1×10⁻⁵

## Como Executar
```bash
git clone git@github.com:lcsgborges/trabalho-metodos-numericos.git
cd trabalho-metodos-numericos
```

> Crie um ambiente virtual antes 

### 1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

### 2. **Executar o programa:**
```bash
python src/cabo_flexivel.py
```

### 3. **Resultados:**
- **Gráficos** salvos em `results/images/`
- **Dados** salvos em `results/data/`

### Alunos

- DANIEL FERNANDES SILVA (222008459)
- LUCAS GUIMARÃES BORGES (222015159)
- SAMUEL SUCENA DE MORAES (222006436)
