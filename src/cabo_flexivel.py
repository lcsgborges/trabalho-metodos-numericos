import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configurar matplotlib para salvar figuras
plt.switch_backend('Agg')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Configurar caminhos
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = RESULTS_DIR / "images"
DATA_DIR = RESULTS_DIR / "data"

# Criar diretórios se não existirem
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Parâmetros do problema
C = 0.041  # Constante da equação diferencial (m^-1)
y0 = 15    # Condição inicial y(0) = 15 m
y_target = 10  # Condição final y(20) = 10 m
h = 0.01   # Passo do método Runge-Kutta
tol = 1e-5 # Tolerância para o critério de parada

class CableHangingSolver:
    """Classe para resolver o problema do cabo flexível suspenso"""
    
    def __init__(self, C, y0, y_target, h, tol):
        self.C = C
        self.y0 = y0
        self.y_target = y_target
        self.h = h
        self.tol = tol
        self.solution = None
        
    def shooting_method(self, s, return_full=False):
        """
        Implementa o método do tiro para resolver a EDO
        d²y/dx² = C * sqrt(1 + (dy/dx)²)
        
        Sistema equivalente:
        du/dx = v
        dv/dx = C * sqrt(1 + v²)
        
        onde u = y e v = dy/dx
        """
        u, v = self.y0, s  # u = y, v = y'
        x = 0
        
        if return_full:
            x_vals = [x]
            y_vals = [u]
            v_vals = [v]
        
        # Integração numérica com Runge-Kutta de 4ª ordem
        while x < 20:
            # Runge-Kutta de 4ª ordem para sistema de EDOs
            k1_u = self.h * v
            k1_v = self.h * self.C * np.sqrt(1 + v**2)
            
            k2_u = self.h * (v + 0.5 * k1_v)
            k2_v = self.h * self.C * np.sqrt(1 + (v + 0.5 * k1_v)**2)
            
            k3_u = self.h * (v + 0.5 * k2_v)
            k3_v = self.h * self.C * np.sqrt(1 + (v + 0.5 * k2_v)**2)
            
            k4_u = self.h * (v + k3_v)
            k4_v = self.h * self.C * np.sqrt(1 + (v + k3_v)**2)
            
            # Atualização das variáveis
            u += (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
            v += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            x += self.h
            
            if return_full:
                x_vals.append(x)
                y_vals.append(u)
                v_vals.append(v)
        
        if return_full:
            return np.array(x_vals), np.array(y_vals), np.array(v_vals)
        else:
            return u - self.y_target

    def newton_method(self, s_guess=-0.1):
        """Método de Newton para encontrar o valor correto de s = y'(0)"""
        s = s_guess
        for iteration in range(100):
            F = self.shooting_method(s)
            if abs(F) < self.tol:
                print(f"Convergência atingida em {iteration + 1} iterações")
                return s
            
            # Aproximação numérica da derivada
            ds = 1e-6
            F_deriv = (self.shooting_method(s + ds) - F) / ds
            s -= F / F_deriv
        
        print("Aviso: Número máximo de iterações atingido")
        return s
    
    def solve(self):
        """Resolve o problema completo"""
        print("=" * 60)
        print("SOLUÇÃO DO PROBLEMA DO CABO FLEXÍVEL SUSPENSO")
        print("=" * 60)
        print(f"Parâmetros do problema:")
        print(f"- Constante C = {self.C} m^-1")
        print(f"- Condição inicial: y(0) = {self.y0} m")
        print(f"- Condição final: y(20) = {self.y_target} m")
        print(f"- Passo h = {self.h}")
        print(f"- Tolerância = {self.tol}")
        print()
        
        # Encontrar o valor inicial da derivada
        print("QUESTÃO 1: MÉTODO DO TIRO COM RUNGE-KUTTA DE 4ª ORDEM")
        print("-" * 55)
        print("Resolvendo pelo método do tiro...")
        s_final = self.newton_method()
        
        # Obter solução completa
        x_vals, y_vals, v_vals = self.shooting_method(s_final, return_full=True)
        
        # Armazenar resultados
        self.solution = {
            's_final': s_final,
            'x_vals': x_vals,
            'y_vals': y_vals,
            'v_vals': v_vals,
            'boundary_error': abs(self.shooting_method(s_final))
        }
        
        print(f"Valor ajustado de s = y'(0): {s_final:.6f}")
        print(f"Erro em y(20): {self.solution['boundary_error']:.6f}")
        print()
        
        return self.solution
    
    def verify_differential_equation(self):
        """Verifica se a solução satisfaz a EDO usando diferenciação numérica"""
        if self.solution is None:
            raise ValueError("Resolva o problema primeiro usando solve()")
        
        print("QUESTÃO 2: VERIFICAÇÃO DA EDO ATRAVÉS DE DIFERENCIAÇÃO NUMÉRICA")
        print("-" * 65)
        
        x_vals = self.solution['x_vals']
        y_vals = self.solution['y_vals']
        v_vals = self.solution['v_vals']
        
        # Usar np.gradient para diferenciação numérica
        y_prime_num = np.gradient(y_vals, self.h)
        y_double_prime_num = np.gradient(y_prime_num, self.h)
        
        # Verificação da EDO: d²y/dx² = C * sqrt(1 + (dy/dx)²)
        lhs = y_double_prime_num  # valor numérico
        rhs = self.C * np.sqrt(1 + y_prime_num**2)  # valor teórico
        
        # Erro absoluto
        error_abs = np.abs(lhs - rhs)
        
        # Erro relativo: |valor_teorico - valor_numerico| / |valor_teorico|
        error_rel = np.abs(rhs - lhs) / np.abs(rhs)
        
        results = {
            'max_error_abs': np.max(error_abs),
            'mean_error_abs': np.mean(error_abs),
            'max_error_rel': np.max(error_rel),
            'mean_error_rel': np.mean(error_rel),
            'std_error': np.std(error_abs),
            'derivative_error': np.max(np.abs(y_prime_num - v_vals))
        }
        
        print(f"Erro máximo absoluto: {results['max_error_abs']:.6f}")
        print(f"Erro médio absoluto: {results['mean_error_abs']:.6f}")
        print(f"Erro máximo relativo: {results['max_error_rel']:.6f} ({results['max_error_rel']*100:.4f}%)")
        print(f"Erro médio relativo: {results['mean_error_rel']:.6f} ({results['mean_error_rel']*100:.4f}%)")
        print(f"Desvio padrão do erro: {results['std_error']:.6f}")
        print(f"Erro máximo na 1ª derivada: {results['derivative_error']:.6f}")
        print()
        
        return results
    
    def polynomial_regression(self, degree=4):
        """Realiza regressão polinomial e verifica a EDO"""
        if self.solution is None:
            raise ValueError("Resolva o problema primeiro usando solve()")
        
        print("QUESTÃO 3: REGRESSÃO POLINOMIAL DE 4º GRAU")
        print("-" * 42)
        
        x_vals = self.solution['x_vals']
        y_vals = self.solution['y_vals']
        
        # Ajuste do polinômio
        coefficients = np.polyfit(x_vals, y_vals, deg=degree)
        P = np.poly1d(coefficients)
        
        print("Coeficientes do polinômio de 4º grau:")
        for i, coef in enumerate(coefficients):
            print(f"  a_{degree-i} = {coef:.8f}")
        print()
        
        # Derivadas do polinômio
        P_prime = P.deriv()
        P_double_prime = P_prime.deriv()
        
        # Avaliação do polinômio e suas derivadas
        P_y = P(x_vals)
        P_y_prime = P_prime(x_vals)
        P_y_double_prime = P_double_prime(x_vals)
        
        # Verificação da EDO com o polinômio
        lhs_poly = P_y_double_prime  # valor numérico
        rhs_poly = self.C * np.sqrt(1 + P_y_prime**2)  # valor teórico
        
        # Erro absoluto
        error_abs = np.abs(lhs_poly - rhs_poly)
        
        # Erro relativo: |valor_teorico - valor_numerico| / |valor_teorico|
        error_rel = np.abs(rhs_poly - lhs_poly) / np.abs(rhs_poly)
        
        results = {
            'coefficients': coefficients,
            'polynomial': P,
            'P_y': P_y,
            'P_y_prime': P_y_prime,
            'P_y_double_prime': P_y_double_prime,
            'max_error_abs': np.max(error_abs),
            'mean_error_abs': np.mean(error_abs),
            'max_error_rel': np.max(error_rel),
            'mean_error_rel': np.mean(error_rel),
            'std_error': np.std(error_abs)
        }
        
        print(f"Erro máximo absoluto: {results['max_error_abs']:.6f}")
        print(f"Erro médio absoluto: {results['mean_error_abs']:.6f}")
        print(f"Erro máximo relativo: {results['max_error_rel']:.6f} ({results['max_error_rel']*100:.4f}%)")
        print(f"Erro médio relativo: {results['mean_error_rel']:.6f} ({results['mean_error_rel']*100:.4f}%)")
        print(f"Desvio padrão do erro: {results['std_error']:.6f}")
        print()
        
        return results
    
    def generate_plots(self):
        """Gera gráficos essenciais focados nas três questões do exercício"""
        if self.solution is None:
            raise ValueError("Resolva o problema primeiro usando solve()")
        
        print("Gerando gráficos...")
        
        # Obter dados
        x_vals = self.solution['x_vals']
        y_vals = self.solution['y_vals']
        v_vals = self.solution['v_vals']
        
        # Calcular derivadas numericamente
        y_prime_num = np.gradient(y_vals, self.h)
        y_double_prime_num = np.gradient(y_prime_num, self.h)
        
        # Regressão polinomial
        coefficients = np.polyfit(x_vals, y_vals, deg=4)
        P = np.poly1d(coefficients)
        P_prime = P.deriv()
        P_double_prime = P_prime.deriv()
        P_y = P(x_vals)
        P_y_prime = P_prime(x_vals)
        P_y_double_prime = P_double_prime(x_vals)
        
        # GRÁFICO 1: Questão 1 - Forma do cabo
        plt.figure(figsize=(12, 8))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label="Solução numérica (Método do Tiro + RK4)")
        plt.scatter([0, 20], [self.y0, self.y_target], color='red', s=100, zorder=5, 
                   label=f"Condições de contorno: y(0)={self.y0}, y(20)={self.y_target}")
        plt.xlabel("x (m)", fontsize=12)
        plt.ylabel("y (m)", fontsize=12)
        plt.title("QUESTÃO 1: Forma do Cabo Flexível Suspenso", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / 'cabo_forma.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # GRÁFICO 2: Questões 2 e 3 - Análise das derivadas
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Função y(x) - Questão 1
        plt.subplot(2, 3, 1)
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label="Solução numérica")
        plt.plot(x_vals, P_y, 'r--', linewidth=2, label="Polinômio 4º grau")
        plt.scatter([0, 20], [self.y0, self.y_target], color='green', s=50, zorder=5)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("y(x) - Função Original")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Primeira derivada - Questão 2
        plt.subplot(2, 3, 2)
        plt.plot(x_vals, v_vals, 'b-', linewidth=2, label="RK4 (y')")
        plt.plot(x_vals, y_prime_num, 'g--', linewidth=2, label="Diferenciação numérica")
        plt.plot(x_vals, P_y_prime, 'r--', linewidth=2, label="Polinômio derivado")
        plt.xlabel("x (m)")
        plt.ylabel("dy/dx")
        plt.title("dy/dx - Primeira Derivada")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Segunda derivada - Questões 2 e 3
        plt.subplot(2, 3, 3)
        plt.plot(x_vals, y_double_prime_num, 'g-', linewidth=2, label="Diferenciação numérica")
        plt.plot(x_vals, P_y_double_prime, 'r--', linewidth=2, label="Polinômio 2ª derivada")
        plt.plot(x_vals, self.C * np.sqrt(1 + y_prime_num**2), 'k:', linewidth=2, 
                label=r"$C\sqrt{1+(dy/dx)^2}$ (EDO)")
        plt.xlabel("x (m)")
        plt.ylabel("d²y/dx²")
        plt.title("d²y/dx² - Segunda Derivada")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Erro da diferenciação numérica - Questão 2
        error_diff = np.abs(y_double_prime_num - self.C * np.sqrt(1 + y_prime_num**2))
        plt.subplot(2, 3, 4)
        plt.plot(x_vals, error_diff, 'g-', linewidth=2)
        plt.xlabel("x (m)")
        plt.ylabel("Erro absoluto")
        plt.title("Questão 2: Erro da Diferenciação Numérica")
        plt.grid(True, alpha=0.3)
        # Não usar log scale para evitar problemas com valores muito pequenos
        plt.ylim(0, np.max(error_diff) * 1.1)
        # Adicionar anotação sobre o erro máximo
        max_idx = np.argmax(error_diff)
        plt.annotate(f'Erro máx: {np.max(error_diff):.2e}\n(x={x_vals[max_idx]:.1f})', 
                    xy=(x_vals[max_idx], np.max(error_diff)), 
                    xytext=(x_vals[max_idx] + 2, np.max(error_diff) * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=8, ha='left')
        
        # Subplot 5: Erro da regressão polinomial - Questão 3
        error_poly = np.abs(P_y_double_prime - self.C * np.sqrt(1 + P_y_prime**2))
        plt.subplot(2, 3, 5)
        plt.plot(x_vals, error_poly, 'r-', linewidth=2)
        plt.xlabel("x (m)")
        plt.ylabel("Erro absoluto")
        plt.title("Questão 3: Erro da Regressão Polinomial")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, np.max(error_poly) * 1.1)
        # Adicionar anotação sobre o erro máximo
        max_idx_poly = np.argmax(error_poly)
        plt.annotate(f'Erro máx: {np.max(error_poly):.2e}\n(x={x_vals[max_idx_poly]:.1f})', 
                    xy=(x_vals[max_idx_poly], np.max(error_poly)), 
                    xytext=(x_vals[max_idx_poly] + 2, np.max(error_poly) * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=8, ha='left')
        
        # Subplot 6: Comparação dos erros (escala log apenas se necessário)
        plt.subplot(2, 3, 6)
        plt.plot(x_vals, error_diff, 'g-', linewidth=2, label=f"Diferenciação (máx: {np.max(error_diff):.2e})")
        plt.plot(x_vals, error_poly, 'r-', linewidth=2, label=f"Regressão (máx: {np.max(error_poly):.2e})")
        plt.xlabel("x (m)")
        plt.ylabel("Erro absoluto")
        plt.title("Comparação dos Erros")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        # Usar escala log apenas se a diferença for muito grande
        if np.max(error_diff) / np.min(error_diff[error_diff > 0]) > 100:
            plt.yscale('log')
            plt.ylabel("Erro absoluto (log)")
        else:
            plt.ylim(0, max(np.max(error_diff), np.max(error_poly)) * 1.1)
        
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / 'analise_completa.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
    def save_data(self):
        """Salva os dados da solução"""
        if self.solution is None:
            raise ValueError("Resolva o problema primeiro usando solve()")
        
        # Salvar dados principais
        data = np.column_stack([
            self.solution['x_vals'],
            self.solution['y_vals'],
            self.solution['v_vals']
        ])
        
        header = f"""Solução do Problema do Cabo Flexível Suspenso
Parâmetros: C={self.C}, y0={self.y0}, y_target={self.y_target}, h={self.h}
Valor inicial da derivada: s = {self.solution['s_final']:.6f}
Erro na condição de contorno: {self.solution['boundary_error']:.6f}
Colunas: x(m), y(m), dy/dx"""
        
        np.savetxt(DATA_DIR / 'solucao_cabo.txt', data, header=header, fmt='%.6f')
        print(f"Dados salvos em: {DATA_DIR}")

def main():
    """Função principal"""
    # Criar solver
    solver = CableHangingSolver(C, y0, y_target, h, tol)
    
    # Resolver o problema
    solution = solver.solve()
    
    # Verificar EDO
    edo_results = solver.verify_differential_equation()
    
    # Regressão polinomial
    poly_results = solver.polynomial_regression()
    
    # Gerar gráficos
    solver.generate_plots()
    
    # Salvar dados
    solver.save_data()
    
    # Criar resumo dos resultados
    with open(DATA_DIR / 'resumo_resultados.txt', 'w') as f:
        f.write("RESUMO DOS RESULTADOS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"QUESTÃO 1 - Método do Tiro:\n")
        f.write(f"   - Valor de y'(0): {solution['s_final']:.6f}\n")
        f.write(f"   - Erro na condição de contorno: {solution['boundary_error']:.6f}\n\n")
        
        f.write(f"QUESTÃO 2 - Verificação da EDO:\n")
        f.write(f"   - Erro máximo absoluto: {edo_results['max_error_abs']:.6f}\n")
        f.write(f"   - Erro médio absoluto: {edo_results['mean_error_abs']:.6f}\n")
        f.write(f"   - Erro máximo relativo: {edo_results['max_error_rel']:.6f} ({edo_results['max_error_rel']*100:.4f}%)\n")
        f.write(f"   - Erro médio relativo: {edo_results['mean_error_rel']:.6f} ({edo_results['mean_error_rel']*100:.4f}%)\n\n")
        
        f.write(f"QUESTÃO 3 - Regressão Polinomial:\n")
        f.write(f"   - Erro máximo absoluto: {poly_results['max_error_abs']:.6f}\n")
        f.write(f"   - Erro médio absoluto: {poly_results['mean_error_abs']:.6f}\n")
        f.write(f"   - Erro máximo relativo: {poly_results['max_error_rel']:.6f} ({poly_results['max_error_rel']*100:.4f}%)\n")
        f.write(f"   - Erro médio relativo: {poly_results['mean_error_rel']:.6f} ({poly_results['mean_error_rel']*100:.4f}%)\n\n")
        
        f.write("4. Coeficientes do polinômio de 4º grau:\n")
        for i, coef in enumerate(poly_results['coefficients']):
            f.write(f"   a_{4-i} = {coef:.8f}\n")
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO FINAL DOS RESULTADOS")
    print("="*60)
    print(f"✓ QUESTÃO 1: Solução encontrada: y'(0) = {solution['s_final']:.6f}")
    print(f"  - Erro na condição de contorno: {solution['boundary_error']:.6f}")
    print(f"✓ QUESTÃO 2: Erro máximo absoluto na verificação da EDO: {edo_results['max_error_abs']:.6f}")
    print(f"  - Erro máximo relativo: {edo_results['max_error_rel']:.6f} ({edo_results['max_error_rel']*100:.4f}%)")
    print(f"✓ QUESTÃO 3: Erro máximo absoluto na regressão polinomial: {poly_results['max_error_abs']:.6f}")
    print(f"  - Erro máximo relativo: {poly_results['max_error_rel']:.6f} ({poly_results['max_error_rel']*100:.4f}%)")
    print(f"✓ Gráficos salvos em: results/images/")
    print(f"✓ Dados salvos em: results/data/")
    print("="*60)

if __name__ == "__main__":
    main()
