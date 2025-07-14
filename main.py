import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

C = 0.041
y0 = 15
y_target = 10
h = 0.01
tol = 1e-5


# metodo do tiro
def metodo_tiro(s, return_full=False):
    u, v = y0, s  
    x = 0
    x_vals = [x]
    y_vals = [u]
    

    # runge kutta 4
    while x < 20:
        k1_u = v
        k1_v = C * np.sqrt(1 + v**2)
        
        k2_u = v + 0.5 * h * k1_v
        k2_v = C * np.sqrt(1 + (v + 0.5 * h * k1_v)**2)
        
        k3_u = v + 0.5 * h * k2_v
        k3_v = C * np.sqrt(1 + (v + 0.5 * h * k2_v)**2)
        
        k4_u = v + h * k3_v
        k4_v = C * np.sqrt(1 + (v + h * k3_v)**2)
        
        u += h * (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
        v += h * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        x += h
        
        if return_full:
            x_vals.append(x)
            y_vals.append(u)
    
    if return_full:
        return np.array(x_vals), np.array(y_vals)
    else:
        return u - y_target

def newton(s_guess=0.0):
    s = s_guess
    for _ in range(100):
        F = metodo_tiro(s)
        if abs(F) < tol:
            return s
        
        ds = 1e-6
        F_deriv = (metodo_tiro(s + ds) - F) / ds
        s -= F / F_deriv
    return s

s_final = newton(s_guess=-0.1)
x_vals, y_vals = metodo_tiro(s_final, return_full=True)

print(f"Valor ajustado de s = y'(0): {s_final:.6f}")
print(f"Erro em y(20): {abs(metodo_tiro(s_final)):.6f}")

plt.plot(x_vals, y_vals, label="Solução numérica")
plt.scatter([0, 20], [y0, y_target], color='red', label="Condições de contorno")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.legend(); plt.grid(); plt.show()


def numerical_derivatives(x, y, h):
    y_prime = np.gradient(y, h)  
    y_double_prime = np.gradient(y_prime, h)  
    return y_prime, y_double_prime

y_prime, y_double_prime = numerical_derivatives(x_vals, y_vals, h)


lhs = y_double_prime
rhs = C * np.sqrt(1 + y_prime**2)
error = np.max(np.abs(lhs - rhs))
print(f"Erro máximo na verificação da EDO: {error:.6f}")





coefficients = np.polyfit(x_vals, y_vals, deg=4)
P = np.poly1d(coefficients)


P_prime = P.deriv()
P_double_prime = P_prime.deriv()


P_y = P(x_vals)
P_y_prime = P_prime(x_vals)
P_y_double_prime = P_double_prime(x_vals)


rhs_poly = C * np.sqrt(1 + P_y_prime**2)
error_poly = np.max(np.abs(P_y_double_prime - rhs_poly))
print(f"Erro máximo na regressão polinomial: {error_poly:.6f}")


plt.plot(x_vals, y_vals, label="Solução numérica")
plt.plot(x_vals, P_y, '--', label="Regressão polinomial (4º grau)")
plt.legend(); plt.grid(); plt.show()