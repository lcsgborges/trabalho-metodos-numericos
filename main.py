# ARQUIVO PRINCIPAL PARA RESOLUÇÃO DO TRABALHO (PROBLEMA 1):

import math

import matplotlib.pyplot as plt 


class Metodos:
    
    def __init__(self):
        self.C = 0.041**(-1)
        self.h = 0
        self.k1 = 0
        self.k2 = 0 
        self.k3 = 0 
        self.k4 = 0 
        self.xy = []

    def k1_calculate(self , x , y):
        self.k1 = (self.C * (math.sqrt(1 + y**2) ) )  * self.h
    def k2_calculate(self, x , y):
        self.k2 =  (self.C * (math.sqrt(1 + (y+self.k1/2 )**2) ) )  * self.h
    def k3_calculate(self, x , y):
        self.k3 =  (self.C * (math.sqrt(1 + (y+self.k2/2 )**2) ) )  * self.h
    def k4_calculate(self, x , y):
        self.k4 =  (self.C * (math.sqrt(1 + (y+self.k3/2 )**2) ) )  * self.h

    def calculate_y(self , x , y )->int:
        self.k1_calculate(x , y)
        self.k2_calculate(x , y)
        self.k3_calculate(x , y)
        self.k4_calculate(x , y)

        return y + ((self.h/ 6) *(self.k1 + 2*self.k2 + 2*self.k3 + self.k4))
    
    def runge_kutta(self,passo:float ,  x0 , y0 , x_final , parada):
        self.h = passo
        self.xy.append({"x0": x0 , "y0":y0})

        yn = self.calculate_y(x=x0 , y=y0)
        xn = 0
        i = 0 
        self.xy.append({f"x{i}": xn , f"y{i}": yn})
        while xn + self.h/2 < x_final:
            i+= 1
            yn = self.calculate_y(x=xn, y=yn)
            xn += self.h
            self.xy.append({f"x{i}": xn , f"y{i}": yn})

        return { "xn":xn , "yn": yn}      
    def plot_graph(self):
        x_vals = []
        y_vals = []

        for ponto in self.xy:
            for k, v in ponto.items():
                if k.startswith("x"):
                    x_vals.append(v)
                elif k.startswith("y"):
                    y_vals.append(v)

        plt.plot(x_vals, y_vals, label="Solução RK4")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Solução da EDO com Runge-Kutta 4ª ordem")
        plt.grid(True)
        plt.legend()
        plt.show()



result =  Metodos()

result.runge_kutta(passo=0.01 , x0=0 , y0=15 , x_final=20 , parada=  0.00001)
result.plot_graph()


