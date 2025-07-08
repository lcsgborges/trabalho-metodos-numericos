# ARQUIVO PRINCIPAL PARA RESOLUÇÃO DO TRABALHO (PROBLEMA 1):

import math

import matplotlib.pyplot as plt 


class Metodos:
    
    def __init__(self):
        self.C = 0.041**(-1)
        self.h = 0.01
        self.xy = []
    def fun(self, x, y)->int:
        result = (self.C * (math.sqrt(1 + y**2) ) )  * self.h
        return result


    def calculate_y(self , x , y )->int:
        k1 = self.fun(x , y)
        k2 = self.fun(x+ self.h/2, y + k1/2)
        k3 = self.fun(x + self.h/2 , y + k2/2)
        k4 = self.fun(x + self.h/2 , y + k3/2)
        y_ = y + ((self.h/ 6) *(k1 + 2*k2 + 2*k3 + k4))
        return y_
    
    def runge_kutta(self,passo:float ,  x0 , y0 , x_final , parada):
        
        y_ = y0
        
        while True:
            y_ = self.calculate_y(x0, y_)

            x0 = x0+self.h

            if abs(x_final - x0) <  parada:
                break




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
#result.plot_graph()


