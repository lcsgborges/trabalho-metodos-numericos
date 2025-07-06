# ARQUIVO PRINCIPAL PARA RESOLUÇÃO DO TRABALHO (PROBLEMA 1):

import math


class Metodos:
    
    def __init__(self):
        self.C = 0.041**(-1)
        self.h = 0
        self.k1 = 0
        self.k2 = 0 
        self.k3 = 0 
        self.k4 = 0 
        self.xy = [{}]

    def k1_calculate(self , x , y):
        self.k1 = (self.C * (math.sqrt(1 + y**2) ) )  * self.h
    def k2_calculate(self, x , y):
        self.k2 =  (self.C * (math.sqrt(1 + (y+self.k1/2 )**2) ) )  * self.h
    def k3_calculate(self, x , y):
        self.k3 =  (self.C * (math.sqrt(1 + (y+self.k2/2 )**2) ) )  * self.h
    def k4_calculate(self, x , y):
        self.k4 =  (self.C * (math.sqrt(1 + (y+self.k3/2 )**2) ) )  * self.h
    
    def runge_kutta(self,passo:float ,  x0 , y0):
        
        return self.k1 , self.k2

result =  Metodos()

print(result.runge_kutta(0.01 , 0 , 15))