import numpy as np

class Param():

    def __init__(self, nk, Dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe):
        self.nk = nk
        self.Dk = Dk
        self.D1 = D1
        self.D2 = D2
        self.alfa_0 = alfa_0
        self.lam_fe = lam_fe
        self.l = l
        self.Q = Q
        self.t = t
        self.alt = alt
        self.material_properties()

    def material_properties(self):
        # VentCalc Properties
        self.Pressure = 101325 * \
            (((273.15 + self.t) - 0.0065 * self.alt) / (273.15 + self.t)) ** 5.2559
        self.Density = 1.276 / (1 + 0.00366 * self.t) * \
            self.Pressure / 101325
        self.Viscosity = 9.81 * (1.478 * 10 ** -7 * (273.15 + self.t)
                                 ** 0.5) / (1 + 110.4 / (273.15 + self.t)) / self.Density

        # TempCalc Properties
        self.cv = (0.0116 * self.t ** 2 - 4.5615 * self.t
                   + 1299.7) * (self.Pressure / 101325)
        self.lam = 0.0243 * (1 + 0.00306 * self.t)
        self.k_alf = 0.022 * self.cv * self.Viscosity ** 0.2 \
            * (self.cv * self.Viscosity / self.lam) ** (-2 / 3)
            
class VentCalc():

    def r_cont(self, D1, D2, nk, dk, dens):

        S1 = np.pi/4*(D2**2 - D1**2)
        S2 = np.pi/4 * (dk**2 * nk)

        mi = 12.174 * (S2/S1) ** 6 - 36.685 * (S2/S1) ** 5 + 44.366 * (S2/S1) ** 4 - \
            27.069 * (S2/S1) ** 3 + 8.7337 * \
            (S2/S1) ** 2 - 1.2192*(S2/S1) + 0.6797

        if mi > 1:
            mi = 1

        Ksi_cont = (1 / mi - 1) ** 2
        K_cont = 0.5 * dens * (Ksi_cont + 1 - (S2 / S1) ** 2) / S2 ** 2

        return K_cont

    def r_duct(self, dk, nk, l, Re, dens):
        S = np.pi/4 * (dk**2 * nk)
        e = 1.2e-04 / dk  # 1e-04
        f = 0.02
        it = 0
        F_ = (1 / (1.74 - 2 * np.log10(2 * e + 18.7 / (Re * np.sqrt(f))))) ** 2
        while (abs(F_ - f) > 0.0001):
            f = F_
            F_ = (1 / (1.74 - 2 * np.log10(2 * e + 18.7 / (Re * np.sqrt(f))))) ** 2

            if it >= 100:
                break
            else:
                it += 1

        coeff_duct = f * l / dk
        K_duct = 0.5 * dens * coeff_duct / S ** 2

        return K_duct

    def r_exp(self, D1, D2, nk, dk, dens):
        S1 = np.pi/4 * (dk ** 2 * nk)
        S2 = np.pi/4*(D1 ** 2 - D2 ** 2)
        K_exp = 0.5 * dens * (1 / S1 - 1 / S2) ** 2

        return K_exp

    def frict(self, Re, dk, e):
        K = e / dk
        L = 0.04
        L2 = (1 / (1.74 - 2 * np.log10(2 * K + 18.7 / (Re * L ** (0.5))))) ** 2
        while (abs(L2 - L) > 0.0001):
            L = L2
            L2 = (1 / (1.74 - 2 * np.log10(2 * K + 18.7 / (Re * L ** (0.5))))) ** 2

        return L

class TempCalc():

    def g_cond(self, lam, L, S):
        G = (lam*S) / L
        return G

    def g_conv(self, alfa, S):
        G = alfa * S
        return G

    def heat_tranfer(self, dk, l, v, fl, fe, Kalf):
        Kl = 1+(dk/l)**0.67
        alfa_duct = Kalf * v**0.8 * dk**(-0.2) * Kl * np.sqrt(fe/fl)

        return alfa_duct
    

def calc(nk, dk, Ds):
    par = Param(nk, dk, 0.297, 0.4, 0.23, 0.3, 30, 325, 16, 29)
    Pl = 5000  # Power loss
    V = VentCalc()
    T = TempCalc()
    # init Values
    Dp_ = 100
    Dp = 0
    Vs = 0.1
  
    # Geom constrain
    duct_gap = 0.005
    D1_gap = 0.002
    D2_gap = 0.003

    const_gap = nk * (dk + duct_gap) / np.pi
    const_D2 = Ds + dk/2
    const_D1 = Ds - dk/2

    # Vent calculation
    while abs(Dp_ - Dp) > 1:
        Dp_ = Dp

        K_cont = V.r_cont(par.D1, par.D2, par.nk, par.Dk, par.Density)
        Re = Vs * par.Dk / par.Viscosity
        K_duct = V.r_duct(par.Dk, par.nk, par.l, Re, par.Density)
        K_exp = V.r_exp(par.Dk, par.nk, par.l, Re, par.Density)
        Ks = K_cont + K_duct + K_exp

        Dp = Ks * par.Q ** 2  # Pressure drop
        Vs = np.sqrt(Dp / Ks) / (np.pi / 4 * par.Dk **
                                 2 * par.nk)  # Mean velocity
    #Temp calculation

    # Heat transfer coefficient calculation for duct
    fl = V.frict(Re, par.Dk, 0.00001)
    fe = V.frict(Re, par.Dk, 0.0001)
    alfa_duct = T.heat_tranfer(par.Dk, par.l, Vs, fl, fe, par.k_alf)

    # Geom for stator radial conductivity
    h1 = (Ds - par.D1)/2
    h2 = (par.D2 - Ds) / 2

    Sgp1 = np.pi * (Ds+par.D1)/2 * par.l * par.lam_fe
    Sgp2 = np.pi * (Ds+par.D2)/2 * par.l * par.lam_fe
    Sgd = np.pi * par.Dk * par.nk * par.l
    S0 = np.pi * par.D2 * par.l

    Gp1 = T.g_cond(par.lam_fe, h1, Sgp1)
    Gp2 = T.g_cond(par.lam_fe, h2, Sgp2)
    Gd = T.g_conv(alfa_duct, Sgd)
    G0 = T.g_conv(par.alfa_0, S0)

    Gm = np.array([[Gp1,   -Gp1,   0],
                   [-Gp1, Gp1+Gp2+Gd, -Gp2],
                   [0,   -Gp2,  Gp2+G0],
                   ])
    Pm = np.array([Pl, Gd*30, G0*30])
    T = np.linalg.solve(Gm, Pm)
    T_av = np.mean(T)
    
    # Solid surface
    S_dct =  np.pi/4 * (dk**2 * nk)
    S_solid =   np.pi/4*(par.D2**2 - par.D1**2)
    S_stator =  S_solid - S_dct
 
    # # Feasibility check
    #
    # const_1 = const_gap - Ds
    # const_2 = (par.D1 + 2 * D1_gap) - const_D1
    # const_3 = const_D2 - (par.D2 - 2 * D2_gap)
    #
    # if const_1 < 0 and const_2 < 0 and const_3 < 0:
    #     pass
    # else:
    #     raise Exception("Infeasible")

    g1 = - (const_gap - Ds)
    g2 = - ((par.D1 + 2 * D1_gap) - const_D1)
    g3 = - ((const_D2 - (par.D2 - 2 * D2_gap)))

    return [Dp, T_av, S_stator], [g1, g2, g3]
