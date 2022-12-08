module MotorParam
using LinearSolve
using Distributions

# Constructability constraints
const D1_ = 0.374
const D2_ = 0.52
const duct_gap = 0.005
const D1_gap = 0.002
const D2_gap = 0.003

# nk, dk, Ds
domain() = [29.9, 0.018, 0.410], [60.1, 0.030, 0.520]

struct Param
    nk::Float64
    a::Float64
    dk::Float64
    D1::Float64
    D2::Float64
    alfa_0::Float64
    lam_fe::Float64
    l::Float64
    Q::Float64
    t::Float64
    alt::Float64
    Sk::Float64
    Ok::Float64
    Dh::Float64
    Pressure::Float64
    Density::Float64
    Viscosity::Float64
    cv::Float64
    lam::Float64
    k_alf::Float64

    function Param(nk, a, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe)
        Sk = a*dk + pi/4*dk^2
        Ok = 2 * a + pi*dk
        Dh = 4*Sk / Ok
        Pressure, Density, Viscosity, cv, lam, k_alf = material_properties(t, alt)
        new(nk, a, dk, D1, D2, alfa_0, lam_fe, l, Q, t, alt, Sk, Ok, Dh, Pressure, Density, Viscosity, cv, lam, k_alf)
    end 
end

function material_properties(t, alt)
    # VentCalc Properties
    Pressure = 101325 * (((273.15 + t) - 0.0065 * alt) / (273.15 + t)) ^ 5.2559
    Density = 1.276 / (1 + 0.00366 * t) * Pressure / 101325
    Viscosity = 9.81 * (1.478 * 10 ^ -7 * (273.15 + t) ^ 0.5) / (1 + 110.4 / (273.15 + t)) / Density

    # TempCalc Properties
    cv = (0.0116 * t ^ 2 - 4.5615 * t + 1299.7) * (Pressure / 101325)
    lam = 0.0243 * (1 + 0.00306 * t)
    k_alf = 0.022 * cv * Viscosity ^ 0.2 * (cv * Viscosity / lam) ^ (-2 / 3)

    return Pressure, Density, Viscosity, cv, lam, k_alf
end    

module VentCalc

    function r_cont(D1, D2, Sk, nk, dens)
        S1 = pi/4*(D2^2 - D1^2)
        S2 = Sk * nk

        mi = 12.174 * (S2/S1) ^ 6 - 36.685 * (S2/S1) ^ 5 + 44.366 * (S2/S1) ^ 4 - 27.069 * (S2/S1) ^ 3 + 8.7337 * (S2/S1) ^ 2 - 1.2192*(S2/S1) + 0.6797

        if mi > 1
            mi = 1
        end

        Ksi_cont = (1 / mi - 1) ^ 2
        K_cont = 0.5 * dens * (Ksi_cont + 1 - (S2 / S1) ^ 2) / S2 ^ 2

        return K_cont
    end

    function r_duct(Sk, Dh, nk, l, Re, dens)
        S = Sk * nk
        e = 1.2e-04 / Dh  # 1e-04
        f = 0.04 #0.02
        it = 0
        F_ = (1 / (1.74 - 2 * log10(2 * e + 18.7 / (Re * sqrt(f))))) ^ 2
        while (abs(F_ - f) > 0.0001)
            f = F_
            F_ = (1 / (1.74 - 2 * log10(2 * e + 18.7 / (Re * sqrt(f))))) ^ 2

            if it >= 100
                break
            else
                it += 1
            end
        end

        coeff_duct = f * l / Dh
        K_duct = 0.5 * dens * coeff_duct / S ^ 2

        return K_duct
    end

    function r_exp(D1, D2, nk, Sk, dens)
        S1 = Sk
        S2 = pi/4*(D1 ^ 2 - D2 ^ 2)
        K_exp = 0.5 * dens * (1 / S1 - 1 / S2) ^ 2

        return K_exp
    end

    function frict(Re, Dh, e)
        K = e / Dh
        L = 0.04
        L2 = (1 / (1.74 - 2 * log10(2 * K + 18.7 / (Re * L ^ (0.5))))) ^ 2
        while (abs(L2 - L) > 0.01) # 0.0001
            L = L2
            L2 = (1 / (1.74 - 2 * log10(2 * K + 18.7 / (Re * L ^ (0.5))))) ^ 2
        end

        return L
    end

end  # VentCalc

module TempCalc

    function g_cond(lam, L, S)
        G = (lam*S) / L
        return G
    end

    function g_conv(alfa, S)
        G = alfa * S
        return G
    end

    function heat_tranfer(nk, a, dk, Dh, cp, Q, dns, vsc, lam)

        v = Q / ( pi * (nk * Dh ^ 2) / 4 )
        Re = dns * v * (Dh/vsc)     
        f = (0.79 * log(Re) - 1.64) ^ -2
        Pr = cp * vsc/lam
        Nu = ( (f/8) * (Re - 1000) * Pr )/(1 + 12.7 * (f/8) ^ 0.5 * (Pr ^ (2/3) - 1))
        alfa_duct = Nu * lam / Dh
        # print(alfa_duct)
        
        return alfa_duct
    end

end  # TempCalc
    
# function calc(nk, a, dk, Ds, Q)
function calc(nk, dk, Ds;
    a = 0.,
    D1 = D1_,
    D2 = D2_,
    l = 0.23,
    Q = 0.5,
    t = 30,
    alt = 325,
    alfa_0 = 16,
    lam_fe = 29,
    Pl = 5000,  # Power loss
    alfa_a = 1.,
    alfa_b = 0.,
)
    
    par = Param(nk, a, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe)
    
    # init Values
    dP_ = 100
    dP = 0
    Vs = 0.1
    Re = 0.

    # Vent calculation
    while abs(dP_ - dP) > 1
        dP_ = dP

        K_cont = VentCalc.r_cont(par.D1, par.D2, par.nk, par.Sk, par.Density)
        Re = Vs * par.Dh / par.Viscosity
        K_duct = VentCalc.r_duct(par.Sk,par.Dh, par.nk, par.l, Re, par.Density)
        K_exp = VentCalc.r_exp(par.D1, par.D2, par.Sk, par.nk, par.Density)
        Ks = K_cont + K_duct + K_exp

        dP = Ks * par.Q ^ 2  # Pressure drop
        Vs = sqrt(dP / Ks) / (par.Sk *  par.nk)  # Mean velocity
    end


    #Temp calculation

    # Heat transfer coefficient calculation for duct
    fl = VentCalc.frict(Re, par.Dh, 0.00001)
    fe = VentCalc.frict(Re, par.Dh, 0.0001)
    alfa_duct = alfa_a * TempCalc.heat_tranfer(nk, a, dk, par.Dh, par.cv, Q, par.Density, par.Viscosity, par.lam) + alfa_b
    # @show alfa_duct


    # Geom for stator radial conductivity
    h1 = (Ds - par.D1) / 2
    h2 = (par.D2 - Ds) / 2

    Sgp1 = pi * (Ds+par.D1)/2 * par.l * par.lam_fe
    Sgp2 = pi * (Ds+par.D2)/2 * par.l * par.lam_fe
    Sgd = par.Ok * par.nk * par.l
    S0 = pi * par.D2 * par.l

    Gp1 = TempCalc.g_cond(par.lam_fe, h1, Sgp1)
    Gp2 = TempCalc.g_cond(par.lam_fe, h2, Sgp2)
    Gd = TempCalc.g_conv(alfa_duct, Sgd)
    G0 = TempCalc.g_conv(par.alfa_0, S0)

    Gm = [
        Gp1   -Gp1        0
        -Gp1  Gp1+Gp2+Gd  -Gp2
        0     -Gp2        Gp2+G0
    ]
    Pm = [Pl, Gd*30, G0*30]
    T = solve(LinearProblem(Gm, Pm))  # T = np.linalg.solve(Gm, Pm)
    T_av = mean(T)
    
    # Solid surface
    S_dct =  par.Sk * nk
    S_solid =   pi/4*(par.D2^2 - par.D1^2)
    S_stator =  S_solid - S_dct

    return [dP, T_av, S_stator]
end

# Constraints:  c < 0.
function domain_constraints(x)
    nk, dk, Ds = x
    
    const_gap = nk * (dk + duct_gap) / pi
    const_D1 = Ds - dk/2
    const_D2 = Ds + dk/2
    
    c1 = const_gap - Ds
    c2 = (D1_ + 2 * D1_gap) - const_D1
    c3 = const_D2 - (D2_ - 2 * D2_gap)

    return [c1, c2, c3]
end

end  # MotorParam
