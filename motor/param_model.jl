module MotorParam

using LinearSolve
using Distributions

### CONSTANTS
const Q = 0.5
const D1 = 0.297
const D2 = 0.4
const l = 0.23
const t = 30
const alt = 325
const alfa_0 = 16
const lam_fe = 29


### Param

struct Param
    nk::Float64
    Dk::Float64
    D1::Float64
    D2::Float64
    alfa_0::Float64
    lam_fe::Float64
    l::Float64
    Q::Float64
    t::Float64
    alt::Float64
    Pressure::Float64
    Density::Float64
    Viscosity::Float64
    cv::Float64
    lam::Float64
    k_alf::Float64

    function Param(nk, Dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe)
        Pressure, Density, Viscosity, cv, lam, k_alf = material_properties(t, alt)
        new(nk, Dk, D1, D2, alfa_0, lam_fe, l, Q, t, alt, Pressure, Density, Viscosity, cv, lam, k_alf)
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

### end


### VentCalc

function r_cont(D1, D2, nk, dk, dens)
    S1 = pi/4*(D2^2 - D1^2)
    S2 = pi/4 * (dk^2 * nk)

    mi = 12.174 * (S2/S1) ^ 6 - 36.685 * (S2/S1) ^ 5 + 44.366 * (S2/S1) ^ 4 - 27.069 * (S2/S1) ^ 3 + 8.7337 * (S2/S1) ^ 2 - 1.2192*(S2/S1) + 0.6797

    if mi > 1
        mi = 1
    end

    Ksi_cont = (1 / mi - 1) ^ 2
    K_cont = 0.5 * dens * (Ksi_cont + 1 - (S2 / S1) ^ 2) / S2 ^ 2
    return K_cont
end

function r_duct(dk, nk, l, Re, dens)
    S = pi/4 * (dk^2 * nk)
    e = 1.2e-04 / dk  # 1e-04
    f = 0.02 #0.02
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

    coeff_duct = f * l / dk
    K_duct = 0.5 * dens * coeff_duct / S ^ 2
    return K_duct
end

function r_exp(D1, D2, nk, dk, dens)
    S1 = pi/4 * (dk ^ 2 * nk)
    S2 = pi/4*(D1 ^ 2 - D2 ^ 2)
    K_exp = 0.5 * dens * (1 / S1 - 1 / S2) ^ 2
    return K_exp
end

function frict(Re, dk, e)
    K = e / dk
    L = 0.04
    L2 = (1 / (1.74 - 2 * log10(2 * K + 18.7 / (Re * L ^ (0.5))))) ^ 2
    while (abs(L2 - L) > 0.0001)
        L = L2
        L2 = (1 / (1.74 - 2 * log10(2 * K + 18.7 / (Re * L ^ (0.5))))) ^ 2
    end
    return L
end

### end


### TempCalc

function g_cond(lam, L, S)
    G = (lam*S) / L
    return G
end

function g_conv(alfa, S)
    G = alfa * S
    return G
end

function heat_tranfer(dk, l, v, fl, fe, Kalf)
    Kl = 1+(dk/l)^0.67 
    alfa_duct = Kalf * v^0.8 * dk^(-0.2) * Kl * sqrt(fe/fl)
    return alfa_duct
end

function calc(nk, dk, Ds;
    Pl=5000,
    alfa_a=1.,
    alfa_b=0.,
)
    par = Param(nk, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe)
    # init Values
    Dp_ = 100
    Dp = 0
    Vs = 0.1

    # Vent calculation
    Re = 0.
    while abs(Dp_ - Dp) > 1
        Dp_ = Dp

        K_cont = r_cont(par.D1, par.D2, par.nk, par.Dk, par.Density)
        Re = Vs * par.Dk / par.Viscosity
        K_duct = r_duct(par.Dk, par.nk, par.l, Re, par.Density)
        K_exp = r_exp(par.Dk, par.nk, par.l, Re, par.Density)
        Ks = K_cont + K_duct + K_exp

        Dp = Ks * par.Q ^ 2  # Pressure drop
        Vs = sqrt(Dp / Ks) / (pi / 4 * par.Dk ^ 2 * par.nk)  # Mean velocity
    end

    #Temp calculation

    # Heat transfer coefficient calculation for duct
    fl = frict(Re, par.Dk, 0.00001)
    fe = frict(Re, par.Dk, 0.0001)
    alfa_duct = alfa_a * heat_tranfer(par.Dk, par.l, Vs, fl, fe, par.k_alf) + alfa_b

    # Geom for stator radial conductivity
    # Ds = (par.D2 + par.D1) / 2
    h1 = (Ds - par.D1)/2
    h2 = (par.D2 - Ds) / 2

    Sgp1 = pi * (Ds+par.D1)/2 * par.l * par.lam_fe
    Sgp2 = pi * (Ds+par.D2)/2 * par.l * par.lam_fe
    Sgd = pi * par.Dk * par.nk * par.l
    S0 = pi * par.D2 * par.l

    Gp1 = g_cond(par.lam_fe, h1, Sgp1)
    Gp2 = g_cond(par.lam_fe, h2, Sgp2)
    Gd = g_conv(alfa_duct, Sgd)
    G0 = g_conv(par.alfa_0, S0)

    Gm = [
        Gp1   -Gp1        0;
        -Gp1  Gp1+Gp2+Gd  -Gp2;
        0     -Gp2        Gp2+G0;
    ]
    Pm = [Pl, Gd*30, G0*30]
    T = solve(LinearProblem(Gm, Pm))

    T_av = mean(T)
    
    # Solid surface
    S_dct =  pi/4 * (dk^2 * nk)
    S_solid =   pi/4*(par.D2^2 - par.D1^2)
    S_stator =  S_solid - S_dct

    return [Dp, T_av, S_stator]
end

### end


# Domain constraints for Optim optimization.
#   c < 0.
function domain_c!(c, x)
    nk, dk, Ds = x

    duct_gap = 0.005
    D1_gap = 0.002
    D2_gap = 0.003
    
    const_gap = nk * (dk + duct_gap) / pi
    const_D1 = Ds - dk/2
    const_D2 = Ds + dk/2
    
    c[1] = const_gap - Ds
    c[2] = (D1 + 2 * D1_gap) - const_D1
    c[3] = const_D2 - (D2 - 2 * D2_gap)

    return c
end

# function domain_j!(j, x)
#     nk, dk, Ds = x
#     duct_gap = 0.005
#
#     # 1st constraint
#     j[1,1] = (dk + duct_gap) / pi
#     j[1,2] = nk / pi
#     j[1,3] = -1.
#
#     # 2nd constraint
#     j[2,1] = 0.
#     j[2,2] = 0.5
#     j[2,3] = -1.
#
#     # 3rd constraint
#     j[3,1] = 0.
#     j[3,2] = 0.5
#     j[3,3] = 1.
#
#     return j
# end

# function domain_h!(h, x, λ)
#     h[1,2] += λ[1] * (1. / pi)
#     h[2,1] = h[1,2]
#     return h
# end

end  # MotorParam
