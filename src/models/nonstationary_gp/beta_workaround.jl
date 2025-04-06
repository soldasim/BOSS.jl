
# TODO autodiff issues

function SpecialFunctions._beta_inc_inv(a, b, p, q=1-p)
    #change tail if necessary
    if p > 0.5
        y, x = SpecialFunctions._beta_inc_inv(b, a, q, p)
        return x, y
    end

    if p == 0.0
        return (0.0, 1.0)
    end

    #Initial approx
    x = p
    r = sqrt(-2*log(p))
    p_approx = r - SpecialFunctions.@horner(r, 2.30753e+00, 0.27061e+00) / SpecialFunctions.@horner(r, 1.0, .99229e+00, .04481e+00)
    fpu = floatmin(Float64)
    lb = SpecialFunctions.logbeta(a, b)

    if a > 1.0 && b > 1.0
        r = (p_approx^2 - 3.0)/6.0
        s = 1.0/(2*a - 1.0)
        t = 1.0/(2*b - 1.0)
        h = 2.0/(s + t)
        w = p_approx*sqrt(h + r)/h - (t - s)*(r + 5.0/6.0 - 2.0/(3.0*h))
        x = a/(a + b*exp(w + w))
    else
        r = 2.0*b
        t = 1.0/(9.0*b)
        t = r*(1.0 - t + p_approx*sqrt(t))^3
        if t <= 0.0
            x = -expm1((log((1.0 - p)*b) + lb)/b)
        else
            t = (4.0*a + r - 2.0)/t
            if t <= 1.0
                x = exp((log(p*a) + lb)/a)
            else
                x = 1.0 - 2.0/(t + 1.0)
            end
        end
    end

    #solve x using modified newton-raphson iteration

    r = 1.0 - a
    t = 1.0 - b
    p_approx_prev = 0.0
    sq = 1.0
    prev = 1.0

    x = clamp(x, 0.0001, 0.9999)

    # This first argument was proposed in
    #
    # K. J. Berry, P. W. Mielke, Jr and G. W. Cran (1990).
    # Algorithm AS R83: A Remark on Algorithm AS 109: Inverse of the
    #   Incomplete Beta Function Ratio.
    # Journal of the Royal Statistical Society.
    # Series C (Applied Statistics), 39(2), 309–310. doi:10.2307/2347779
    #
    # but the last term has been changed from 13 to 34 since the
    # the original article
    #
    # Majumder, K. L., & Bhattacharjee, G. P. (1973).
    # Algorithm as 64: Inverse of the incomplete beta function ratio.
    # Journal of the Royal Statistical Society.
    # Series C (Applied Statistics), 22(3), 411-414.
    #
    # argues that the iex value should be set to -2r - 2 where r is the
    # required number of accurate digits.
    #
    # The idea with the -5.0/a^2 - 1.0/p^0.2 - 34.0 correction is to
    # use -2r - 2 (for 16 digits) for large values of a and p but use
    # a much smaller tolerance for small values of a and p.
    iex = -5.0/a^2 - 1.0/p^0.2 - 34.0
    acu = max(exp10(iex), 10 * fpu) # 10 * fpu instead of fpu avoids hangs for small values

    #iterate
    while true
        p_approx = beta_inc(a, b, x)[1]
        xin = x
        p_approx = (p_approx - p)*min(
            floatmax(),
            exp(lb + SpecialFunctions.LogExpFunctions.xlogy(r, xin) + SpecialFunctions.LogExpFunctions.xlog1py(t, -xin))
        )

        if p_approx * p_approx_prev <= 0.0
            prev = max(sq, fpu)
        end

        adj = p_approx
        tx = x - adj
        while prev <= (sq = adj^2) || tx < 0.0 || tx > 1.0
            adj /= 3.0
            tx = x - adj
        end

        #check if current estimate is acceptable
        if prev <= acu || p_approx^2 <= acu
            x = tx
            return (x, 1.0 - x)
        end

        if tx == x
            return (x, 1.0 - x)
        end

        x = tx
        p_approx_prev = p_approx
    end
end

function SpecialFunctions._beta_inc(a, b, x, y=1-x)
    p = 0.0
    q = 0.0
   # lambda = a - (a+b)*x
    if a < 0.0 || b < 0.0
        return throw(DomainError((a, b), "a or b is negative"))
    elseif a == 0.0 && b == 0.0
        return throw(DomainError((a, b), "a and b are 0.0"))
    elseif x < 0.0 || x > 1.0
        return throw(DomainError(x, "x < 0 or x > 1"))
    elseif y < 0.0 || y > 1.0
        return throw(DomainError(y, "y < 0 or y > 1"))
    else
        z = x + y - 1.0
        if abs(z) > 3.0*eps()
            return throw(DomainError((x, y), "x + y != 1.0"))         # ERROR HANDLING
        end
    end

    if isnan(x) || isnan(y) || isnan(a) || isnan(b)
        return (NaN, NaN)
    elseif x == 0.0
        return (0.0, 1.0)
    elseif y == 0.0
        return (1.0, 0.0)
    elseif a == 0.0
        return (1.0, 0.0)
    elseif b == 0.0
        return (0.0, 1.0)
    end
#EVALUATION OF ALGOS FOR PROPER SUB-DOMAINS ABOVE
    epps = max(eps(), 1.0e-15)
    if max(a, b) < 1.0E-3 * epps
        return (b/(a + b), a/(a + b))
    end
    ind = false
    a0 = a
    b0 = b
    x0 = x
    y0 = y

    if min(a0, b0) > 1.0
        #PROCEDURE FOR A0>1 AND B0>1
        lambda = a > b ? (a + b)*y - b : a - (a + b)*x
        if lambda < 0.0
            ind = true
            a0 = b
            b0 = a
            x0 = y
            y0 = x
            lambda = abs(lambda)
        end
        if b0 < 40.0 && b0*x0 <= 0.7
            p = SpecialFunctions.beta_inc_power_series(a0, b0, x0, epps)
            q = 1.0 - p
        elseif b0 < 40.0
            n = trunc(Int, b0)
            b0 -= n
            if b0 == 0.0
                n -= 1
                b0 = 1.0
            end
            p = SpecialFunctions.beta_inc_diff(b0, a0, y0, x0, n, epps)
            if x0 <= 0.7
                p += SpecialFunctions.beta_inc_power_series(a0, b0, x0, epps)
                q = 1.0 - p
            else
                if a0 <= 15.0
                    n = 20
                    p += SpecialFunctions.beta_inc_diff(a0, b0, x0, y0, n, epps)
                    a0 += n
                end
                p = SpecialFunctions.beta_inc_asymptotic_asymmetric(a0, b0, x0, y0, p, 15.0*eps())
                q = 1.0 - p
            end
        elseif a0 > b0
            if b0 <= 100.0 || lambda > 0.03*b0
                p = SpecialFunctions.beta_inc_cont_fraction(a0, b0, x0, y0, lambda, 15.0*eps())
                q = 1.0 - p
            else
                p = SpecialFunctions.beta_inc_asymptotic_symmetric(a0, b0, lambda, 100.0*eps())
                q = 1.0 - p
            end
        elseif a0 <= 100.0 || lambda > 0.03*a0
            p = SpecialFunctions.beta_inc_cont_fraction(a0, b0, x0, y0, lambda, 15.0*eps())
            q = 1.0 - p
        else
            p = SpecialFunctions.beta_inc_asymptotic_symmetric(a0, b0, lambda, 100.0*eps())
            q = 1.0 - p
        end
        return ind ? (q, p) : (p, q)
    end
#PROCEDURE FOR A0<=1 OR B0<=1
    if x > 0.5
        ind = true
        a0 = b
        b0 = a
        y0 = x
        x0 = y
    end

    if b0 < min(epps, epps*a0)
        p = SpecialFunctions.beta_inc_power_series2(a0, b0, x0, epps)
        q = 1.0 - p
    elseif a0 < min(epps, epps*b0) && b0*x0 <= 1.0
        q = SpecialFunctions.beta_inc_power_series1(a0, b0, x0, epps)
        p = 1.0 - q
    elseif max(a0, b0) > 1.0
        if b0 <= 1.0
            p = SpecialFunctions.beta_inc_power_series(a0, b0, x0, epps)
            q = 1.0 - p
        elseif x0 >= 0.3
            q = SpecialFunctions.beta_inc_power_series(b0, a0, y0, epps)
            p = 1.0 - q
        elseif x0 >= 0.1
            if b0 > 15.0
                q = SpecialFunctions.beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, 15.0*eps())
                p = 1.0 - q
            else
                n = 20
                q = SpecialFunctions.beta_inc_diff(b0, a0, y0, x0, n, epps)
                b0 += n
                q = SpecialFunctions.beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, 15.0*eps())
                p = 1.0 - q
            end
        elseif (x0*b0)^(a0) <= 0.7
            p = SpecialFunctions.beta_inc_power_series(a0, b0, x0, epps)
            q = 1.0 - p
        else
            n = 20
            q = SpecialFunctions.beta_inc_diff(b0, a0, y0, x0, n, epps)
            b0 += n
            q = SpecialFunctions.beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, 15.0*eps())
            p = 1.0 - q
        end
    elseif a0 >= min(0.2, b0)
        p = SpecialFunctions.beta_inc_power_series(a0, b0, x0, epps)
        q = 1.0 - p
    elseif x0^a0 <= 0.9
        p = SpecialFunctions.beta_inc_power_series(a0, b0, x0, epps)
        q = 1.0 - p
    elseif x0 >= 0.3
        q = SpecialFunctions.beta_inc_power_series(b0, a0, y0, epps)
        p = 1.0 - q
    else
        n = 20
        q = SpecialFunctions.beta_inc_diff(b0, a0, y0, x0, n, epps)
        b0 += n
        q = SpecialFunctions.beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, 15.0*eps())
        p = 1.0 - q
    end

#TERMINATION
    return ind ? (q, p) : (p, q)
end

function SpecialFunctions.beta_inc_power_series(a, b, x, epps)
    @assert b <= 1.0 || b*x <= 0.7
    ans = 0.0
    if x == 0.0
        return 0.0
    end
    a0 = min(a,b)
    b0 = max(a,b)
    if a0 >= 1.0
        z = a*log(x) - SpecialFunctions.logbeta(a,b)
        ans = exp(z)/a
    else

        if b0 >= 8.0
            u = SpecialFunctions.loggamma1p(a0) + SpecialFunctions.loggammadiv(a0,b0)
            z = a*log(x) - u
            ans = (a0/a)*exp(z)
            if ans == 0.0 || a <= 0.1*epps
                return ans
            end
        elseif b0 > 1.0
            u = SpecialFunctions.loggamma1p(a0)
            m = b0 - 1.0
            if m >= 1.0
                c = 1.0
                for i = 1:m
                    b0 -= 1.0
                    c *= (b0/(a0+b0))
                end
                u += log(c)
            end
            z = a*log(x) - u
            b0 -= 1.0
            apb = a0 + b0
            if apb > 1.0
                u = a0 + b0 - 1.0
                t = (1.0 + SpecialFunctions.rgamma1pm1(u))/apb
            else
                t = 1.0 + SpecialFunctions.rgamma1pm1(apb)
            end
            ans = exp(z)*(a0/a)*(1.0 + SpecialFunctions.rgamma1pm1(b0))/t
            if ans == 0.0 || a <= 0.1*epps
                return ans
            end
        else
        #PROCEDURE FOR A0 < 1 && B0 < 1
            ans = x^a
            if ans == 0.0
                return ans
            end
            apb = a + b
            if apb > 1.0
                u = a + b - 1.0
                z = (1.0 + SpecialFunctions.rgamma1pm1(u))/apb
            else
                z = 1.0 + SpecialFunctions.rgamma1pm1(apb)
            end
            c = (1.0 + SpecialFunctions.rgamma1pm1(a))*(1.0 + SpecialFunctions.rgamma1pm1(b))/z
            ans *= c*(b/apb)
            #label l70 start
            if ans == 0.0 || a <= 0.1*epps
                return ans
            end
        end
    end
    if ans == 0.0 || a <= 0.1*epps
        return ans
    end
    # COMPUTE THE SERIES

    sm = 0.0
    n = 0.0
    c = 1.0
    tol = epps/a
    n += 1.0
    c *= x*(1.0 - b/n)
    w = c/(a + n)
    sm += w
    while abs(w) > tol
        n += 1.0
        c *= x*(1.0 - b/n)
        w = c/(a+n)
        sm += w
    end
    return ans*(1.0 + a*sm)
end

function SpecialFunctions.beta_inc_diff(a, b, x, y, n, epps)
    apb = a + b
    ap1 = a + 1.0
    mu = 0.0
    d = 1.0
    if n != 1 && a >= 1.0 && apb >= 1.1*ap1
        mu = abs(SpecialFunctions.exparg_n)
        k = SpecialFunctions.exparg_p
        if k < mu
            mu = k
        end
        t = mu
        d = exp(-t)
    end

    ans = SpecialFunctions.beta_integrand(a, b, x, y, mu)/a
    if n == 1 || ans == 0.0
        return ans
    end
    nm1 = n -1
    w = d

    k = 0
    if b <= 1.0
        kp1 = k + 1
        for i = kp1:nm1
            l = i - 1
            d *= ((apb + l)/(ap1 + l))*x
            w += d
            if d <= epps*w
                break
            end
        end
        return ans*w
    elseif y > 1.0e-4
        r = trunc(Int,(b - 1.0)*x/y - a)
        if r < 1.0
            kp1 = k + 1
            for i = kp1:nm1
                l = i - 1
                d *= ((apb + l)/(ap1 + l))*x
                w += d
                if d <= epps*w
                    break
                end
            end
            return ans*w
        end
        k = t = nm1
        if r < t
            k = r
        end
        # ADD INC TERMS OF SERIES
        for i = 1:k
            l = i -1
            d *= ((apb + l)/(ap1 + l))*x
            w += d
        end
        if k == nm1
            return ans*w
        end
    else
        k = nm1
        for i = 1:k
            l = i -1
            d *= ((apb + l)/(ap1 + l))*x
            w += d
        end
        if k == nm1
            return ans*w
        end
    end

    kp1 = k + 1
    for i in kp1:nm1
        l = i - 1
        d *= ((apb + l)/(ap1 + l))*x
        w += d
        if d <= epps*w
            break
        end
   end
   return ans*w
end

function SpecialFunctions.beta_integrand(a, b, x, y, mu=0.0)
    a0, b0 = minmax(a,b)
    if a0 >= 8.0
        if a > b
            h = b/a
            x0 = 1.0/(1.0 + h)
            y0 = h/(1.0 + h)
            lambda = (a+b)*y - b
        else
            h = a/b
            x0 = h/(1.0 + h)
            y0 = 1.0/(1.0 + h)
            lambda = a - (a+b)*x
        end
        e = -lambda/a
        u = abs(e) > 0.6 ? e - log(x/x0) : - SpecialFunctions.LogExpFunctions.log1pmx(e)
        e = lambda/b
        v = abs(e) > 0.6 ? e - log(y/y0) : - SpecialFunctions.LogExpFunctions.log1pmx(e)
        z = esum(mu, -(a*u + b*v))
        return sqrt(inv2π*b*x0)*z*exp(-SpecialFunctions.stirling_corr(a,b))
    elseif x > 0.375
        if y > 0.375
            lnx = log(x)
            lny = log(y)
        else
            lnx = log1p(-y)
            lny = log(y)
        end
    else
        lnx = log(x)
        lny = log1p(-x)
    end
    z = a*lnx + b*lny
    if a0 < 1.0
        b0 = max(a,b)
        if b0 >= 8.0
            u = SpecialFunctions.loggamma1p(a0) + SpecialFunctions.loggammadiv(a0,b0)
            return a0*(esum(mu, z-u))
        elseif b0 > 1.0
            u = SpecialFunctions.loggamma1p(a0)
            n = trunc(Int,b0 - 1.0)
            if n >= 1
                c = 1.0
                for i = 1:n
                    b0 -= 1.0
                    c *= (b0/(a0+b0))
                end
                u += log(c)
            end
            z -= u
            b0 -= 1.0
            apb = a0 + b0
            if apb > 1.0
                u = a0 + b0 - 1.0
                t = (1.0 + SpecialFunctions.rgamma1pm1(u))/apb
            else
                t = 1.0 + SpecialFunctions.rgamma1pm1(apb)
            end
            return a0*(esum(mu,z))*(1.0 + SpecialFunctions.rgamma1pm1(b0))/t
        else
            ans = esum(mu, z)
            if ans == 0.0
                return 0.0
            end
            apb = a + b
            if apb > 1.0
                z = (1.0 + SpecialFunctions.rgamma1pm1(apb - 1.0))/apb
            else
                z = 1.0 + SpecialFunctions.rgamma1pm1(apb)
            end
            c = (1.0 + SpecialFunctions.rgamma1pm1(a))*(1.0 + SpecialFunctions.rgamma1pm1(b))/z
            return ans*(a0*c)/(1.0 + a0/b0)
        end
    else
        z -= SpecialFunctions.logbeta(a, b)
        ans = SpecialFunctions.esum(mu, z)
        return ans
    end
end

function SpecialFunctions.esum(mu, x)
    if x > 0.0
        if mu > 0.0 || mu + x < 0.0
            return exp(mu)*exp(x)
        else
            return exp(mu + x)
        end
    elseif mu < 0.0 || mu + x > 0.0
        return exp(mu)*exp(x)
    else
        return exp(mu + x)
    end
end
