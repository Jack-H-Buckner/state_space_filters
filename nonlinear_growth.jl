module nonlinear_growth



function logNormal(x,mu,sigma)
    return 1/(x*sigma*sqrt(2*pi))*exp(-1/2*(((log(x)-log(mu))/sigma)^2))
end 

g(x) = 1.1*x/(1+0.1*x)   

function integrated_logNormal_reimann(x,g,mu,sigma_x,sigma_mu)
    grid = 0.05:0.005:3
    return sum(logNormal.(x,g.(grid),sigma_x).*logNormal.(grid,mu,sigma_mu).*0.05)
end

function log_Beverton_Holt(logX,a,b)
    return log(a*exp(logX)/(1+b*exp(logX)))
end 
    
function dx_log_Beverton_Holt(logX,a,b)
    return 1/(1+b*exp(logX))
end 
    
function dx2_log_Beverton_Holt(logX,a,b)
    return b*exp(logX)/(1+b*exp(logX))^2
end 

"""
Aproximates firt two moments of a normally distributed random variable 
transformed by the log_Beverton_Hold function 
"""
function moments_FX(E_logX, V_logX,a,b)
    E = log_Beverton_Holt(E_logX,a,b) + 1/2*dx2_log_Beverton_Holt(E_logX,a,b)*V_logX
    V = dx_log_Beverton_Holt(E_logX,a,b)^2*V_logX-1/4*dx2_log_Beverton_Holt(E_logX,a,b)^2*V_logX
    return E, V
end 

function aprox_Normal_taylor(LogX,mu,sigma_x,sigma_mu,a,b)
    E, V = moments_FX(mu, sigma_x^2,a,b)
    V = sigma_mu^2 + V
    p = exp(-(LogX - E)^2/(2*V))/(sqrt(2*pi*V))
    return p
end 

end # module 