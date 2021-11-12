module nonlinear_growth



function logNormal(x,mu,sigma)
    return 1/(x*sigma*sqrt(2*pi))*exp(-1/2*(((log(x)-log(mu))/sigma)^2))
end 

g(x) = x/(1+x)   

function integrated_logNormal_reimann(x,g,mu,sigma_x,sigma_mu)
    grid = 0.05:0.01:10
    return sum(logNormal.(x,g.(grid),sigma_x).*logNormal.(grid,mu,sigma_mu).*0.05)
end
    
    

end # module 