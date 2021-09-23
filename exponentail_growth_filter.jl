module exponentail_growth_filter


# state transitions 
function update_states_log(log_x,r,d)
    return log(r) + log_x +rand(d,1)[1]
end
    
# belief update
function sigma(sigma, sigma_nu)
    return  sigma + sigma_nu
end 
    
function mu(log_x_hat, r)
    return log_x_hat + log(r) 
end     

# bayesian update
function sample(log_x,d_sample)
    return exp(log_x + rand(d_sample, 1)[1])
end 

function sigma_prime(sigma, sigma_omega)
    return sqrt((1/sigma^2 + 1/sigma_omega^2)^(-1))
end 
    
function mu_prime(log_x_hat, sigma, O_t, sigma_omega)
    return sigma_prime(sigma, sigma_omega)^2*(log_x_hat/sigma^2+log(O_t)/sigma_omega^2)
end 
      
function Z(log_x_hat, sigma, O_t, sigma_omega)
    x = (log_x_hat^2/sigma^2 + log(O_t)^2/sigma_omega^2) - (1/sigma^2+1/sigma_omega^2)^(-1)*(log_x_hat/sigma^2 + log(O_t)/sigma_omega^2)^2
    return exp(-0.5*x)
end 
    
function p(p, log_x_hat0, log_x_hat1, sigma_nu, O_t, sigma_omega)
    Z_0 = Z(log_x_hat0, sigma_nu, O_t, sigma_omega) 
    Z_1 = Z(log_x_hat1, sigma_nu, O_t, sigma_omega) 
    return p*Z_0/(p*Z_0 + (1-p)*Z_1)
end 


end # module