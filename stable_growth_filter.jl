module stable_growth_filter

include("exponentail_growth_filter.jl")
using NLsolve
# state transitions 
function update_states_H0(log_x_t,r,K, d)
    x_t = exp(log_x_t)
    x_t1 = (r*x_t+ (1-r)*K)*exp(rand(d,1)[1])
    return log(x_t1)
end

function update_states_H1(log_x_t,r, d)
    return log(r) + log_x_t + rand(d,1)[1]
end


## belief update
log_normal_mean(mu, sigma) = exp(mu + (sigma^2)/2)
 
log_normal_sigma(mu, sigma) = (exp(sigma^2)-1)*exp(2*mu+sigma^2)

function update_moments(log_x_hat,sigma,sigma_nu,r,K)
    mu_prime = log(r) + log_x_hat
    sigma_prime = sigma + sigma_nu
    E = log_normal_mean(mu_prime+ (1-r)*K, sigma_prime) 
    V = log_normal_sigma(mu_prime, sigma_prime)
    return E, V
end 


function match_moments(E,V)
    function f(x)
        y = zeros(2)
        y[1] = E - log_normal_mean(x[1],x[2])
        y[2] = V - log_normal_sigma(x[1],x[2])
        return y
    end 
    sol = NLsolve.nlsolve(f, [log(E),sqrt(V)]; xtol = 10^-6)
    return sol.zero
end 

function update_params(log_x_hat,sigma,sigma_nu,r,K)
    E,V = update_moments(log_x_hat,sigma,sigma_nu,r,K)
    x = match_moments(E,V)
    return x[1],x[2]
end 


# full update

function update_beleifs(B,log_x_t,r,K,sigma_nu,sigma_omega, d_process,d_observation)
    log_x_H0, log_x_H1 = B[1], B[2]
    sigma_H0, sigma_H1 = B[3], B[4]
    p_t = B[5]
    # state transition
    #H0
    log_x_H0,sigma_H0 = update_params(log_x_H0,sigma_H0,sigma_nu,r,K)
    #H1
    sigma_H1 = exponentail_growth_filter.sigma(sigma_H1, sigma_nu)
    log_x_H1 = exponentail_growth_filter.mu(log_x_H1, r)
    
    # bayesian update
    O_t = exponentail_growth_filter.sample(log_x_t,d_observation)
    # p
    p_t = exponentail_growth_filter.p(p_t, log_x_H0, log_x_H1, sigma_nu, O_t, sigma_omega)
    #H0
    log_x_H0 = exponentail_growth_filter.mu_prime(log_x_H0, sigma_H0, O_t, sigma_omega)
    sigma_H0 = exponentail_growth_filter.sigma_prime(sigma_H0, sigma_omega)
    #H1
    log_x_H1 = exponentail_growth_filter.mu_prime(log_x_H1, sigma_H1, O_t, sigma_omega)
    sigma_H1 = exponentail_growth_filter.sigma_prime(sigma_H1, sigma_omega)
    return [log_x_H0,log_x_H1,sigma_H0,sigma_H1,p_t]
end 


end # module 