"""
This module provides function that run a particle filtering algorithm 

"""
module particle_filter
using Distributions
using StatsBase
"""
a partially observable markov process model object

prior - a function that take an argument N and retuns
        an array with N rows with a colum for each 
        paramter and state variables
process - a function that takes a vector of state variables and 
        parameters for the model and returns a vector with updated 
        state varaibles
observaiton - a function that takes an observaiton Y_t and a 
            vector of states and paramters and return the 
            likelihood of the Y_t condiitonal on the states and
            paramters
"""
struct POMP
    prior
    process!
    observaiton
end 



function filter(POMP, Y_t; N=10000)
    X_t = POMP.prior(N)

    T = length(Y_t)
    N,m = size(X_t)
    samples_before = zeros(N,m,T)
    samples_after = zeros(N,m,T)
    for t in 1:T
        p = mapslices(x -> POMP.observaiton(Y_t[t],x),X_t,dims = 2)
        p = p./sum(p)
   
        S = sample(collect(1:N),StatsBase.pweights(p),N)
        
        X_t = X_t[S,:]
        samples_before[:,:,t] = X_t
        X_t = mapslices(x -> POMP.process!(x),X_t,dims = 2)
        samples_after[:,:,t] = X_t
    end 
    return samples_before,samples_after 
end 

function blury_filter(POMP, Y_t, noise!; N=10000)
    X_t = POMP.prior(N)

    T = length(Y_t)
    N,m = size(X_t)
    samples_before = zeros(N,m,T)
    samples_after = zeros(N,m,T)
    for t in 1:T
        p = mapslices(x -> POMP.observaiton(Y_t[t],x),X_t,dims = 2)
        p = p./sum(p)
   
        S = sample(collect(1:N),StatsBase.pweights(p),N)
        
        X_t = X_t[S,:]
        samples_before[:,:,t] = X_t
        X_t = mapslices(x -> POMP.process!(x),X_t,dims = 2)
        X_t = mapslices(x -> noise!(x),X_t,dims = 2)
        samples_after[:,:,t] = X_t
    end 
    return samples_before,samples_after 
end 


function weighted_filter(POMP, Y_t; N=10000)
    X_t = POMP.prior(N)
    weights = zeros(N)
    T = length(Y_t)
    N,m = size(X_t)
    weights_before = zeros(N,T)
    weights_after = zeros(N,T)
    for t in 1:T
        p = mapslices(x -> POMP.observaiton(Y_t[t],x),X_t,dims = 2)
        weights += log.(p)
        weights_before[:,t] = weights
        X_t = mapslices(x -> POMP.process!(x),X_t,dims = 2)
        weights_after[:,t] = weights
    end 
    return weights_before,weights_after
end 



end # module