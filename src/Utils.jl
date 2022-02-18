module Utils

using LinearAlgebra
using Distributions
using Random
using DataFrames

#=
    Identity function, i.e. 
        y = x
=#
function identity(x)
    return x
end 

function d_identity(x)
    return 1
end

function relu(x)
    x = max.(x, 0)
end

function d_relu(x)
    x = replace!(el -> el > 0 ? 1 : 0, x)
    return x
end

function sigmoid(x, a)
    one = ones((length(x), 1))
    return one ./ (one + ℯ^(- a.* x))
end

function d_sigmoid(x, a)
    one = ones((length(x), 1))
    sig = sigmoid(x,a)
    return sig .* (one - sig)
end

function hyper_tan(x)
    return tanh(x)
end

function d_hyper_tan(x)
    one = ones((length(x),1))
    return one - hyper_tan(x).^2
end

function LS(predicted, target)
    return 0.5 .* (predicted - target).^2
end 

function d_LS(predicted, target)
    return predicted - target
end


function he_init(in_features, out_features)
    s = sqrt( 2 / in_features )
    d = Normal(0, s)
    return rand(d, out_features)
end


function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end


end