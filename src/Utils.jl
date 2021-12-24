module Utils

using LinearAlgebra

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



end