module Layers


using LinearAlgebra


mutable struct Network
    layers :: Array{Layer}
    η :: Float64 # Learning rate
    m :: Float64 # Momentum parameter
    λ :: Float64 # Tikhonov ridge regression hyperparameter
    mb_size :: Float64 # Minibatch size
end 

mutable struct Layer
    N_neurons :: Int64
    neurons :: Vector{Neuron}
    weights :: Array{Float64}
    nets :: Array{Float64}
    inputs :: Array{Float64}
    output :: Array{Float64}
    type :: Int8
end

mutable struct Neuron
    activation_function :: String
    weights :: Array{Float64}
    nets :: Array{Float64}
    inputs :: Array{Float64}
    outputs :: Array{Float64}
end


function pass_through_network(network,input)

    for i=1:1:length(network.layers)

        if layer.type == 0

            # We are in the case of input layer
            layer.output = input

            return 

        elseif layer.type == 1
            # Hidden neuron
        
            for neuron in layer.neurons
                # Take output from previous layer
                net = ∑(network.layers[i-1].output, neuron.weights)
                neurons.outputs = neuron.activation_function(net)
                push!(layer.output, neurons.outputs)
            end

        else
            # Output neuron

            output = 0

        end

    end

end

function feedforward(network, minibatch :: Array{Float64}#=multidimensional array=#)

    for input in minibatch
        pass_through_network
    end

end



function backward()

end



end