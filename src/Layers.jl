module Layers

using LinearAlgebra
using Printf

mutable struct Data
    input :: Array{Array{Float64}}
    true_output :: Array{Array{Float64}}
end

mutable struct Neuron
    activation_function :: String
    weights :: Array{Float64}
    nets :: Array{Float64}
    inputs :: Array{Float64}
    outputs :: Array{Float64}
    error :: Array{Float64}
end

mutable struct Layer
    N_neurons :: Int64
    neurons :: Vector{Neuron}
    nets :: Array{Float64}
    inputs :: Array{Float64}
    output :: Array{Float64}
    type :: Int8
    error :: Array{Float64}
end

mutable struct Network
    layers :: Array{Layer}
    η :: Float64 # Learning rate
    m :: Float64 # Momentum parameter
    λ :: Float64 # Tikhonov ridge regression hyperparameter
    mb_size :: Float64 # Minibatch size
    problem :: Int8
end 


function pass_through_network(network, input :: Array{Float64})

    for i=1:1:length(network.layers)

        layer = network.layers[i]

        if layer.type == 0

            # We are in the case of input layer
            layer.output = input

            continue 

        elseif layer.type == 1
            # Hidden neuron
        
            for neuron in layer.neurons
                # Take output from previous layer
                print((network.layers[i-1].output))
                print((neuron.weights))
                net = network.layers[i-1].output ⋅ neuron.weights
                neurons.outputs = neuron.activation_function(net)
                push!(layer.output, neurons.outputs)
            end

            continue

        else
            # Output neuron

            for neuron in layer.neurons

                net = ∑(network.layers[i-1].output, neuron.weights)
                neurons.outputs = neuron.activation_function(net)
                push!(layer.output, neurons.outputs)

            end

            continue

        end

    end

end

function feedforward(network, data :: Data#=multidimensional array=#)

    # for i=1:1:network.mb_size

    for j=1:1:length(data.input)

        pass_through_network(network, data.input[j])
    end

    print(network.layers[2].output)

    # end
    

    backward(network,data.true_output)

end



function backward(network, true_output)

end



function format(net :: Network)

    @printf "LR \tMomentum \tλ \tMinibatch \n" 
    @printf "%.4f \t%.4f \t%.4f \t%.4f \n\n" net.η net.m net.λ net.mb_size

    @printf "Layers \n"

    for i=1:1:length(net.layers)

        layer = net.layers[i]

        @printf "Layer level: %d\t Neurons: %d\n" layer.type layer.N_neurons

        for j=1:1:length(layer.neurons)

            neuron = layer.neurons[j]

            @printf "Act. func: %s \nWeights: " neuron.activation_function

            for weight in neuron.weights

                @printf "%.4f, " weight

            end

            @printf "\n"

        end

    end

end



end