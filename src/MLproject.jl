include("./Utils.jl")
include("./Layers.jl")

using LinearAlgebra
using CSV
using DataFrames
using Random
using .Utils
using .Layers

Random.seed!(123)

# Set up dataframe
df = DataFrame(Feature1=Float32[], Feature2=Float32[], Feature3=Float32[], Feature4=Float32[], Class=Int8[])

# Read input file
words = readlines("data/data_banknote_authentication.txt")

for line in words
    local vect = Vector{Float32}(undef, 5)
    fill!(vect, 0.0)
    datas = split(line, ",")

    s = ""
    for item in datas[1]
        s = s * item
    end

    vect[1] = parse(Float32, s)


    s = ""
    for item in datas[2]
        s = s * item
    end

    vect[2] = parse(Float32, s)


    s = ""
    for item in datas[3]
        s = s * item
    end

    vect[3] = parse(Float32, s)


    s = ""
    for item in datas[4]
        s = s * item
    end

    vect[4] = parse(Float32, s)


    s = ""
    for item in datas[5]
        s = s * item
    end

    vect[5] = parse(Int8, s)

    push!(df, vect)
end

# print(df)

# Shuffle dataset
df = df[shuffle(axes(df, 1)), :]

# Separate features and target 
portion = 0.75

training, testing = Utils.splitdf(df, portion)

training_features = training[:, Cols( Between(:Feature1, :Feature4) ) ]
training_target = training[:, :Class]

testing_features = testing[:, Cols( Between(:Feature1, :Feature4) ) ]
testing_target = testing[:, :Class]

# Instantiate neural network
hid = 1
neurons = 10
output_size = 1

# Create Input neurons
num_features = size(training_features)[2]

layers = Layers.Layer[]

vect = Layers.Neuron[]

for i=1:1:num_features

    weights = Utils.he_init(num_features, neurons)

    local neuron = Layers.Neuron(
        "id",
        weights,
        [],
        [],
        [],
        []
    )

    push!(vect, neuron)
end 

input_layer = Layers.Layer(
    num_features,
    vect,
    [],
    [],
    [],
    0,
    []
)

push!(layers, input_layer)


for i=1:1:hid

    local vect = Layers.Neuron[]
    for j=1:1:neurons
        
        weights = Utils.he_init(num_features, 1)

        local neuron = Layers.Neuron(
            "sig",
            weights,
            [],
            [],
            [],
            []
        )

        push!(vect, neuron)

    end

    layer = Layers.Layer(
        neurons,
        vect,
        [],
        [],
        [],
        1,
        []
    )

    push!(layers, layer)

end

neuron = Layers.Neuron(
    "sig",
    [],
    [],
    [],
    [],
    []
)

output_layer = Layers.Layer(
    output_size,
    [neuron],
    [],
    [],
    [],
    2,
    []
)

push!(layers, output_layer)

net = Layers.Network(
    layers,
    1e-2,
    1e-2,
    1e-2,
    32,
    0
)


Layers.format(net)


# Setting up data
training_features = collect.(eachrow(training_features))

training_target = collect.(eachrow(training_target))


testing_features = collect.(eachrow(testing_features))

testing_target = collect.(eachrow(testing_target))

# Launch training specifying parameters 

training_data = Layers.Data(
    training_features,
    training_target
)

testing_data = Layers.Data(
    testing_features,
    testing_target
)


Layers.feedforward(net, training_data)