using MLDatasets
using NPZ
using GraphNeuralNetworks
using Flux
# Load the dataset
dataset = METRLA()
gdataset=dataset.graphs[1]
gs=Array{GNNGraph}(undef,12)
for i in 1:12
gs[i]= GNNGraph(gdataset.edge_index, edata = gdataset.edge_data, ndata = gdataset.node_data.features[1][i,:,:]) 
end

gs[1]

f = GNNChain(GCNConv(2 => 8,relu), GCNConv(8 => 2,sigmoid))


struct GCN
    layers::NamedTuple
end

Flux.@functor GCN 

function GCN(num_features, nout, hidden_channels;add_self_loops = false)
    layers = (conv1 = GCNConv(num_features => hidden_channels,relu; add_self_loops),
              conv2 = GCNConv(hidden_channels => nout; add_self_loops))
    return GCN(layers)
end

function (gcn::GCN)(g::GNNGraph, x)
    l = gcn.layers
    x = l.conv1(g, x)
    x = l.conv2(g, x)
    return x
end


struct update
    layers::NamedTuple
end

Flux.@functor update 

function update(num_features, nout, hidden_channels;add_self_loops = false)
    layers = (conv = GCN(num_features, nout, hidden_channels; add_self_loops),
              dense = Dense(2 * nout => nout,sigmoid))
    return update(layers)
end

function (update::update)(g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix)
    l = update.layers
    x = l.conv(g, x)
    x = l.dense([x; h])
    return x
end

struct reset
    layers::NamedTuple
end

Flux.@functor reset 

function reset(num_features, nout, hidden_channels;add_self_loops = false)
    layers = (conv = GCN(num_features, nout, hidden; add_self_loops),
              dense = Dense(hidden_channels => nout, σ = sigmoid))
    return reset(layers)
end

function (reset::reset)(g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix)
    l = reset.layers
    x = l.conv(g, [x; h])
    x = l.dense(g, x)
    return x
end

struct candidate
    layers::NamedTuple
end

Flux.@functor candidate 

function candidate(num_features, nout, hidden_channels;add_self_loops = false)
    layers = (conv = GCN(num_features, nout, hidden; add_self_loops),
              dense = Dense(hidden_channels => nout, σ = tanh))
    return candidate(layers)
end

function (candidate::candidate)(g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix, r)
    l = candidate.layers
    x = l.conv(g, [x; h * r])
    x = l.dense(g, x)
    return x
end

function set_hidden_state(x::AbstractMatrix,h ::Union{AbstractMatrix, Nothing})
    if h == nothing
        h = zeros(size(x))        
    end
    return h
end

function compute_hidden_state(z::AbstractMatrix, h::AbstractMatrix, c::AbstractMatrix)
    h = z * h + (1 - z) * c
    return h
end