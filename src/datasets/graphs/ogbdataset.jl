function __init__ogbdataset()
    DEPNAME = "OGBDataset"
    LINK = "http://snap.stanford.edu/ogb/data"
    DOCS = "https://ogb.stanford.edu/docs/dataset_overview/"
    
    # NOT WORKING AS EXPECTED
    function fetch_method(remote_filepath, local_directory_path)
        if endswith(remote_filepath, "nodeproppred/master.csv")
            local_filepath = joinpath(local_directory_path, "nodeproppred_metadata.csv")
            download(remote_filepath, local_filepath)
        elseif endswith(remote_filepath, "linkproppred/master.csv")
            local_filepath = joinpath(local_directory_path, "linkproppred_metadata.csv")
            download(remote_filepath, local_filepath)
        elseif endswith(remote_filepath, "graphproppred/master.csv")
            local_filepath = joinpath(local_directory_path, "graphproppred_metadata.csv")
            download(remote_filepath, local_filepath)
        else 
            local_filepath = DataDeps.fetch_default(remote_filepath, local_directory_path)
        end
        return local_filepath
    end

    register(DataDep(
        DEPNAME,
        """
        Dataset: The $DEPNAME dataset.
        Website: $DOCS
        Download Link: $LINK
        """,
        ["https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv",      
        "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/linkproppred/master.csv",      
        "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/graphproppred/master.csv",
        ],
        fetch_method=fetch_method
    ))
end


"""
    OGBDataset(name; dir=nothing)

The collection of datasets from the [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687)
paper. 

`name` is the name  of one of the datasets (listed [here](https://ogb.stanford.edu/docs/dataset_overview/))
available for node prediction, edge prediction, or graph prediction tasks.

# Examples

## Node prediction tasks

```julia-repl
julia> data = OGBDataset("ogbn-arxiv")
dataset OGBDataset:
  name        =>    ogbn-arxiv
  metadata    =>    Dict{String, Any} with 17 entries
  graphs      =>    1-element Vector{MLDatasets.Graph}
  graph_data  =>    nothing

julia> data[:]
Graph:
  num_nodes   =>    169343
  num_edges   =>    1166243
  edge_index  =>    ("1166243-element Vector{Int64}", "1166243-element Vector{Int64}")
  node_data   =>    (val_mask = "29799-trues BitVector", test_mask = "48603-trues BitVector", year = "169343 Vector{Int64}", features = "128×169343 Matrix{Float32}", label = "169343 Vector{Int64}", train_mask = "90941-trues BitVector")
  edge_data   =>    nothing

julia> data.metadata
Dict{String, Any} with 17 entries:
  "download_name"         => "arxiv"
  "num classes"           => 40
  "num tasks"             => 1
  "binary"                => false
  "url"                   => "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
  "additional node files" => "node_year"
  "is hetero"             => false
  "task level"            => "node"
  ⋮                       => ⋮
```

## Edge prediction task

```julia-repl
julia> data = OGBDataset("ogbl-collab")
dataset OGBDataset:
  name        =>    ogbl-collab
  metadata    =>    Dict{String, Any} with 15 entries
  graphs      =>    1-element Vector{MLDatasets.Graph}
  graph_data  =>    nothing

julia> data[:]
Graph:
  num_nodes   =>    235868
  num_edges   =>    2358104
  edge_index  =>    ("2358104-element Vector{Int64}", "2358104-element Vector{Int64}")
  node_data   =>    (features = "128×235868 Matrix{Float32}",)
  edge_data   =>    (year = "2×1179052 Matrix{Int64}", weight = "2×1179052 Matrix{Int64}")
```

## Graph prediction task

```julia-repl
julia> data = OGBDataset("ogbg-molhiv")
dataset OGBDataset:
  name        =>    ogbg-molhiv
  metadata    =>    Dict{String, Any} with 17 entries
  graphs      =>    41127-element Vector{MLDatasets.Graph}
  graph_data  =>    (labels = "41127-element Vector{Int64}", train_mask = "32901-trues BitVector", val_mask = "4113-trues BitVector", test_mask = "4113-trues BitVector")

julia> data[1]
```
"""
struct OGBDataset{GD} <: AbstractDataset
    name::String
    metadata::Dict{String, Any}
    graphs::Vector{Graph}
    graph_data::GD
end

function OGBDataset(fullname; dir = nothing)
    metadata = read_ogb_metadata(fullname, dir)
    path = makedir_ogb(fullname, metadata["url"], dir)
    metadata["path"] = path
    graph_dicts, graph_data = read_ogb_graph(path, metadata)
    graphs = ogbdict2graph.(graph_dicts)
    return OGBDataset(fullname, metadata, graphs, graph_data)
end

function read_ogb_metadata(fullname, dir = nothing)
    dir = isnothing(dir) ? datadep"OGBDataset" : dir
    @assert contains(fullname, "-") "The full name should be provided, e.g. ogbn-arxiv"
    prefix, name = split(fullname, "-")
    if prefix == "ogbn"
        path_metadata = joinpath(dir, "nodeproppred_metadata.csv")
    elseif prefix == "ogbl"
        path_metadata = joinpath(dir, "linkproppred_metadata.csv")
    elseif prefix == "ogbg"
        path_metadata = joinpath(dir, "graphproppred_metadata.csv")
    else 
        @assert "The dataset name should start with ogbn, ogbl, or ogbg."
    end
    df = read_csv(path_metadata)
    @assert fullname ∈ names(df)
    metadata = Dict{String, Any}(String(r[1]) => parse_pystring(r[2]) for r in eachrow(df[!,[names(df)[1], fullname]]))
    if prefix == "ogbn"
        metadata["task level"] = "node"
    elseif prefix == "ogbl"
        metadata["task level"] = "link"
    elseif prefix == "ogbg"
        metadata["task level"] = "graph"
    end
    return metadata
end

function makedir_ogb(fullname, url, dir = nothing)
    root_dir = isnothing(dir) ? datadep"OGBDataset" : dir
    @assert contains(fullname, "-") "The full name should be provided, e.g. ogbn-arxiv"
    prefix, name = split(fullname, "-")
    data_dir = joinpath(root_dir, name)
    if !isdir(data_dir)
        local_filepath = DataDeps.fetch_default(url, root_dir)
        currdir = pwd()
        cd(root_dir) # Needed since `unpack` extracts in working dir
        DataDeps.unpack(local_filepath)
        unzipped = local_filepath[1:findlast('.', local_filepath)-1]
        mv(unzipped, data_dir) 
        for (root, dirs, files) in walkdir(data_dir)
            for file in files
                if endswith(file, r"zip|gz")
                    cd(root)
                    DataDeps.unpack(joinpath(root, file))
                end
            end
        end
        cd(currdir)
    end
    @assert isdir(data_dir)

    return data_dir
end

## See https://github.com/snap-stanford/ogb/blob/master/ogb/io/read_graph_raw.py
function read_ogb_graph(path, metadata)
    dict = Dict{String, Any}()
    
    dict["edge_index"] = read_ogb_file(joinpath(path, "raw", "edge.csv"), Int, transp=false)
    dict["edge_index"] = dict["edge_index"] .+ 1 # from 0-indexing to 1-indexing
        
    dict["node_features"] = read_ogb_file(joinpath(path, "raw", "node-feat.csv"), Float32)
    dict["edge_features"] = read_ogb_file(joinpath(path, "raw", "edge-feat.csv"), Float32)
    
    dict["num_nodes"] = read_ogb_file(joinpath(path, "raw", "num-node-list.csv"), Int; tovec=true)
    dict["num_edges"] = read_ogb_file(joinpath(path, "raw", "num-edge-list.csv"), Int; tovec=true)
    
    for file in readdir(joinpath(path, "raw"))
        if file ∉ ["edge.csv", "num-node-list.csv", "num-edge-list.csv", "node-feat.csv", "edge-feat.csv"]
            propname = replace(split(file,".")[1], "-" => "_")
            dict[propname] = read_ogb_file(joinpath(path, "raw", file), Any)

            # from https://github.com/snap-stanford/ogb/blob/5e9d78e80ffd88787d9a1fdfdf4079f42d171565/ogb/io/read_graph_raw.py
            # # hack for ogbn-proteins
            # if additional_file == 'node_species' and osp.exists(osp.join(raw_dir, 'species.csv.gz')):
            #     os.rename(osp.join(raw_dir, 'species.csv.gz'), osp.join(raw_dir, 'node_species.csv.gz'))
        end
    end

    node_keys = [k for k in keys(dict) if startswith(k, "node_")]
    edge_keys = [k for k in keys(dict) if startswith(k, "edge_") && k != "edge_index"]
    # graph_keys = [k for k in keys(dict) if startswith(k, "graph_")] # no graph-level features in official implementation
    
    num_graphs = length(dict["num_nodes"])
    @assert num_graphs == length(dict["num_edges"])

    graphs = Dict[]
    num_node_accum = 0
    num_edge_accum = 0
    for i in 1:num_graphs
        graph = Dict{String, Any}()
        n, m = dict["num_nodes"][i], dict["num_edges"][i]
        graph["num_nodes"] = n
        graph["num_edges"] = metadata["add_inverse_edge"] ? 2*m : m

        # EDGE FEATURES
        if metadata["add_inverse_edge"]
            ei = dict["edge_index"][num_edge_accum+1:num_edge_accum+m, :]
            s, t = ei[:,1], ei[:,2]
            graph["edge_index"] = [s t; t s]
            
            for k in edge_keys
                v = dict[k]
                if v === nothing
                    graph[k] = nothing
                else 
                    x = v[:, num_edge_accum+1:num_edge_accum+m]
                    graph[k] = [x; x]
                end
            end
        else
            graph["edge_index"] = dict["edge_index"][num_edge_accum+1:num_edge_accum+m, :]
            
            for k in edge_keys
                v = dict[k]
                if v === nothing
                    graph[k] = nothing
                else 
                    graph[k] = v[:, num_edge_accum+1:num_edge_accum+m]
                end
            end
            num_edge_accum += m
        end

        # NODE FEATURES
        for k in node_keys
            v = dict[k]
            if v === nothing
                graph[k] = nothing
            else 
                graph[k] = v[:, num_node_accum+1:num_node_accum+n]
            end
        end
        num_node_accum += n

        push!(graphs, graph)
    end

    # PROCESS LABELS
    dlabels = Dict{String, Any}()
    for k in keys(dict)
        if contains(k, "label")
            if k ∉ [node_keys; edge_keys]
                dlabels[k] = dict[k]
            end
        end
    end
    labels = isempty(dlabels) ? nothing : 
             length(dlabels) == 1 ? first(dlabels)[2] : dlabels

    splits = readdir(joinpath(path, "split"))
    @assert length(splits) == 1 # TODO check if datasets with multiple splits existin in OGB
    
    # TODO sometimes splits are given in .pt format
    # Use read_pytorch in src/io.jl to load them.
    split_idx = (train = read_ogb_file(joinpath(path, "split", splits[1], "train.csv"), Int; tovec=true),
                 val = read_ogb_file(joinpath(path, "split", splits[1], "valid.csv"), Int; tovec=true),
                test = read_ogb_file(joinpath(path, "split", splits[1], "test.csv"), Int; tovec=true))
    
    if split_idx.train !== nothing 
        split_idx.train .+= 1
    end
    if split_idx.val !== nothing 
        split_idx.val .+= 1
    end
    if split_idx.test !== nothing 
        split_idx.test .+= 1
    end


    graph_data = nothing
    if metadata["task level"] == "node"
        @assert length(graphs) == 1
        g = graphs[1]
        if split_idx.train !== nothing
            g["node_train_mask"] = indexes2mask(split_idx.train, g["num_nodes"])
        end
        if split_idx.val !== nothing
            g["node_val_mask"] = indexes2mask(split_idx.val, g["num_nodes"])
        end
        if split_idx.test !== nothing
            g["node_test_mask"] = indexes2mask(split_idx.test, g["num_nodes"])
        end

    end
    if metadata["task level"] == "link"
        @assert length(graphs) == 1
        g = graphs[1]
        if split_idx.train !== nothing
            g["edge_train_mask"] = indexes2mask(split_idx.train, g["num_edges"])
        end
        if split_idx.val !== nothing
            g["edge_val_mask"] = indexes2mask(split_idx.val, g["num_edges"])
        end
        if split_idx.test !== nothing
            g["edge_test_mask"] = indexes2mask(split_idx.test, g["num_edges"])
        end
    end
    if metadata["task level"] == "graph"
        train_mask = split_idx.train !== nothing ? indexes2mask(split_idx.train, num_graphs) : nothing
        val_mask = split_idx.val !== nothing ? indexes2mask(split_idx.val, num_graphs) : nothing 
        test_mask = split_idx.test !== nothing ? indexes2mask(split_idx.test, num_graphs) : nothing

        graph_data = clean_nt((; labels=maybesqueeze(labels), train_mask, val_mask, test_mask))
    end
    return graphs, graph_data
end

function read_ogb_file(p, T; tovec = false, transp = true)
    res = isfile(p) ? read_csv(p, Matrix{T}, header=false) : nothing
    if tovec && res !== nothing
        @assert size(res, 1) == 1 || size(res, 2) == 1
        res = vec(res)
    end
    if transp && res !== nothing && !tovec
        res = collect(res')
    end
    if res !== nothing T === Any
        res = restrict_array_type(res)
    end
    return res
end


function ogbdict2graph(d::Dict)
    edge_index = d["edge_index"][:,1], d["edge_index"][:,2] 
    num_nodes = d["num_nodes"]
    node_data = Dict(Symbol(k[6:end]) => maybesqueeze(v) for (k,v) in d if startswith(k, "node_") && v !== nothing)
    edge_data = Dict(Symbol(k[6:end]) => maybesqueeze(v) for (k,v) in d if startswith(k, "edge_") && k!="edge_index" && v !== nothing)
    node_data = isempty(node_data) ? nothing : (; node_data...)
    edge_data = isempty(edge_data) ? nothing : (; edge_data...)
    return Graph(; num_nodes, edge_index, node_data, edge_data)
end

Base.length(data::OGBDataset) = length(data.graphs)
Base.getindex(data::OGBDataset{Nothing}, ::Colon) = length(data.graphs) == 1 ? data.graphs[1] : data.graphs
Base.getindex(data::OGBDataset, ::Colon) = (; data.graphs, data.graph_data.labels)
Base.getindex(data::OGBDataset{Nothing}, i) = getobs(data.graphs, i)
Base.getindex(data::OGBDataset, i) = getobs((; data.graphs, data.graph_data.labels), i) 