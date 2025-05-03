


#define here the models to test against the framework


#model to test
gnn_model_graphsage = GraphSAGE(
    in_channels=data.num_features,
    hidden_channels=256,
    num_layers=3,
    out_channels=2,
)