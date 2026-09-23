"""Current paper model keys, with explicit historical-release aliases."""
LEGACY_PFLOTRAN_MODELS = (
    "sa_tc_wgno_h10_k5", "sa_tc_wgno_h10_k7", "plain_graph_wno_h10",
    "graph_fno_h10", "meshgraphnet_h10", "gat_h10", "gatv2_h10",
    "gps_transformer_h10", "rnn_h10", "rnn_gnn_fusion_h10",
)

LEGACY_SI_MODELS = (
    "SwiGLU-TC-WGNO-D14-K6", "SA-TC-WGNO-D14-K6", "SA-TC-WGNO-D14-K2",
    "GraphFNO-D14", "GWNO-D14", "GPS-Transformer-D14", "GATv2-D14",
    "GAT-D14", "MeshGraphNet-D14", "RNN-GNN-Fusion-D14", "RNN-D14",
)


def pflotran_registry(graph, modes, eigenvalues):
    from tcwgno.pflotran_benchmark.paper import build_paper_specs
    return build_paper_specs(graph, modes, eigenvalues)

PFLOTRAN_MODELS = (
    "wavelet_swiglu_raw_sa_tcwno_h10_k5_m128_d10", "plain_graph_wno_h10",
    "graph_fno_h10", "meshgraphnet_h10", "gat_h10", "gatv2_h10",
    "gps_transformer_h10", "rnn_h10", "rnn_gnn_fusion_h10",
)
SI_MODELS = (
    "SwiGLU-TC-WGNO-D14-K5-H133", "GraphFNO-D14", "GWNO-D14",
    "GPS-Transformer-D14", "GATv2-D14", "GAT-D14", "MeshGraphNet-D14",
    "RNN-GNN-Fusion-D14", "RNN-D14",
)
