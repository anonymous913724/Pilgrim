from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset, LinkPropPredDataset
from torch_geometric.data import Data
import numpy as np
import torch

    
class TGBDatasetLoader(object):
    def __init__(self, name, root="datasets"):
        self._dataset_name = name
        self._dataset_root = root
        self._get_data()


    def _get_data(self):
        # in PyG TemporalData format (not PyG Temporal library)
        self._dataset_pyg = PyGLinkPropPredDataset(name=self._dataset_name, root=self._dataset_root)
        self._data_pyg = self._dataset_pyg.get_TemporalData()
        # in numpy array format
        self._dataset_np = LinkPropPredDataset(name=self._dataset_name, root="datasets", preprocess=True)
        self._data_np = self._dataset_np.full_data

        self.snapshot_count = len(np.unique(self._data_pyg.t.numpy()))
        self.num_nodes = self._data_pyg.num_nodes

        if 'edge_feat' in self._data_np:
            self.num_edge_feats = self._data_np['edge_feat'][0].shape[0]
        else:
            self.num_edge_feats = 0

        # TGB datasets don't use node features, so we just define them as 1
        self.num_node_feats = 1

    def get_snapshots(self, warmup_steps=0):
        num_events = self._data_pyg.num_events
        data_edges = self._data_pyg.edge_index.T
        data_edge_weights = torch.squeeze(self._data_pyg.msg)
        t = self._data_pyg.t

        assert warmup_steps < self.snapshot_count, f"warmup steps {warmup_steps} is more than the total number of snapshots {self.snapshot_count}"

        # init edges/weights for a snapshot   
        edge_to_idx = dict()
        nodes_remap = dict()

        edge_t = []
        edge_weights_t = []

        warmup_steps_i = 0
        num_snapshots_seen = 0


        # warmup
        while num_snapshots_seen < warmup_steps:
            edge_tuple = tuple(data_edges[warmup_steps_i].detach().numpy())
            reverse_edge_tuple = (edge_tuple[1], edge_tuple[0])

            # remap node ids
            if edge_tuple[0] not in nodes_remap:
                nodes_remap[edge_tuple[0]] = len(nodes_remap)
            if edge_tuple[1] not in nodes_remap:
                nodes_remap[edge_tuple[1]] = len(nodes_remap)

            # check to see if we have attribute update rather than a new edge being added
            if edge_tuple not in edge_to_idx:
                # remap and create bidirectional edge
                new_edge = np.array([nodes_remap[int(data_edges[warmup_steps_i][0])], nodes_remap[int(data_edges[warmup_steps_i][1])]])
                new_edge_reverse = np.array([nodes_remap[int(data_edges[warmup_steps_i][1])], nodes_remap[int(data_edges[warmup_steps_i][0])]])

                edge_t.append(new_edge)
                edge_t.append(new_edge_reverse)

                edge_weights_t.append(data_edge_weights[warmup_steps_i])
                edge_weights_t.append(data_edge_weights[warmup_steps_i])

                edge_to_idx[edge_tuple] = len(edge_t) - 2
                edge_to_idx[reverse_edge_tuple] = len(edge_t) - 1
            else:
                edge_weights_t[edge_to_idx[edge_tuple]] = data_edge_weights[warmup_steps_i]
                edge_weights_t[edge_to_idx[reverse_edge_tuple]] = data_edge_weights[warmup_steps_i]

            if warmup_steps_i + 1 >= num_events or t[warmup_steps_i] != t[warmup_steps_i + 1]:
                num_snapshots_seen += 1
            
            warmup_steps_i += 1

        for i in range(warmup_steps_i, num_events):

            edge_tuple = tuple(data_edges[i].detach().numpy())
            reverse_edge_tuple = (edge_tuple[1], edge_tuple[0])

            # remap node ids
            if edge_tuple[0] not in nodes_remap:
                nodes_remap[edge_tuple[0]] = len(nodes_remap)
            if edge_tuple[1] not in nodes_remap:
                nodes_remap[edge_tuple[1]] = len(nodes_remap)

            # check to see if we have attribute update rather than a new edge being added
            if edge_tuple not in edge_to_idx:
                # remap and create bidirectional edge
                new_edge = np.array([nodes_remap[int(data_edges[i][0])], nodes_remap[int(data_edges[i][1])]])
                new_edge_reverse = np.array([nodes_remap[int(data_edges[i][1])], nodes_remap[int(data_edges[i][0])]])

                edge_t.append(new_edge)
                edge_t.append(new_edge_reverse)

                edge_weights_t.append(data_edge_weights[i])
                edge_weights_t.append(data_edge_weights[i])

                edge_to_idx[edge_tuple] = len(edge_t) - 2
                edge_to_idx[reverse_edge_tuple] = len(edge_t) - 1
            else:
                edge_weights_t[edge_to_idx[edge_tuple]] = data_edge_weights[i]
                edge_weights_t[edge_to_idx[reverse_edge_tuple]] = data_edge_weights[i]

            
            if i + 1 >= num_events or t[i] != t[i + 1]:
                # cut off for this snapshot
                self._edges = np.array(edge_t).T
                self._edge_weights = np.array(edge_weights_t)
                node_feats = np.full((len(nodes_remap), 1), 1)

                snap = Data(
                    x=torch.from_numpy(node_feats),
                    edge_index=torch.from_numpy(np.stack(self._edges)),
                    edge_attr=torch.from_numpy(np.stack(self._edge_weights))
                )

                yield snap