"""
Create observational and interventional datasets from synthetic SCMs.

The script generates .npz files containing arrays with the following keys:
 - input: observational samples of shape (n_scms, n_samples, n_vars)
 - target_data: interventional samples arranged as
     (n_interventions, n_scms, n_targets, n_samples, n_vars)
 - graphs: adjacency matrices for each SCM (n_scms, n_vars, n_vars)
 - targets: intervention targets for each SCM

These files are consumed by the training and evaluation scripts in the repo.
"""

import argparse
import inspect
import json
import os
from typing import List
import random
# Must be set before JAX is imported to prevent XLA from spawning threads,
# which would make os.fork() deadlock in the multiprocessing pool.
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm
from CausalPlayground import StructuralCausalModel, SCMGenerator, CausalGraphSetGenerator, CausalGraphGenerator
import networkx as nx
from concurrent.futures import ThreadPoolExecutor


class RandFct(torch.nn.Module):
    def __init__(self, n: int, n_hidden_nodes: int, n_hidden_layers: int):
        """Initialize the small PyTorch MLP used as a random function.

        This helper is only used by the dataset generation script to create
        randomized structural functions. It is intentionally lightweight and
        non-critical to training reproducibility.
        """
        super(RandFct, self).__init__()
        self.inlayer = torch.nn.Linear(n, n_hidden_nodes)
        self.hiddens = []
        self.n_hidden_layers = n_hidden_layers
        [self.hiddens.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))for _ in range(self.n_hidden_layers)]
        self.outlayer = torch.nn.Linear(n_hidden_nodes, 1)

        for i in range(self.n_hidden_layers):
            self.hiddens[i].weight.data.uniform_(-1, 1)
            self.hiddens[i].bias.data.uniform_(-20, 20)

    def forward(self, x):
        x = torch.relu(self.inlayer(x))
        for i in range(self.n_hidden_layers):
            x = torch.relu(self.hiddens[i](x))
        return self.outlayer(x) + random.gauss(0, 1)  # Adding the noise here instead of exogenous nodes to make implementation easier




def f_rand_net(parents: List[str]):
    """Create a random function by initializing an MLP with random weights."""
    net = RandFct(len(parents), 10, 5)

    def f(**kwargs):
        # collect parents values
        vals = torch.tensor([kwargs[p] for p in parents]).float()
        out = net(vals)
        return float(out)
    return f

class RandFctNoise(torch.nn.Module):
    def __init__(self, n: int, n_hidden_nodes: int, n_hidden_layers: int):
        """Initialize the network with n input parameters and n_hidden_layers with n_hidden nodes."""
        super(RandFct, self).__init__()
        self.inlayer = torch.nn.Linear(n+1, n_hidden_nodes)
        self.hiddens = []
        self.n_hidden_layers = n_hidden_layers
        [self.hiddens.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))for _ in range(self.n_hidden_layers)]
        self.outlayer = torch.nn.Linear(n_hidden_nodes, 1)

        for i in range(self.n_hidden_layers):
            self.hiddens[i].weight.data.uniform_(-1, 1)
            self.hiddens[i].bias.data.uniform_(-20, 20)

    def forward(self, x):
        x = torch.relu(self.inlayer(torch.cat((x, torch.tensor(random.gauss(0, 1))))))
        for i in range(self.n_hidden_layers):
            x = torch.relu(self.hiddens[i](x))
        return self.outlayer(x)  # Adding the noise here instead of exogenous nodes to make implementation easier

def f_rand_net_noise(parents: List[str]):
    """Create a random function by initializing an MLP with random weights."""
    net = RandFct(len(parents), 10, 5)

    def f(**kwargs):
        # collect parents values
        vals = torch.tensor([kwargs[p] for p in parents]).float()
        out = net(vals)
        return float(out)
    return f

def f_linear_noise_normal(parents: List[str]):
    weights = {p: random.choice([random.uniform(-2, -0.5), random.uniform(0.5, 2)]) for p in parents}

    def f(**kwargs):
        mu = jnp.array([weights[p] * kwargs[p] for p in parents]).sum()
        return mu + np.random.normal(0, 0.5)

    return f


def f_linear_noise_beta(parents: List[str]):
    weights = {p: random.choice([random.uniform(-2, -0.5), random.uniform(0.5, 2)]) for p in parents}

    def f(**kwargs):
        mu = jnp.array([weights[p] * kwargs[p] for p in parents]).sum()
        return mu + np.random.beta(a=np.random.uniform(0.5, 2), b=np.random.uniform(0.5, 2))

    return f


def f_linear_noise_normalplus10(parents: List[str]):
    weights = {p: random.uniform(1, 3) for p in parents}

    def f(**kwargs):
        mu = jnp.array([weights[p] * kwargs[p] for p in parents]).sum()
        return mu + np.random.normal(0, 1) + 10

    return f


def f_quadratic(parents: List[str]):
    def f(**kwargs):
        mu = jnp.array([kwargs[p] ** 2 for p in parents]).sum()
        return mu + random.gauss(0, 1.5)

    return f


def f_cubic(parents: List[str]):
    def f(**kwargs):
        mu = jnp.array([kwargs[p] ** 3 for p in parents]).sum()
        return mu + 3 + random.gauss(0, 3)

    return f


def _cp_scm_worker(args):
    """Spawn-safe worker: receives a CausalPlayground graph object and creates
    its own SCM instance, so nothing with closures needs to be pickled.
    """
    graph, dgf_name, n_samples, intervention_value, seed = args
    import random as _random
    import numpy as _np
    import networkx as _nx
    from CausalPlayground import SCMGenerator
    # Import structural functions from this module (safe under spawn).
    from generate_data import (f_linear_noise_beta, f_linear_noise_normal,
                                f_rand_net, f_rand_net_noise,
                                f_linear_noise_normalplus10, f_quadratic, f_cubic)
    _random.seed(seed)
    _np.random.seed(seed % (2 ** 31))

    scm_gen = SCMGenerator(
        all_functions={
            'f_linear_noise_beta': f_linear_noise_beta,
            'f_linear_noise_normal': f_linear_noise_normal,
            'f_rand': f_rand_net,
            'f_rand_nonl': f_rand_net_noise,
            'f_linear_noise_normalplus10': f_linear_noise_normalplus10,
            'f_quadratic': f_quadratic,
            'f_cubic': f_cubic,
        },
        seed=seed,
    )
    scm = scm_gen.create_scm_from_graph(graph, [dgf_name],
                                        exo_distribution=None,
                                        exo_distribution_kwargs=None)

    adj = _np.array(_nx.adjacency_matrix(scm.create_graph()).todense())
    varnames = list(scm.endogenous_vars.keys())

    obs = [[float(v) for v in scm.get_next_sample()[0].values()]
           for _ in range(n_samples)]

    int_samples = []
    for varname in varnames:
        scm.do_interventions([(varname, (lambda: _random.gauss(intervention_value, 0.01), {}))])
        samples = [[float(v) for v in scm.get_next_sample()[0].values()]
                   for _ in range(n_samples)]
        scm.undo_interventions()
        int_samples.append(samples)

    return obs, int_samples, adj


def _save_metadata(args, dgf):
    """Save a JSON metadata file alongside the data files."""
    meta = {
        'cli_args': vars(args),
        'data_generating_function': {
            'name': args.data_generating_function,
            'source': inspect.getsource(dgf),
        },
    }
    meta_path = args.save_dir + '_metadata.json'
    os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata saved to {meta_path}')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='data/2var_linadd_beta_150_samples',
                        help='Filepath of where to save the data.')
    parser.add_argument('--samples', type=int, default=150, help='Number of samples to collect for each SCM.')
    parser.add_argument('--n-endo', type=int, default=2, help='Number of endogenous variables.')
    parser.add_argument('--n-exo', type=int, default=0, help='Number of exogenous variables.')
    parser.add_argument('--n-graphs', type=int, default=100, help='Number of graphs to generate.')
    parser.add_argument('--n-scms-per-graph', type=int, default=500, help='Number of randomly sampled SCMs '
                                                                        'to generate data for each graph structure.')
    parser.add_argument('--allow-exo-confounders', type=bool, default=False, help='Allow the generation of'
                                                                                  'unobserved, exogenous confounders if'
                                                                                  'set to True')
    parser.add_argument('--data-mode', type=str, default='interventional', help='"observational" or "interventional"')
    parser.add_argument('--train-fraction', type=float, default=0.8, help='Fraction of data used for training')
    parser.add_argument('--val-fraction', type=float, default=0.1, help='Fraction of data used for validation (remainder goes to test)')
    parser.add_argument('--data-generating-function', type=str, default='f_linear_noise_beta',
                        help='Name of the structural function to use (f_linear_noise_beta, f_linear_noise_normal, f_rand, f_rand_nonl)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_params()

    available_functions = {
        'f_linear_noise_beta': f_linear_noise_beta,
        'f_linear_noise_normal': f_linear_noise_normal,
        'f_rand': f_rand_net,
        'f_rand_nonl': f_rand_net_noise,
        'f_linear_noise_normalplus10': f_linear_noise_normalplus10,
        'f_quadratic': f_quadratic,
        'f_cubic': f_cubic,
    }
    if args.data_generating_function not in available_functions:
        raise ValueError(f'Unknown data-generating-function "{args.data_generating_function}". '
                         f'Choose from: {list(available_functions.keys())}')
    dgf = available_functions[args.data_generating_function]

    max_distinct = CausalGraphSetGenerator.max_n_dags(args.n_endo)
    n_graphs = min(args.n_graphs, max_distinct)
    if n_graphs < args.n_graphs:
        print(f'Warning: only {max_distinct} distinct DAGs exist for {args.n_endo} variables; '
              f'reducing n_graphs from {args.n_graphs} to {n_graphs}.')
    graph_set_gen = CausalGraphSetGenerator(args.n_endo, args.n_exo, args.allow_exo_confounders)
    graph_set_gen.generate(n_graphs, method='ER', p=0.3)
    distinct_graphs = graph_set_gen.graphs
    # Build the flat list of (graph, scm_instance_index) pairs.
    # SCMs are NOT created in the main process — workers create them from the
    # graph object to avoid pickling closures (incompatible with spawn).
    graph_list = [graph
                  for graph in distinct_graphs
                  for _ in range(args.n_scms_per_graph)]
    n_total = len(graph_list)

    assert args.train_fraction + args.val_fraction < 1.0, \
        '--train-fraction + --val-fraction must be less than 1.0'
    train_end = int(args.train_fraction * n_total)
    val_end = train_end + int(args.val_fraction * n_total)

    if args.data_mode == "observational":
        # Observational mode: create SCMs in main process (no parallelism needed).
        scm_gen = SCMGenerator(all_functions={'f_rand': f_rand_net, 'f_linear_noise_beta': f_linear_noise_beta, 'f_linear_noise_normal': f_linear_noise_normal, 'f_rand_nonl': f_rand_net_noise}, seed=42)
        scms = [scm_gen.create_scm_from_graph(g, [args.data_generating_function], exo_distribution=None, exo_distribution_kwargs=None) for g in graph_list]
        data = jnp.array([jnp.array([jnp.array(jax.tree.leaves(scms[i].get_next_sample()[0])) for j in range(args.samples)]) for i in range(len(scms))])
        jnp.savez(args.save_dir, data=data)
        _save_metadata(args, dgf)
    elif args.data_mode == "interventional":
        key, permutationkey = jax.random.split(jax.random.PRNGKey(42))

        worker_args = [(graph, args.data_generating_function, args.samples, 5.0, idx)
                       for idx, graph in enumerate(graph_list)]

        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))
        chunksize = max(1, len(worker_args) // (n_workers * 4))
        print(f'Sampling {n_total} SCMs with {n_workers} workers...')
        mp_ctx = __import__('multiprocessing').get_context('spawn')
        with mp_ctx.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_cp_scm_worker, worker_args, chunksize=chunksize),
                total=n_total, desc='Sampling SCMs',
                miniters=n_total // 20, mininterval=30,
                file=__import__('sys').stdout, dynamic_ncols=False,
            ))

        input_data_np = np.array([r[0] for r in results])    # (n_scms, n_samples, n_vars)
        target_data_np = np.array([[r[1]] for r in results]) # (n_scms, 1, n_targets, n_samples, n_vars)
        graphs_np = np.array([r[2] for r in results])        # (n_scms, n_vars, n_vars)

        permutation = jax.random.permutation(permutationkey, len(results))
        input_data = jnp.array(input_data_np)[permutation]
        output_data = jnp.array(target_data_np)[permutation].swapaxes(0, 1)
        graphs = jnp.array(graphs_np)[permutation]

        base_targets = [jax.nn.one_hot(i, num_classes=args.n_endo + 1)[1:]
                        for i in range(args.n_endo + 1)]
        targets = jnp.repeat(
            jnp.repeat(jnp.array(base_targets)[jnp.newaxis, 1:], n_total, axis=0)[:, jnp.newaxis],
            len(output_data), axis=1
        )[permutation]

        # Save three files: `<save_dir>_train.npz`, `<save_dir>_val.npz`, `<save_dir>_test.npz`
        #   - `input`: shape (n_scms, n_samples, n_vars)
        #   - `target_data`: shape (n_interventions, n_scms, n_targets, n_samples, n_vars)
        #   - `graphs`: shape (n_scms, n_vars, n_vars)
        #   - `targets`: shape (n_scms, n_interventions, n_targets, n_vars)
        jnp.savez(args.save_dir + '_train.npz', input=input_data[:train_end], target_data=output_data[:, :train_end],
                  graphs=graphs[:train_end], targets=targets[:train_end])
        jnp.savez(args.save_dir + '_val.npz', input=input_data[train_end:val_end], target_data=output_data[:, train_end:val_end],
                  graphs=graphs[train_end:val_end], targets=targets[train_end:val_end])
        jnp.savez(args.save_dir + '_test.npz', input=input_data[val_end:], target_data=output_data[:, val_end:],
                  graphs=graphs[val_end:], targets=targets[val_end:])
        _save_metadata(args, dgf)
