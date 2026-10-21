"""
Convert NPZ interventional data to HDF5 format for MACE-TNP training.

Flattens (SCM, intervention) pairs into individual examples and writes
them in chunked HDF5 files compatible with CITNP's MultipleFileDataset.

Usage:
    python convert_data_to_hdf5.py \
        --npz-train data/AAAI/8var_linadd_beta_train.npz \
        --npz-val   data/AAAI/8var_linadd_beta_val.npz \
        --npz-test  data/AAAI/8var_linadd_beta_test.npz \
        --output-dir data/mace_tnp/8var_linadd_beta \
        --examples-per-file 100
"""

import argparse
import os
import numpy as np
import h5py


def flatten_npz_to_examples(npz_path: str) -> list:
    """
    Load NPZ and yield one dict per (SCM, intervention distribution, target variable) triple.

    NPZ layout:
        input       : (n_scms, n_obs_samples, n_vars)
        target_data : (n_interventions, n_scms, n_targets, n_int_samples, n_vars)
        graphs      : (n_scms, n_vars, n_vars)
        targets     : (n_scms, n_interventions, n_targets, n_vars)

    Returns:
        list of dicts with keys:
            obs_data        : (n_obs_samples, n_vars)  float32
            int_data        : (n_int_samples, n_vars)  float32
            causal_graphs   : (n_vars, n_vars)         float32
            intvn_indices   : scalar int64
            variable_counts : scalar int64
    """
    with np.load(npz_path) as f:
        input_data  = f['input'].astype(np.float32)    # (n_scms, n_obs, n_vars)
        target_data = f['target_data'].astype(np.float32)  # (n_intvn, n_scms, n_targets, n_int, n_vars)
        graphs      = f['graphs'].astype(np.float32)   # (n_scms, n_vars, n_vars)
        targets     = f['targets'].astype(np.float32)  # (n_scms, n_intvn, n_targets, n_vars)

    n_scms          = input_data.shape[0]
    n_vars          = input_data.shape[2]
    n_interventions = target_data.shape[0]
    n_targets       = target_data.shape[2]

    examples = []
    for k in range(n_scms):
        for i in range(n_interventions):
            for t in range(n_targets):
                # target_data[i, k, t] -> (n_int_samples, n_vars)
                int_data = target_data[i, k, t]
                # intvn_index: argmax of the one-hot intervention mask for (scm k, intvn i, target t)
                intvn_idx = int(np.argmax(targets[k, i, t]))
                examples.append({
                    'obs_data':        input_data[k],       # (n_obs, n_vars)
                    'int_data':        int_data,             # (n_int, n_vars)
                    'causal_graphs':   graphs[k],            # (n_vars, n_vars)
                    'intvn_indices':   np.int64(intvn_idx),
                    'variable_counts': np.int64(n_vars),
                })

    return examples


def write_hdf5_chunk(examples: list, out_path: str) -> None:
    """
    Write a list of example dicts into a single HDF5 file.

    Each key is stored as a dataset with the example index on axis 0.
    Scalar keys (intvn_indices, variable_counts) become shape (N,).
    """
    N = len(examples)
    # Determine shapes from first example
    first = examples[0]
    shapes = {k: np.array(v).shape for k, v in first.items()}
    dtypes = {
        'obs_data':        np.float32,
        'int_data':        np.float32,
        'causal_graphs':   np.float32,
        'intvn_indices':   np.int64,
        'variable_counts': np.int64,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        # Pre-allocate datasets
        dsets = {}
        for key, shape in shapes.items():
            full_shape = (N,) + shape if shape else (N,)
            dsets[key] = f.create_dataset(key, shape=full_shape, dtype=dtypes[key])

        # Fill datasets
        for idx, ex in enumerate(examples):
            for key in shapes:
                val = np.array(ex[key])
                if val.shape:
                    dsets[key][idx] = val
                else:
                    dsets[key][idx] = val.item()


def convert_split(examples: list, split_dir: str, examples_per_file: int = 100) -> None:
    """
    Chunk examples into numbered HDF5 files and write to split_dir.

    Truncates examples to a multiple of examples_per_file so all chunks have
    equal size — required by MultipleFileDataset which assumes uniform chunk size.
    """
    os.makedirs(split_dir, exist_ok=True)
    # Truncate to exact multiple so every chunk has the same size
    n_usable = (len(examples) // examples_per_file) * examples_per_file
    if n_usable < len(examples):
        print(f"  Truncating {len(examples)} → {n_usable} examples "
              f"(divisible by {examples_per_file})")
    examples = examples[:n_usable]
    n = n_usable
    chunk_idx = 0
    for start in range(0, n, examples_per_file):
        chunk = examples[start:start + examples_per_file]
        out_path = os.path.join(split_dir, f'chunk_{chunk_idx:04d}.hdf5')
        write_hdf5_chunk(chunk, out_path)
        chunk_idx += 1
    print(f"  Wrote {chunk_idx} HDF5 chunks ({n} examples) → {split_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert NPZ SCM data to MACE-TNP HDF5 format')
    parser.add_argument('--npz-train',        required=True,  help='Path to training NPZ file')
    parser.add_argument('--npz-val',          default=None,   help='Path to validation NPZ file (optional; if not provided, split train 80/20)')
    parser.add_argument('--npz-test',         required=True,  help='Path to test NPZ file')
    parser.add_argument('--output-dir',       required=True,  help='Root output directory')
    parser.add_argument('--examples-per-file', type=int, default=256,
                        help='Number of examples per HDF5 chunk file')
    args = parser.parse_args()

    # --- Training split ---
    print(f"Loading train NPZ: {args.npz_train}")
    train_examples = flatten_npz_to_examples(args.npz_train)
    print(f"  Total training examples: {len(train_examples)}")

    # --- Validation split ---
    if args.npz_val:
        print(f"Loading val NPZ: {args.npz_val}")
        val_examples = flatten_npz_to_examples(args.npz_val)
        print(f"  Total val examples: {len(val_examples)}")
    else:
        # Split train 80/20
        split_idx = int(0.8 * len(train_examples))
        val_examples = train_examples[split_idx:]
        train_examples = train_examples[:split_idx]
        print(f"  Splitting train 80/20: train={len(train_examples)}, val={len(val_examples)}")

    # --- Test split ---
    print(f"Loading test NPZ: {args.npz_test}")
    test_examples = flatten_npz_to_examples(args.npz_test)
    print(f"  Total test examples: {len(test_examples)}")

    # --- Write splits ---
    print("Writing HDF5 chunks...")
    convert_split(train_examples, os.path.join(args.output_dir, 'train'), args.examples_per_file)
    convert_split(val_examples,   os.path.join(args.output_dir, 'val'),   args.examples_per_file)
    convert_split(test_examples,  os.path.join(args.output_dir, 'test'),  args.examples_per_file)

    print("\nDone! HDF5 structure:")
    print(f"  {args.output_dir}/train/chunk_XXXX.hdf5")
    print(f"  {args.output_dir}/val/chunk_XXXX.hdf5")
    print(f"  {args.output_dir}/test/chunk_XXXX.hdf5")
    print("\nVerify with:")
    print(f"  python -c \"import h5py, numpy as np; "
          f"f=h5py.File('{args.output_dir}/train/chunk_0000.hdf5','r'); "
          f"print({{k: np.array(f[k]).shape for k in f}})\"")


if __name__ == '__main__':
    main()
