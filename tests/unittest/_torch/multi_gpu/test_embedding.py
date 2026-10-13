import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from torch import nn

import tensorrt_llm
from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def run_single_rank(tensor_parallel_size, single_rank_forward_func, input,
                    weights, vocab_size, hidden_size, dtype):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(input, vocab_size, hidden_size, dtype,
                                 tensor_parallel_size, rank, weights)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode
def column_embedding_forward(x, vocab_size, hidden_size, dtype,
                             tensor_parallel_size, tensor_parallel_rank,
                             weight):

    x = x.cuda()
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
    )
    embedding.load_weights([dict(weight=weight)])
    embedding.cuda()

    output = embedding.forward(x)

    # torch run
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
    )
    embedding.weight.data.copy_(weight)
    embedding.cuda()

    torch_output = embedding.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def row_embedding_forward(x, vocab_size, hidden_size, dtype,
                          tensor_parallel_size, tensor_parallel_rank, weight):

    x = x.cuda()
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
        gather_output=True,
    )
    embedding.load_weights([dict(weight=weight)])
    embedding.cuda()

    output = embedding.forward(x)

    # torch run
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
    )
    embedding.weight.data.copy_(weight)
    embedding.cuda()

    torch_output = embedding.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def column_lm_head_forward(x, vocab_size, hidden_size, dtype,
                           tensor_parallel_size, tensor_parallel_rank, weight):

    x = x.cuda()
    lm_head = LMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    )
    lm_head.load_weights([dict(weight=weight)])
    lm_head.cuda()

    output = lm_head.forward(x)

    # torch run
    lm_head = nn.Linear(
        in_features=hidden_size,
        out_features=vocab_size,
        bias=False,
        dtype=dtype,
    )
    lm_head.weight.data.copy_(weight)
    lm_head.cuda()

    torch_output = lm_head.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output)


@torch.inference_mode
def row_lm_head_forward(x, vocab_size, hidden_size, dtype, tensor_parallel_size,
                        tensor_parallel_rank, weight):

    x = x.cuda()
    lm_head = LMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
        ),
        tensor_parallel_mode=TensorParallelMode.ROW,
    )
    lm_head.load_weights([dict(weight=weight)])
    lm_head.cuda()

    xs = torch.chunk(x, 2, dim=-1)
    output = lm_head.forward(xs[tensor_parallel_rank])

    # torch run
    lm_head = nn.Linear(
        in_features=hidden_size,
        out_features=vocab_size,
        bias=False,
        dtype=dtype,
    )
    lm_head.weight.data.copy_(weight)
    lm_head.cuda()

    torch_output = lm_head.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch_output, rtol=0.05, atol=0.05)


@torch.inference_mode
def lm_head_tp_in_adp_forward(x, vocab_size, hidden_size, dtype,
                              tensor_parallel_size, tensor_parallel_rank,
                              weight):
    # Attention-DP ranks own different numbers of rows: rank r owns 2r+1.
    rows_per_rank = [2 * r + 1 for r in range(tensor_parallel_size)]
    num_rows_per_rank = max(rows_per_rank)
    all_rows = torch.split(x.cuda(), rows_per_rank)
    rows = all_rows[tensor_parallel_rank]

    lm_head = LMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        dtype=dtype,
        mapping=Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=tensor_parallel_rank,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
        ),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        reduce_output=False,
    )
    lm_head.load_weights([dict(weight=weight)])
    lm_head.cuda()

    shards, mapping_lm_head_tp = lm_head.forward_lm_head_tp_in_adp(
        rows, num_rows_per_rank)
    logits = lm_head.forward_lm_head_tp_in_adp_full_vocab(
        rows, num_rows_per_rank)

    # torch run
    torch_lm_head = nn.Linear(
        in_features=hidden_size,
        out_features=vocab_size,
        bias=False,
        dtype=dtype,
    )
    torch_lm_head.weight.data.copy_(weight)
    torch_lm_head.cuda()

    # Every rank's rows, zero-padded, projected onto this rank's vocab shard;
    # the shard past the vocabulary is -inf.
    shard_size = (vocab_size + tensor_parallel_size - 1) // tensor_parallel_size
    shard_start = tensor_parallel_rank * shard_size
    stacked_rows = torch.cat([
        nn.functional.pad(r, (0, 0, 0, num_rows_per_rank - r.shape[0]))
        for r in all_rows
    ])
    stacked_logits = torch_lm_head.forward(stacked_rows)
    torch_shards = stacked_logits[:, shard_start:shard_start + shard_size]
    torch_shards = nn.functional.pad(torch_shards,
                                     (0, shard_size - torch_shards.shape[-1]),
                                     value=float("-inf"))
    torch_logits = torch_lm_head.forward(rows)

    # compare
    torch.cuda.synchronize()
    assert mapping_lm_head_tp.tp_size == tensor_parallel_size
    torch.testing.assert_close(shards, torch_shards)
    torch.testing.assert_close(logits, torch_logits)


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("vocab_size", [128, 127],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_column_embedding(vocab_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 10
    hidden_size = 16
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randint(0, vocab_size, (seq_len, ))
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, column_embedding_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden_size", [16, 15],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_row_embedding(hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 2
    vocab_size = 128
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randint(0, vocab_size, (seq_len, ))
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, row_embedding_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("vocab_size", [128, 127],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_column_lm_head(vocab_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 10
    hidden_size = 16
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randn((seq_len, hidden_size), dtype=dtype)
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, column_lm_head_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden_size", [16, 15],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_row_lm_head(hidden_size, mpi_pool_executor):
    torch.manual_seed(42)
    seq_len = 2
    vocab_size = 128
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    input = torch.randn((seq_len, hidden_size), dtype=dtype)
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, row_lm_head_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("vocab_size", [128, 127],
                         ids=["balanced", "unbalanced"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_lm_head_tp_in_adp(vocab_size, mpi_pool_executor):
    torch.manual_seed(42)
    hidden_size = 16
    dtype = torch.bfloat16
    tensor_parallel_size = mpi_pool_executor.num_workers
    num_rows = sum(2 * r + 1 for r in range(tensor_parallel_size))
    input = torch.randn((num_rows, hidden_size), dtype=dtype)
    weight = torch.randn((vocab_size, hidden_size), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, lm_head_tp_in_adp_forward, input, weight,
                vocab_size, hidden_size, dtype)] * 2))
    for r in results:
        assert r is True
