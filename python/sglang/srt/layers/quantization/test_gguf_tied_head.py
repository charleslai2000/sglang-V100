import pytest
import torch
import gguf
from gguf.quants import dequantize
import numpy as np

from sglang.srt.layers.quantization.gguf import GGUFConfig, GGUFLinearMethod


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("qtype", [gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.Q4_0])
@torch.inference_mode()
def test_tied_gguf_head_uses_packed_rows_in_vocab_chunks(dtype, qtype):
    class Embedding(torch.nn.Module):
        pass

    vocab, hidden = 1024, 256
    if qtype == gguf.GGMLQuantizationType.F16:
        rows = torch.randn(vocab, hidden, dtype=dtype, device="cuda")
        expected_weight = rows
    else:
        values = torch.randn(vocab, hidden, dtype=torch.float32).numpy()
        packed_np = gguf.quants.quantize(values, qtype)
        rows = torch.from_numpy(np.array(packed_np, copy=True)).cuda()
        decoded = dequantize(packed_np, qtype)
        expected_weight = torch.from_numpy(np.asarray(decoded).copy()).to(dtype=dtype, device="cuda")
    embedding = Embedding()
    embedding.embedding_dim = hidden
    embedding.num_embeddings = vocab
    embedding._gguf_qweight = rows
    embedding._gguf_qweight_type = int(qtype)
    embedding.weight = torch.nn.Parameter(torch.zeros((vocab, hidden), dtype=dtype, device="cuda"), requires_grad=False)
    x = torch.randn(3, hidden, dtype=dtype, device="cuda")

    result = GGUFLinearMethod(GGUFConfig()).apply_tied_embedding(
        embedding, x, output_vocab_size=vocab - 7
    )
    expected = x @ expected_weight[: vocab - 7].T
    tolerance = 0.1 if dtype == torch.float16 else 0.5
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=0.1)
    assert result.shape == (3, vocab - 7)

