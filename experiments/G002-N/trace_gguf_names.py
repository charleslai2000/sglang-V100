import logging
import sys

sys.path.insert(0, "python")

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

logging.basicConfig(level=logging.WARNING)
path = "/data/t011/models/gemma4-quasar-q4_0/gemma-4-E4B-it-QUASAR-Q4_0.gguf"
args = ServerArgs(
    model_path=path,
    dtype="float16",
    quantization="gguf",
    trust_remote_code=True,
    device="cuda",
    disable_cuda_graph=True,
    mem_fraction_static=0.76,
    context_length=8192,
    enable_multimodal=False,
)
set_global_server_args_for_scheduler(args)
torch = __import__("torch")
torch.cuda.set_device(0)
init_distributed_environment(
    world_size=1,
    rank=0,
    local_rank=0,
    distributed_init_method="tcp://127.0.0.1:29735",
    backend="nccl",
)
initialize_model_parallel()
config = ModelConfig(
    path,
    trust_remote_code=True,
    dtype="float16",
    quantization="gguf",
    context_length=8192,
    enable_multimodal=False,
)
get_model_loader(LoadConfig(load_format="gguf"), config).load_model(
    model_config=config,
    device_config=DeviceConfig(device="cuda", gpu_id=0),
)
