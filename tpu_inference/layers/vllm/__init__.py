# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tpu_inference.layers.vllm import backends as backends
from tpu_inference.layers.vllm import custom_ops as custom_ops
from tpu_inference.layers.vllm import ops as ops
from tpu_inference.layers.vllm import quantization as quantization
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


# NOTE: this function is the entry_points target for the vllm general plugin.
def register_layers():
    _register_keras_hub()


def _register_keras_hub():
    """Register KerasHub's neutral config and architecture with transformers/vLLM.

    KerasHub presets are served as native flax/nnx ``KerasNNXModel`` (selected by
    ``keras_hub_preset``), but vLLM still validates the config's ``model_type``
    (via transformers) and ``architectures`` (via its own registry) before the
    model loads. We register a neutral ``keras_hub`` / ``KerasHubForCausalLM``
    for both, so no real model family's name has to be borrowed. Best-effort
    and fully isolated: any failure logs and is skipped so it can never abort
    the shared ``register_layers`` hook for other models.
    """
    try:
        from keras_hub.src.vllm.hf_config import KERAS_HUB_ARCHITECTURE
        from keras_hub.src.vllm.hf_config import register_hf_config
        from keras_hub.src.vllm.nnx_adapter import KerasNNXModel

        from tpu_inference.models.common.model_loader import register_model

        # transformers side: resolve `model_type: keras_hub` in every worker.
        register_hf_config()
        # vLLM side: KerasNNXModel satisfies the loader's model interface, so
        # the helper wraps it to pass vLLM's arch validation. The real model is
        # still chosen by `keras_hub_preset` in `_get_model_architecture`.
        register_model(KERAS_HUB_ARCHITECTURE, KerasNNXModel)
    except ImportError:
        return  # keras_hub/flax not installed; nothing KerasHub to serve.
    except Exception:
        logger.exception(
            "KerasHub serving registration failed; KerasHub presets will not "
            "load, but other models are unaffected.")
