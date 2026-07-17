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
from vllm.tokenizers.registry import TokenizerRegistry

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
    """Register KerasHub's config and model class with transformers/vLLM.

    KerasHub presets are served by the native flax/nnx ``KerasHubForCausalLM``
    model class. Registration follows the standard mechanisms end to end:
    ``register_hf_config`` teaches transformers the ``keras_hub`` model_type,
    and ``register_model`` puts ``KerasHubForCausalLM`` in the model registry
    (and vLLM's), so the loader resolves it by architecture name exactly like
    any other model. The preset to serve rides in the config's
    ``keras_hub_preset`` field, which the model class reads.
    """
    # keras_hub is an optional dependency, and model_loader must not be
    # imported at module scope (see the inline-import NOTE in
    # `_get_model_architecture`) — so these resolve at registration time.
    from keras_hub.src.vllm.hf_config import (KERAS_HUB_ARCHITECTURE,
                                              register_hf_config)

    from tpu_inference.models.common.model_loader import register_model
    from tpu_inference.models.jax.keras_hub_model import KerasHubForCausalLM

    # transformers side: resolve `model_type: keras_hub` in every worker.
    register_hf_config()
    register_model(KERAS_HUB_ARCHITECTURE, KerasHubForCausalLM)
    # Tokenizer side: serve the preset's own KerasHub tokenizer
    # (tokenizer_mode="keras_hub"); registered by module path, resolved
    # lazily by whichever process loads the tokenizer.
    TokenizerRegistry.register("keras_hub", "keras_hub.src.vllm.tokenizer",
                               "KerasHubTokenizer")
