import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from infersim.errors import InputValidationError, UnsupportedModelError
from infersim.schema.model import ModelSpec


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "hf_configs"


def dense_config(**overrides):
    config = {
        "model_type": "example",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
    }
    config.update(overrides)
    return config


def moe_config(**overrides):
    config = dense_config(
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
    )
    config.update(overrides)
    return config


class ModelSpecTests(unittest.TestCase):
    def test_normalizes_dense_gqa(self):
        spec = ModelSpec.from_dict({
            "model_type": "example", "hidden_size": 4096,
            "num_hidden_layers": 32, "vocab_size": 32000,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 11008, "tie_word_embeddings": True,
        })
        self.assertEqual(spec.attention_kind, "gqa")
        self.assertEqual(spec.head_dim, 128)
        self.assertFalse(spec.is_moe)
        self.assertTrue(spec.tie_word_embeddings)

    def test_normalizes_dense_mha_and_defaults(self):
        spec = ModelSpec.from_dict(dense_config(num_key_value_heads=32))

        self.assertEqual(spec.attention_kind, "mha")
        self.assertEqual(spec.num_routed_experts, 0)
        self.assertEqual(spec.experts_per_token, 0)
        self.assertEqual(spec.num_full_attention_layers, 32)
        self.assertEqual(spec.num_linear_attention_layers, 0)
        self.assertFalse(spec.attention_output_gate)

    def test_normalizes_mqa_when_there_is_one_kv_head(self):
        spec = ModelSpec.from_dict(dense_config(num_key_value_heads=1))
        self.assertEqual(spec.attention_kind, "mqa")

    def test_defaults_kv_heads_to_attention_heads(self):
        config = dense_config()
        del config["num_key_value_heads"]

        spec = ModelSpec.from_dict(config)

        self.assertEqual(spec.num_key_value_heads, 32)
        self.assertEqual(spec.attention_kind, "mha")

    def test_normalizes_moe_aliases(self):
        spec = ModelSpec.from_dict({
            "model_type": "example_moe", "hidden_size": 1024,
            "num_hidden_layers": 8, "vocab_size": 4096,
            "num_attention_heads": 16, "num_key_value_heads": 4,
            "num_experts": 64, "num_experts_per_token": 4,
            "moe_intermediate_size": 256,
        })
        self.assertEqual(spec.num_routed_experts, 64)
        self.assertEqual(spec.experts_per_token, 4)
        self.assertEqual(spec.intermediate_size, 256)
        self.assertTrue(spec.is_moe)

    def test_normalizes_moe_mla_and_optional_dimensions(self):
        spec = ModelSpec.from_dict({
            "model_type": "mla_moe",
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "vocab_size": 129280,
            "num_attention_heads": 128,
            "num_routed_experts": 256,
            "num_experts_per_tok": 8,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        })

        self.assertEqual(spec.attention_kind, "mla")
        self.assertEqual(spec.intermediate_size, 2048)
        self.assertEqual(spec.q_lora_rank, 1536)
        self.assertEqual(spec.kv_lora_rank, 512)
        self.assertEqual(spec.qk_nope_head_dim, 128)
        self.assertEqual(spec.qk_rope_head_dim, 64)
        self.assertEqual(spec.v_head_dim, 128)

    def test_uses_explicit_head_dim(self):
        spec = ModelSpec.from_dict(
            dense_config(hidden_size=2048, num_attention_heads=16, head_dim=256)
        )
        self.assertEqual(spec.head_dim, 256)

    def test_unwraps_text_config_and_reports_nested_paths(self):
        root = {"model_type": "wrapper", "text_config": dense_config()}
        spec = ModelSpec.from_dict(root)
        self.assertEqual(spec.model_type, "example")

        del root["text_config"]["hidden_size"]
        with self.assertRaisesRegex(InputValidationError, "text_config.hidden_size"):
            ModelSpec.from_dict(root)

    def test_normalizes_shared_expert_total_size_per_expert(self):
        spec = ModelSpec.from_dict({
            **dense_config(),
            "num_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1024,
            "num_shared_experts": 2,
            "shared_expert_intermediate_size": 2048,
        })
        self.assertEqual(spec.num_shared_experts, 2)
        self.assertEqual(spec.shared_expert_intermediate_size, 1024)

    def test_shared_size_without_count_implies_one_expert(self):
        spec = ModelSpec.from_dict({
            **dense_config(),
            "num_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 512,
        })
        self.assertEqual(spec.num_shared_experts, 1)
        self.assertEqual(spec.shared_expert_intermediate_size, 512)

    def test_normalizes_hybrid_linear_attention(self):
        spec = ModelSpec.from_dict({
            **dense_config(num_hidden_layers=12),
            "num_full_attention_layers": 3,
            "num_linear_attention_layers": 9,
            "attn_output_gate": True,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_value_head_dim": 128,
            "linear_num_value_heads": 32,
        })

        self.assertEqual(spec.num_full_attention_layers, 3)
        self.assertEqual(spec.num_linear_attention_layers, 9)
        self.assertTrue(spec.attention_output_gate)
        self.assertEqual(spec.linear_conv_kernel_dim, 4)
        self.assertEqual(spec.linear_key_head_dim, 128)
        self.assertEqual(spec.linear_num_key_heads, 16)
        self.assertEqual(spec.linear_value_head_dim, 128)
        self.assertEqual(spec.linear_num_value_heads, 32)

    def test_accepts_canonical_attention_output_gate(self):
        spec = ModelSpec.from_dict(dense_config(attention_output_gate=True))
        self.assertTrue(spec.attention_output_gate)

    def test_accepts_matching_alias_values(self):
        spec = ModelSpec.from_dict(moe_config(
            num_experts=8,
            n_routed_experts=8,
            num_experts_per_token=2,
            num_shared_experts=1,
            n_shared_experts=1,
            shared_expert_intermediate_size=1024,
            attention_output_gate=True,
            attn_output_gate=True,
        ))

        self.assertEqual(spec.num_routed_experts, 8)
        self.assertEqual(spec.experts_per_token, 2)
        self.assertEqual(spec.num_shared_experts, 1)
        self.assertTrue(spec.attention_output_gate)

    def test_rejects_conflicting_routed_expert_aliases(self):
        with self.assertRaisesRegex(
            InputValidationError,
            "num_experts.*conflicts with num_routed_experts",
        ):
            ModelSpec.from_dict(moe_config(num_experts=16))

    def test_rejects_invalid_secondary_routed_expert_alias(self):
        with self.assertRaisesRegex(
            InputValidationError, "num_experts.*must be an integer"
        ):
            ModelSpec.from_dict(moe_config(num_experts="8"))

    def test_rejects_conflicting_selected_expert_aliases(self):
        with self.assertRaisesRegex(
            InputValidationError,
            "num_experts_per_token.*conflicts with num_experts_per_tok",
        ):
            ModelSpec.from_dict(moe_config(num_experts_per_token=3))

    def test_rejects_conflicting_shared_expert_aliases(self):
        with self.assertRaisesRegex(
            InputValidationError,
            "n_shared_experts.*conflicts with num_shared_experts",
        ):
            ModelSpec.from_dict(moe_config(
                num_shared_experts=1,
                n_shared_experts=2,
                shared_expert_intermediate_size=1024,
            ))

    def test_rejects_conflicting_attention_gate_aliases(self):
        with self.assertRaisesRegex(
            InputValidationError,
            "attn_output_gate.*conflicts with attention_output_gate",
        ):
            ModelSpec.from_dict(dense_config(
                attention_output_gate=True,
                attn_output_gate=False,
            ))

    def test_normalizes_qwen_next_fixture_hybrid_layers(self):
        path = FIXTURE_DIR / "qwen3-next-80B-A3B_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))

        spec = ModelSpec.from_dict(config)

        self.assertEqual(spec.num_full_attention_layers, 12)
        self.assertEqual(spec.num_linear_attention_layers, 36)

    def test_normalizes_layer_types(self):
        spec = ModelSpec.from_dict({
            **dense_config(num_hidden_layers=4),
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_value_head_dim": 128,
            "linear_num_value_heads": 32,
        })

        self.assertEqual(spec.num_full_attention_layers, 1)
        self.assertEqual(spec.num_linear_attention_layers, 3)

    def test_rejects_string_layer_types(self):
        with self.assertRaisesRegex(InputValidationError, "layer_types"):
            ModelSpec.from_dict(dense_config(layer_types="full_attention"))

    def test_rejects_wrong_layer_types_length(self):
        with self.assertRaisesRegex(InputValidationError, "layer_types"):
            ModelSpec.from_dict(dense_config(layer_types=["full_attention"]))

    def test_rejects_unknown_layer_type_with_index_path(self):
        with self.assertRaisesRegex(InputValidationError, r"layer_types\[1\]"):
            ModelSpec.from_dict({
                **dense_config(num_hidden_layers=2),
                "layer_types": ["full_attention", "sliding_attention"],
            })

    def test_rejects_conflicting_layer_types_and_interval(self):
        with self.assertRaisesRegex(
            InputValidationError, "full_attention_interval.*conflicts"
        ):
            ModelSpec.from_dict({
                **dense_config(num_hidden_layers=4),
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
                "full_attention_interval": 2,
            })

    def test_rejects_explicit_layer_count_conflicting_with_interval(self):
        with self.assertRaisesRegex(
            InputValidationError, "num_full_attention_layers.*conflicts"
        ):
            ModelSpec.from_dict({
                **dense_config(num_hidden_layers=4),
                "num_full_attention_layers": 1,
                "num_linear_attention_layers": 3,
                "full_attention_interval": 2,
            })

    def test_normalizes_deepseek_v32_fixture_aliases(self):
        path = FIXTURE_DIR / "deepseek_v3.2_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))

        spec = ModelSpec.from_dict(config)

        self.assertEqual(spec.num_routed_experts, 256)
        self.assertEqual(spec.num_shared_experts, 1)
        self.assertEqual(spec.attention_kind, "mla")

    def test_mla_does_not_require_q_lora_rank(self):
        spec = ModelSpec.from_dict(dense_config(
            kv_lora_rank=64,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        ))
        self.assertEqual(spec.attention_kind, "mla")
        self.assertIsNone(spec.q_lora_rank)

    def test_mla_requires_all_head_dimensions(self):
        mla_dimensions = {
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 64,
        }
        for missing_field in mla_dimensions:
            with self.subTest(missing_field=missing_field):
                config = dense_config(kv_lora_rank=64, **mla_dimensions)
                del config[missing_field]
                with self.assertRaisesRegex(InputValidationError, missing_field):
                    ModelSpec.from_dict(config)

    def test_linear_attention_requires_all_dimensions(self):
        linear_dimensions = {
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_value_head_dim": 128,
            "linear_num_value_heads": 32,
        }
        for missing_field in linear_dimensions:
            with self.subTest(missing_field=missing_field):
                config = dense_config(
                    num_hidden_layers=2,
                    num_full_attention_layers=1,
                    num_linear_attention_layers=1,
                    **linear_dimensions,
                )
                del config[missing_field]
                with self.assertRaisesRegex(InputValidationError, missing_field):
                    ModelSpec.from_dict(config)

    def test_rejects_nested_encoder_decoder(self):
        with self.assertRaisesRegex(
            UnsupportedModelError,
            "text_config.is_encoder_decoder.*encoder-decoder",
        ):
            ModelSpec.from_dict({
                "text_config": dense_config(is_encoder_decoder=True),
            })

    def test_is_immutable(self):
        spec = ModelSpec.from_dict(dense_config())
        with self.assertRaises(FrozenInstanceError):
            spec.hidden_size = 1

    def test_rejects_encoder_decoder(self):
        with self.assertRaisesRegex(UnsupportedModelError, "encoder-decoder"):
            ModelSpec.from_dict({"model_type": "t5", "is_encoder_decoder": True})

    def test_rejects_multimodal_before_unwrapping_text_config(self):
        with self.assertRaisesRegex(UnsupportedModelError, "multimodal"):
            ModelSpec.from_dict({
                "vision_config": {},
                "text_config": dense_config(),
            })

    def test_rejects_vision_config_dict(self):
        with self.assertRaisesRegex(UnsupportedModelError, "multimodal"):
            ModelSpec.from_dict({"vision_config_dict": None})

    def test_reports_missing_field_path(self):
        with self.assertRaisesRegex(InputValidationError, "hidden_size"):
            ModelSpec.from_dict({"model_type": "broken"})

    def test_rejects_non_mapping_input_at_root(self):
        with self.assertRaisesRegex(InputValidationError, r"\$: expected a mapping"):
            ModelSpec.from_dict([])

    def test_rejects_non_mapping_text_config(self):
        with self.assertRaisesRegex(
            InputValidationError, "text_config: expected a mapping"
        ):
            ModelSpec.from_dict({"text_config": []})

    def test_rejects_nondivisible_head_dimension_when_not_explicit(self):
        with self.assertRaisesRegex(InputValidationError, "hidden_size"):
            ModelSpec.from_dict(
                dense_config(hidden_size=1025, num_attention_heads=16)
            )

    def test_rejects_invalid_head_relationship(self):
        with self.assertRaisesRegex(InputValidationError, "num_key_value_heads"):
            ModelSpec.from_dict(
                dense_config(
                    hidden_size=3072,
                    num_attention_heads=12,
                    num_key_value_heads=5,
                )
            )

    def test_rejects_more_selected_than_routed_experts(self):
        with self.assertRaisesRegex(InputValidationError, "num_experts_per_tok"):
            ModelSpec.from_dict({
                **dense_config(),
                "num_routed_experts": 4,
                "num_experts_per_tok": 5,
                "moe_intermediate_size": 1024,
            })

    def test_rejects_single_routed_expert(self):
        with self.assertRaisesRegex(InputValidationError, "num_routed_experts"):
            ModelSpec.from_dict({
                **dense_config(),
                "num_routed_experts": 1,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 1024,
            })

    def test_rejects_moe_intermediate_size_on_dense_model(self):
        with self.assertRaisesRegex(InputValidationError, "moe_intermediate_size"):
            ModelSpec.from_dict(
                dense_config(moe_intermediate_size=1024)
            )

    def test_rejects_positive_expert_selection_on_dense_model(self):
        with self.assertRaisesRegex(
            InputValidationError, "num_experts_per_token"
        ):
            ModelSpec.from_dict(
                dense_config(num_experts_per_token=1)
            )

    def test_rejects_explicit_zero_expert_selection_on_dense_model(self):
        with self.assertRaisesRegex(
            InputValidationError, "num_experts_per_tok"
        ):
            ModelSpec.from_dict(
                dense_config(num_experts_per_tok=0)
            )

    def test_rejects_explicit_zero_shared_experts(self):
        with self.assertRaisesRegex(InputValidationError, "num_shared_experts"):
            ModelSpec.from_dict({
                **dense_config(),
                "num_routed_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 1024,
                "num_shared_experts": 0,
            })

    def test_rejects_negative_shared_expert_size(self):
        with self.assertRaisesRegex(
            InputValidationError, "shared_expert_intermediate_size"
        ):
            ModelSpec.from_dict({
                **dense_config(),
                "num_routed_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": -64,
            })

    def test_validates_shared_size_when_shared_count_is_also_invalid(self):
        with self.assertRaisesRegex(
            InputValidationError, "shared_expert_intermediate_size"
        ):
            ModelSpec.from_dict({
                **dense_config(),
                "num_routed_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 1024,
                "num_shared_experts": 0,
                "shared_expert_intermediate_size": -64,
            })

    def test_rejects_shared_experts_on_dense_model(self):
        with self.assertRaisesRegex(InputValidationError, "num_shared_experts"):
            ModelSpec.from_dict({
                **dense_config(),
                "num_shared_experts": 1,
                "shared_expert_intermediate_size": 1024,
            })

    def test_rejects_indivisible_shared_expert_size(self):
        with self.assertRaisesRegex(
            InputValidationError, "shared_expert_intermediate_size"
        ):
            ModelSpec.from_dict({
                **dense_config(),
                "num_routed_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 1024,
                "num_shared_experts": 3,
                "shared_expert_intermediate_size": 1024,
            })

    def test_rejects_mismatched_hybrid_layer_counts(self):
        with self.assertRaisesRegex(InputValidationError, "num_hidden_layers"):
            ModelSpec.from_dict({
                **dense_config(num_hidden_layers=12),
                "num_full_attention_layers": 4,
                "num_linear_attention_layers": 9,
            })

    def test_rejects_nonpositive_optional_dimension(self):
        with self.assertRaisesRegex(InputValidationError, "q_lora_rank"):
            ModelSpec.from_dict(dense_config(q_lora_rank=0))

    def test_rejects_non_boolean_flags(self):
        with self.assertRaisesRegex(InputValidationError, "tie_word_embeddings"):
            ModelSpec.from_dict(dense_config(tie_word_embeddings=1))


if __name__ == "__main__":
    unittest.main()
