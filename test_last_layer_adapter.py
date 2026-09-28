"""Comprehensive test and validation script for last-layer adapter architecture.

Tests:
1. Model instantiation with adapter in the last layer only (layer4).
2. Parameter distribution, trainable parameter counts, and freezing verification.
3. Forward pass with bi-temporal images for each domain (LEVIR and WHU).
4. Backward pass, loss computation, and gradient isolation.
5. Domain parameter freezing and isolation (LEVIR vs WHU).
6. Decoder adapter flexibility (guided, simple, and none).
7. Comparison table between all-layers vs last-layer configurations.
"""

import sys
import torch
import torch.nn as nn

from src.models.factory import build_change_detection_model, print_architecture
from src.utils.helpers import domain_parameters, freeze_domain, shared_parameters


def test_last_layer_adapter():
    print("=" * 80)
    print("TEST 1: Instantiation with adapters in last layer only (layer4)")
    print("=" * 80)

    domains = ["LEVIR", "WHU"]
    model = build_change_detection_model(
        domain_list=domains,
        adapter_stages=["layer4"],
        adapter_type="guided",
        decoder_adapter_type="simple",
    )
    print_architecture(
        mode="multi",
        adapter_type="guided",
        guided_granularity="stage",
        adapter_stages=["layer4"],
        decoder_adapter_type="simple",
    )

    # 1. Verify backbone domain_adapters structure
    for d in domains:
        ad = model.backbone.domain_adapters[d]
        stages_present = list(ad.keys())
        assert stages_present == ["layer4"], (
            f"Expected domain_adapters to only have ['layer4'], but found: {stages_present}"
        )
        print(f"[OK] Domain '{d}' domain_adapters keys strictly: {stages_present}")

    # 2. Verify frozen backbone layers
    for stage_name in ["layer1", "layer2", "layer3"]:
        stage = getattr(model.backbone, stage_name)
        for name, p in stage.named_parameters():
            assert not p.requires_grad, f"Parameter {stage_name}.{name} should be frozen (requires_grad=False)"
    print("[OK] Backbone layers 1, 2, 3 parameters strictly frozen (requires_grad=False)")

    # 3. Verify layer4 adapters are trainable
    for d in domains:
        for p in model.backbone.domain_adapters[d]["layer4"].parameters():
            assert p.requires_grad, f"Layer 4 adapter for domain {d} must have requires_grad=True"
    print("[OK] Backbone layer 4 adapters are trainable (requires_grad=True)")

    print("\n" + "=" * 80)
    print("TEST 2: Forward pass with bi-temporal dummy inputs (2, 3, 512, 512)")
    print("=" * 80)
    batch_size = 2
    x1 = torch.randn(batch_size, 3, 512, 512)
    x2 = torch.randn(batch_size, 3, 512, 512)

    for d in domains:
        logits, aux = model(x1, x2, domain=d)
        assert logits.shape == (batch_size, 1, 512, 512), f"Logits shape mismatch: {logits.shape}"
        assert aux is not None and aux.shape == (batch_size, 1, 512, 512), f"Aux shape mismatch: {aux.shape}"
        print(f"[OK] Domain {d} forward pass output shapes: logits={logits.shape}, aux={aux.shape}")

    # 4. Verify routing balance loss exists from guided adapter in layer4
    assert model.routing_balance is not None, "model.routing_balance should not be None for guided adapters"
    assert model.routing_balance.ndim == 0, f"model.routing_balance must be a scalar, got {model.routing_balance}"
    print(f"[OK] Guided adapter routing balance loss computed: {float(model.routing_balance):.6f}")

    print("\n" + "=" * 80)
    print("TEST 3: Backward pass and gradient flow verification")
    print("=" * 80)
    freeze_domain(model, "LEVIR")
    logits, aux = model(x1, x2, domain="LEVIR")
    target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    if aux is not None:
        loss = loss + 0.4 * nn.functional.binary_cross_entropy_with_logits(aux, target)
    if model.routing_balance is not None:
        loss = loss + 0.01 * model.routing_balance
    loss.backward()

    # Verify gradients for layer4 adapters
    levir_ad_grads = [p.grad for p in model.backbone.domain_adapters["LEVIR"]["layer4"].parameters() if p.grad is not None]
    assert len(levir_ad_grads) > 0, "No gradients accumulated in LEVIR layer4 adapters"
    print(f"[OK] Gradients successfully flowed to LEVIR layer4 adapters ({len(levir_ad_grads)} tensors with grad)")

    # Verify frozen layers 1..3 have NO gradients
    for stage_name in ["layer1", "layer2", "layer3"]:
        stage = getattr(model.backbone, stage_name)
        for name, p in stage.named_parameters():
            assert p.grad is None, f"Frozen parameter {stage_name}.{name} should have grad=None"
    print("[OK] Frozen layers 1..3 correctly have grad=None")

    # Verify WHU adapters have NO gradients when LEVIR was trained
    whu_ad_grads = [p.grad for p in model.backbone.domain_adapters["WHU"]["layer4"].parameters() if p.grad is not None]
    assert len(whu_ad_grads) == 0, f"WHU adapters leaked gradients during LEVIR step! {len(whu_ad_grads)}"
    print("[OK] WHU adapters have NO gradients during LEVIR training step (perfect domain isolation)")

    print("\n" + "=" * 80)
    print("TEST 4: Domain freezing and switching verification")
    print("=" * 80)
    freeze_domain(model, "WHU")
    # LEVIR should be frozen
    assert all(not p.requires_grad for p in model.backbone.domain_adapters["LEVIR"].parameters()), "LEVIR not frozen"
    # WHU should be trainable
    assert all(p.requires_grad for p in model.backbone.domain_adapters["WHU"].parameters()), "WHU not trainable"
    # Shared decoder params should remain trainable
    assert all(p.requires_grad for p in shared_parameters(model)), "Shared parameters not trainable"
    print("[OK] freeze_domain properly unfreezes WHU and freezes LEVIR, keeping shared decoder trainable")

    print("\n" + "=" * 80)
    print("TEST 5: Decoder adapter modes (simple, guided, none)")
    print("=" * 80)
    m_simple = build_change_detection_model(domains, adapter_stages=["layer4"], decoder_adapter_type="simple")
    m_guided = build_change_detection_model(domains, adapter_stages=["layer4"], decoder_adapter_type="guided")
    m_none = build_change_detection_model(domains, adapter_stages=["layer4"], decoder_adapter_type="none")

    print(f"[OK] Decoder adapter 'simple': {sum(p.numel() for p in m_simple.decoder.domain_parameters('LEVIR')):,} domain params/domain")
    print(f"[OK] Decoder adapter 'guided': {sum(p.numel() for p in m_guided.decoder.domain_parameters('LEVIR')):,} domain params/domain")
    print(f"[OK] Decoder adapter 'none':   {sum(p.numel() for p in m_none.decoder.domain_parameters('LEVIR')):,} domain params/domain (BN only)")

    print("\n" + "=" * 80)
    print("TEST 6: Parameter Count Comparison Table")
    print("=" * 80)
    configs = [
        ("Full Guided (all layers: 1..4)", ["layer1", "layer2", "layer3", "layer4"], "guided", "simple"),
        ("Last Layer Guided (layer4 only)", ["layer4"], "guided", "simple"),
        ("Full Simple (all layers: 1..4)", ["layer1", "layer2", "layer3", "layer4"], "simple", "simple"),
        ("Last Layer Simple (layer4 only)", ["layer4"], "simple", "simple"),
        ("Last Layer Guided + Decoder None", ["layer4"], "guided", "none"),
    ]

    header = f"{'Configuration':<35} | {'Backbone Ad/Dom':<16} | {'Decoder Dom/Dom':<16} | {'Trainable':<12} | {'Total Params':<12}"
    print(header)
    print("-" * len(header))
    for name, st, at, dat in configs:
        m = build_change_detection_model(domains, adapter_stages=st, adapter_type=at, decoder_adapter_type=dat)
        bb_dom = sum(p.numel() for p in m.backbone.domain_parameters("LEVIR"))
        dec_dom = sum(p.numel() for p in m.decoder.domain_parameters("LEVIR"))
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"{name:<35} | {bb_dom:<16,d} | {dec_dom:<16,d} | {trainable:<12,d} | {total:<12,d}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_last_layer_adapter()
