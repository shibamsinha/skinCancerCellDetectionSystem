import pytest
import torch
from torch import nn

from skincancer.models import AdditionalConvLayers, build_model
from skincancer.models.googlenet import ReportGoogLeNet


@pytest.fixture(scope="module")
def modified(project_cfg):
    return build_model(project_cfg.model("modified_densenet201"), pretrained=False)


def test_modified_densenet201_additional_layers_match_specification(modified):
    add = modified.additional
    assert isinstance(add, AdditionalConvLayers)
    c11, c12, c2 = add.conv1_1.conv, add.conv1_2.conv, add.conv2.conv
    assert (c11.in_channels, c11.out_channels, c11.kernel_size) == (1920, 128, (3, 1))
    assert (c12.out_channels, c12.kernel_size) == (128, (1, 3))
    assert (c2.out_channels, c2.kernel_size) == (64, (3, 3))


def test_modified_densenet201_is_genuinely_modified(modified, project_cfg):
    plain = build_model(project_cfg.model("densenet201"), pretrained=False)
    assert plain.additional is None
    extra = {k for k in modified.state_dict() if k.startswith("additional.")}
    assert extra and not any(k.startswith("additional.") for k in plain.state_dict())
    features = modified.backbone[0]
    assert [n for n, _ in features.named_children() if n.startswith("denseblock")] == [f"denseblock{i}" for i in range(1, 5)]
    assert [n for n, _ in features.named_children() if n.startswith("transition")] == [f"transition{i}" for i in range(1, 4)]


def test_classification_head_structure(modified):
    h = modified.head
    assert isinstance(h.gap, nn.AdaptiveAvgPool2d) and isinstance(h.batch_norm, nn.BatchNorm1d)
    assert h.batch_norm.num_features == 64
    assert h.dense2.out_features == 2


def test_modified_densenet_forward_and_feature_map(modified):
    modified.eval()
    feats = {}

    def capture(module, inputs, output):  # must return None, or it would replace the output
        feats["conv2"] = output.shape

    hook = modified.additional.conv2.register_forward_hook(capture)
    with torch.no_grad():
        out = modified(torch.rand(2, 3, 224, 224))
    hook.remove()
    assert out.shape == (2, 2)
    assert feats["conv2"] == (2, 64, 7, 7)
    probs = modified.predict_proba(torch.rand(1, 3, 224, 224))
    assert torch.allclose(probs.sum(1), torch.ones(1))


def test_gradcam_target_is_conv2(modified):
    assert modified.gradcam_target_layer is modified.additional.conv2
    assert modified.gradcam_target_name == "additional.conv2"


def test_phase_freezing(modified):
    modified.set_trainable("new_layers")
    assert not any(p.requires_grad for p in modified.backbone.parameters())
    assert all(p.requires_grad for p in modified.head.parameters())
    assert all(p.requires_grad for p in modified.additional.parameters())
    modified.set_trainable("last_block")
    feats = modified.backbone[0]
    assert all(p.requires_grad for p in feats.denseblock4.parameters())
    assert not any(p.requires_grad for p in feats.denseblock3.parameters())
    modified.set_trainable("all")
    assert all(p.requires_grad for p in modified.parameters())


def test_backbone_batchnorm_stays_in_inference_mode(project_cfg):
    m = build_model(project_cfg.model("modified_densenet201"), pretrained=True)
    m.train()
    bns = [x for x in m.backbone.modules() if isinstance(x, nn.BatchNorm2d)]
    assert bns and not any(bn.training for bn in bns)
    assert m.head.batch_norm.training  # the new head's BN still trains


@pytest.mark.parametrize("name", ["densenet201", "mobilenet_v2", "resnet50", "xception", "googlenet"])
def test_comparison_models_forward(project_cfg, name):
    cfg = project_cfg.model(name)
    m = build_model(cfg, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, cfg.input_size, cfg.input_size))
    assert out.shape == (2, 2)
    assert isinstance(m.gradcam_target_layer, nn.Module)


def test_googlenet_matches_notebook():
    m = ReportGoogLeNet(64)
    assert m.inception_1.out_channels == 256 and m.inception_2.out_channels == 480
    assert m.classifier[1].in_features == 480 * 4 * 4
    assert m.classifier[1].out_features == 512 and m.classifier[3].p == 0.5
    with pytest.raises(ValueError):
        m.set_trainable("new_layers")


def test_unknown_architecture_rejected(project_cfg):
    cfg = project_cfg.model("resnet50")
    bad = type(cfg)(**{**cfg.__dict__, "architecture": "vgg16"})
    with pytest.raises(ValueError):
        build_model(bad)
