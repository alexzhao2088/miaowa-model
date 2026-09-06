#!/usr/bin/env python3
"""
XGBoost Booster -> ONNX TreeEnsembleRegressor 直接导出（不依赖 onnxmltools）。
- 输入名固定 float_input，shape [None, n_features]，顺序即特征契约
- 分裂语义 BRANCH_LT（与 XGBoost 的 < split_condition 一致），missing 方向按 JSON dump 的 missing 字段
- base_values 固定 [0.0]，训练侧必须 XGBRegressor(base_score=0) 保证数值一致
- metadata_props 写入 feature_spec_version，后端启动校验用
"""
import json
import tempfile

import numpy as np
import onnx
from onnx import helper, TensorProto

OPSET_ML = 3  # ai.onnx.ml 算子集版本，onnxruntime 1.16+ 均支持


def _parse_tree(node, tree_id, acc):
    """递归遍历 XGBoost JSON dump 的一棵树，累积 ONNX TreeEnsemble 扁平字段。"""
    node_id = node["nodeid"]
    if "leaf" in node:
        acc["nodes_modes"].append("LEAF")
        acc["nodes_featureids"].append(0)
        acc["nodes_values"].append(0.0)
        acc["nodes_truenodeids"].append(0)
        acc["nodes_falsenodeids"].append(0)
        acc["nodes_missing"].append(0)
        acc["target_treeids"].append(tree_id)
        acc["target_nodeids"].append(node_id)
        acc["target_weights"].append(float(node["leaf"]))
        return
    feat_id = int(node["split"].lstrip("f"))
    yes_id, no_id, missing_id = node["yes"], node["no"], node["missing"]
    acc["nodes_modes"].append("BRANCH_LT")
    acc["nodes_featureids"].append(feat_id)
    acc["nodes_values"].append(float(node["split_condition"]))
    acc["nodes_truenodeids"].append(yes_id)
    acc["nodes_falsenodeids"].append(no_id)
    acc["nodes_missing"].append(1 if missing_id == yes_id else 0)
    for child in node["children"]:
        _parse_tree(child, tree_id, acc)


def booster_to_onnx(booster, n_features: int, feature_spec_version: str,
                    model_version: str = "") -> onnx.ModelProto:
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]

    acc = {k: [] for k in ("nodes_modes", "nodes_featureids", "nodes_values",
                           "nodes_truenodeids", "nodes_falsenodeids", "nodes_missing",
                           "target_treeids", "target_nodeids", "target_weights")}
    nodes_treeids, nodes_nodeids = [], []
    for tree_id, tree in enumerate(trees):
        # walk 与 _parse_tree 同为前序遍历，保证 nodes_* 与 acc 字段按同一节点序列对齐
        def walk(n):
            nodes_treeids.append(tree_id)
            nodes_nodeids.append(n["nodeid"])
            if "leaf" not in n:
                for c in n["children"]:
                    walk(c)
        walk(tree)
        _parse_tree(tree, tree_id, acc)

    # 单输出回归：target_ids 全 0，与 target_weights 等长
    target_ids = [0] * len(acc["target_weights"])

    tree_node = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=["float_input"],
        outputs=["score"],
        domain="ai.onnx.ml",
        name="xgb_risk_regressor",
        n_targets=1,
        aggregate_function="SUM",
        post_transform="NONE",
        base_values=[0.0],
        nodes_treeids=nodes_treeids,
        nodes_nodeids=nodes_nodeids,
        nodes_featureids=acc["nodes_featureids"],
        nodes_modes=acc["nodes_modes"],
        nodes_values=acc["nodes_values"],
        nodes_truenodeids=acc["nodes_truenodeids"],
        nodes_falsenodeids=acc["nodes_falsenodeids"],
        nodes_missing_value_tracks_true=acc["nodes_missing"],
        target_treeids=acc["target_treeids"],
        target_nodeids=acc["target_nodeids"],
        target_ids=target_ids,
        target_weights=acc["target_weights"],
    )
    graph = helper.make_graph(
        [tree_node], "miaowa_risk_model",
        inputs=[helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, n_features])],
        outputs=[helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("ai.onnx.ml", OPSET_ML),
                       helper.make_opsetid("", 17)],
        producer_name="miaowa-model/onnx_export.py",
    )
    model.ir_version = 8  # 兼容 onnxruntime（py 侧 1.17+ / Java 侧）均支持 ≤ IR 10
    for k, v in (("feature_spec_version", feature_spec_version),
                 ("model_version", model_version),
                 ("n_features", str(n_features))):
        entry = onnx.StringStringEntryProto()
        entry.key, entry.value = k, v
        model.metadata_props.append(entry)
    onnx.checker.check_model(model)
    return model


def verify_against_booster(model_proto, booster, sample_X: np.ndarray) -> float:
    """onnxruntime 推理与 booster.predict 对比，返回最大绝对误差（发布前自检）。"""
    import onnxruntime as ort
    import xgboost as xgb

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        onnx.save(model_proto, tmp.name)
        sess = ort.InferenceSession(tmp.name, providers=["CPUExecutionProvider"])
    pred_onnx = sess.run(None, {"float_input": sample_X.astype(np.float32)})[0].ravel()
    pred_xgb = booster.predict(xgb.DMatrix(sample_X.astype(np.float32)))
    return float(np.max(np.abs(pred_onnx - pred_xgb)))
