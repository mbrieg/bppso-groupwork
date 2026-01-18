import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd

@dataclass
class TreeNode:
    is_leaf: bool
    prediction: Optional[Any] = None

    distribution: Optional[Dict[Any, int]] = None
    num_samples: int = 0

    split_attr: Optional[str] = None
    split_threshold: Optional[float] = None

    branches: Optional[Dict[Any, "TreeNode"]] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class C45DecisionTree:
    """
    Minimal C4.5-like decision tree with:
      - Gain Ratio
      - nominal multi-way, numeric binary split
      - missing handling (simplified)
      - sparse feature dropping via min_non_nan_ratio
    """

    def __init__(
        self,
        attribute_types: Dict[str, str],
        min_samples_split: int = 2,
        max_depth: Optional[int] = None,
        min_non_nan_ratio: float = 0.2,
    ):
        self.attribute_types = dict(attribute_types)
        self.min_samples_split = max(2, int(min_samples_split))
        self.max_depth = max_depth
        self.min_non_nan_ratio = float(min_non_nan_ratio)

        self.root: Optional[TreeNode] = None
        self._classes_: Optional[List[Any]] = None

        self.attribute_types_: Dict[str, str] = {}
        self.impute_values: Dict[str, Any] = {}
        
        # Internal counter for graphviz export
        self.node_counter = 0

    #  PUBLIC API

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, List[Any]]) -> None:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        self._classes_ = list(y.value_counts().index)

        if len(self._classes_) <= 1:
            num_samples = len(y)
            prediction = self._classes_[0] if self._classes_ else None
            distribution = y.value_counts().to_dict()
            self.root = TreeNode(
                is_leaf=True, 
                prediction=prediction, 
                distribution=distribution, 
                num_samples=num_samples
            )
            return

        declared_cols = [c for c in self.attribute_types.keys() if c in X.columns]
        if not declared_cols:
            raise ValueError("None of the attribute_types columns exist in X.")

        X0 = X[declared_cols].copy()

        kept_cols: List[str] = []
        for col in declared_cols:
            non_nan_ratio = float(X0[col].notna().mean())
            if non_nan_ratio >= self.min_non_nan_ratio:
                kept_cols.append(col)

        if not kept_cols:
            raise ValueError("After applying min_non_nan_ratio, no usable attributes remain.")

        self.attribute_types_ = {c: self.attribute_types[c] for c in kept_cols}

        self.impute_values = {}
        for col, col_type in self.attribute_types_.items():
            s = X0[col]
            if col_type == "numeric":
                if s.notna().any():
                    self.impute_values[col] = float(s.median())
                else:
                    self.impute_values[col] = 0.0
            elif col_type == "nominal":
                self.impute_values[col] = "__MISSING__"
            else:
                raise ValueError(f"Unknown attribute type for {col}: {col_type}")

        X_imp = X0[kept_cols].copy()
        for col, col_type in self.attribute_types_.items():
            if col_type == "numeric":
                X_imp[col] = X_imp[col].fillna(self.impute_values[col]).astype(float)
            else:
                X_imp[col] = X_imp[col].fillna(self.impute_values[col]).astype(str)

        final_attrs: List[str] = []
        for col in kept_cols:
            if X_imp[col].nunique(dropna=False) > 1:
                final_attrs.append(col)

        if not final_attrs:
            raise ValueError("After removing constant attributes, no usable attributes remain.")

        self.root = self._build_tree(X_imp[final_attrs], y, available_attrs=final_attrs, depth=0)

    def predict_one(self, x: Union[pd.Series, Dict[str, Any]]) -> Any:
        if self.root is None:
            raise RuntimeError("Tree is not fitted yet.")

        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        node = self.root
        while not node.is_leaf:
            attr = node.split_attr
            if attr is None:
                return node.prediction

            val = x.get(attr, None)

            if attr in self.impute_values and (val is None or pd.isna(val)):
                val = self.impute_values[attr]

            if node.split_threshold is not None:
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return node.prediction
                node = node.left if v <= node.split_threshold else node.right
                if node is None:
                    return self._fallback_class()
                continue

            if node.branches is not None:
                key = str(val)
                if key in node.branches:
                    node = node.branches[key]
                else:
                    return node.prediction
            else:
                return node.prediction

        return node.prediction

    def predict(self, X: pd.DataFrame) -> List[Any]:
        if self.root is None:
            raise RuntimeError("Tree is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        preds: List[Any] = []
        for i in range(len(X)):
            preds.append(self.predict_one(X.iloc[i, :]))
        return preds

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns a dict of {feature_name: importance_score}.
        Importance is calculated by how often a feature is used for splitting
        and how high up in the tree it appears (weighted by depth).
        """
        importance = {}
        
        def _recurse(node, weight=1.0):
            if node.is_leaf:
                return
            
            if node.split_attr:
                current_score = importance.get(node.split_attr, 0)
                importance[node.split_attr] = current_score + (weight * node.num_samples)
            decay = 0.5
            
            if node.branches:
                for child in node.branches.values():
                    _recurse(child, weight * decay)
            elif node.left and node.right:
                _recurse(node.left, weight * decay)
                _recurse(node.right, weight * decay)

        if self.root:
            _recurse(self.root, weight=1.0)
            
        # Normalize to 0-1 range
        total = sum(importance.values())
        if total > 0:
            return {k: v / total for k, v in importance.items()}
        return {}

    def export_graphviz(self) -> str:
        lines = ["digraph Tree {", 'node [shape=box];']
        self.node_counter = 0
        
        def clean(s):
            return str(s).replace('"', '\\"')

        def _recurse(node, parent_id=None, edge_label=None):
            current_id = f"node_{self.node_counter}"
            self.node_counter += 1
            
            if node.is_leaf:
                label_text = f"Leaf: {clean(node.prediction)}\n(n={node.num_samples})"
                color = "lightred"
            else:
                label_text = f"Split: {clean(node.split_attr)}\n(n={node.num_samples})"
                color = "lightblue"
            
            lines.append(f'{current_id} [label="{label_text}", style=filled, fillcolor={color}];')

            if parent_id is not None:
                lines.append(f'{parent_id} -> {current_id} [label="{clean(edge_label)}"];')

            if node.branches:
                for val, child in node.branches.items():
                    _recurse(child, current_id, val)
            elif node.left and node.right:
                _recurse(node.left, current_id, f"<= {node.split_threshold:.2f}")
                _recurse(node.right, current_id, f"> {node.split_threshold:.2f}")

        if self.root:
            _recurse(self.root)
        
        lines.append("}")
        return "\n".join(lines)

    #  INTERNAL HELPERS

    def _fallback_class(self) -> Any:
        if self._classes_:
            return self._classes_[0]
        return None

    @staticmethod
    def _entropy(labels: pd.Series) -> float:
        n = len(labels)
        if n == 0:
            return 0.0
        counts = labels.value_counts()
        ent = 0.0
        for c in counts:
            p = c / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    @staticmethod
    def _majority_class(labels: pd.Series) -> Any:
        return labels.value_counts().idxmax()

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, available_attrs: List[str], depth: int) -> TreeNode:
        num_samples = len(y)
        counts = y.value_counts().to_dict()

        if y.nunique() <= 1:
            return TreeNode(True, prediction=y.iloc[0] if num_samples else None, distribution=counts, num_samples=num_samples)

        if self.max_depth is not None and depth >= self.max_depth:
            maj = self._majority_class(y)
            return TreeNode(True, prediction=maj, distribution=counts, num_samples=num_samples)

        if not available_attrs or num_samples < self.min_samples_split:
            maj = self._majority_class(y)
            return TreeNode(True, prediction=maj, distribution=counts, num_samples=num_samples)

        best_attr, best_threshold = self._choose_best_attribute(X, y, available_attrs)
        if best_attr is None:
            maj = self._majority_class(y)
            return TreeNode(True, prediction=maj, distribution=counts, num_samples=num_samples)

        node = TreeNode(
            is_leaf=False,
            split_attr=best_attr,
            split_threshold=best_threshold,
            prediction=self._majority_class(y),
            distribution=counts,
            num_samples=num_samples,
        )

        col_type = self.attribute_types_[best_attr]

        if col_type == "nominal":
            node.branches = {}
            values = X[best_attr].astype(str).unique()
            new_attrs = [a for a in available_attrs if a != best_attr]

            for v in values:
                mask = (X[best_attr].astype(str) == v)
                child = self._build_tree(X[mask], y[mask], new_attrs, depth + 1)
                node.branches[v] = child

        else:
            thr = best_threshold
            if thr is None:
                node.is_leaf = True
                node.split_attr = None
                return node

            left_mask = X[best_attr] <= thr
            right_mask = X[best_attr] > thr

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                node.is_leaf = True
                node.split_attr = None
                return node

            node.left = self._build_tree(X[left_mask], y[left_mask], available_attrs, depth + 1)
            node.right = self._build_tree(X[right_mask], y[right_mask], available_attrs, depth + 1)

        return node

    def _choose_best_attribute(
        self, X: pd.DataFrame, y: pd.Series, available_attrs: List[str]
    ) -> Tuple[Optional[str], Optional[float]]:
        base_entropy = self._entropy(y)
        if base_entropy == 0.0:
            return None, None

        best_attr = None
        best_threshold = None
        best_gain_ratio = 0.0

        for attr in available_attrs:
            attr_type = self.attribute_types_.get(attr)
            if attr_type is None:
                continue

            if attr_type == "nominal":
                gain, gain_ratio = self._gain_ratio_nominal(X[attr].astype(str), y, base_entropy)
                threshold = None
            else:
                gain, gain_ratio, threshold = self._gain_ratio_numeric(X[attr].astype(float), y, base_entropy)

            if gain <= 1e-12:
                continue

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_attr = attr
                best_threshold = threshold

        return best_attr, best_threshold

    def _gain_ratio_nominal(self, x_col: pd.Series, y: pd.Series, base_entropy: float) -> Tuple[float, float]:
        n = len(y)
        if n == 0: return 0.0, 0.0

        values = x_col.unique()
        info_after = 0.0
        split_info = 0.0

        for v in values:
            mask = (x_col == v)
            y_v = y[mask]
            n_v = len(y_v)
            if n_v == 0: continue
            p_v = n_v / n
            info_after += p_v * self._entropy(y_v)
            split_info -= p_v * math.log2(p_v)

        gain = base_entropy - info_after
        if split_info == 0.0: return gain, 0.0
        return gain, gain / split_info

    def _gain_ratio_numeric(
        self, x_col: pd.Series, y: pd.Series, base_entropy: float
    ) -> Tuple[float, float, Optional[float]]:
        mask_valid = x_col.notna()
        x_valid = x_col[mask_valid]
        y_valid = y[mask_valid]

        n = len(y_valid)
        if n == 0: return 0.0, 0.0, None

        sort_idx = x_valid.sort_values().index
        x_sorted = x_valid.loc[sort_idx]
        y_sorted = y_valid.loc[sort_idx]

        unique_vals = x_sorted.unique()
        if len(unique_vals) <= 1: return 0.0, 0.0, None

        thresholds = [(unique_vals[i] + unique_vals[i + 1]) / 2.0 for i in range(len(unique_vals) - 1)]

        best_gain_ratio = 0.0
        best_gain = 0.0
        best_threshold = None

        for thr in thresholds:
            left_mask = (x_sorted <= thr)
            right_mask = ~left_mask

            y_left = y_sorted[left_mask]
            y_right = y_sorted[right_mask]

            n_left = len(y_left)
            n_right = len(y_right)
            if n_left == 0 or n_right == 0: continue

            p_left = n_left / n
            p_right = n_right / n

            info_after = p_left * self._entropy(y_left) + p_right * self._entropy(y_right)
            gain = base_entropy - info_after

            split_info = 0.0
            split_info -= p_left * math.log2(p_left)
            split_info -= p_right * math.log2(p_right)

            gain_ratio = 0.0 if split_info == 0.0 else (gain / split_info)

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_gain = gain
                best_threshold = float(thr)

        return best_gain, best_gain_ratio, best_threshold