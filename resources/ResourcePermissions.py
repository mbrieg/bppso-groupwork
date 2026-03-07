import os.path
import pandas as pd


class ResourcePermissions:
    """
    Manages resource permissions for activities using either
    Basic (ACL) or Advanced (RBAC) approach.
    """
    def __init__(self, permissions_file, mode_adv):
        """
        Args:
            mode_adv (bool): False for 'basic' (default) or True for 'advanced'.
            permissions_file (str): Filename in 'resources/' directory.
        """
        self._permissions = {}
        self._mode = mode_adv

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'permissions', permissions_file)
        if mode_adv:
            if not permissions_file:
                raise ValueError("Advanced mode requires a valid permissions_file path.")
            self._load_roles(file_path)
        else:
            self._load_permissions(file_path)

    def _load_permissions(self, path):
        print(f"Loading Basic Permissions (ACL) from {os.path.basename(path)}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Permission file not found at: {path}")

        df = pd.read_csv(path)
        grouped = df.groupby('Activity')['Resource'].unique()
        self._permissions = {
            act: set(res) for act, res in grouped.items()
        }

    def _load_roles(self, path):
        print(f"Loading Advanced Permissions (RBAC) from {os.path.basename(path)}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Permission file not found at: {path}")

        df = pd.read_csv(path)
        grouped = df.groupby('Activity')['Role'].unique()
        self._permissions = {
            act: set(role) for act, role in grouped.items()
        }

    def is_permitted(self, act_name, res):
        allowed = self._permissions.get(act_name)
        if allowed:
            if self._mode:
                return res.role in allowed
            else:
                return res.name in allowed
        return False

    def get_permitted_resources(self, act_name, resources):
        allowed = self._permissions.get(act_name, set())
        if self._mode:
            return [res_id for (res_id, res) in resources.items() if res.get_role() in allowed]
        return list(allowed)
