import os.path
import pandas as pd


class ResourcePermissions:
    """
    Manages resource permissions for activities using either
    Basic (ACL) or Advanced (RBAC) approach.
    """
    def __init__(self, permissions_file, mode='basic'):
        """
        Args:
            mode (str): 'basic' (default) or 'advanced'.
            permissions_file (str): Filename in 'resources/' directory.
        """
        self._permissions = {}
        self._mode = mode

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'resources', permissions_file)

        if mode == 'basic':
            self._load_permissions(file_path)
        elif mode == 'advanced':
            if not permissions_file:
                raise ValueError("Advanced mode requires a valid permissions_file path.")
            self._load_roles(permissions_file)

    def _load_permissions(self, path):
        print(f"Loading Basic Permissions (ACL) from {os.path.basename(path)}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Permission file not found at: {path}")

        df = pd.read_csv(path)
        grouped = df.groupby('Activity')['Resource'].unique()
        self.permissions = {
            act: set(res) for act, res in grouped.items()
        }

    def _load_roles(self, path):
        print(f"Loading Advanced Permissions (RBAC) from {os.path.basename(path)}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Permission file not found at: {path}")

        df = pd.read_csv(path)
        grouped = df.groupby('Activity')['Role'].unique()
        self.permissions = {
            act: set(role) for act, role in grouped.items()
        }

    def is_permitted(self, act_name, res):
        allowed = self._permissions.get(act_name)
        if allowed:
            if self._mode == 'advanced':
                return res.role in allowed
            else:
                return res.name in allowed
        return False

    def get_permitted_resources(self, act_name, resources):
        allowed = self._permissions.get(act_name)
        if self._mode == 'advanced':
            return list(res for res in resources if res.name in allowed)
        return list(allowed)
