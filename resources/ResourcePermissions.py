class ResourcePermissions:
    def __init__(self, df, mode='basic'):
        self.permissions = {}   # {act_name: {user_id, user_id}}

        if mode == 'basic':
            self.load_permissions(df)
        elif mode == 'advanced':
            # TODO
            pass

    def load_permissions(self, df):
        grouped = df.groupby('Activity')['Resource'].unique()
        self.permissions = {
            act: set(resources) for act, resources in grouped.items()
        }

    def is_permitted(self, act_name, res_name):
        allowed = self.permissions.get(act_name)
        if allowed:
            return res_name in allowed
        return False

    def get_permitted_resources(self, act_name):
        return list(self.permissions.get(act_name, []))
