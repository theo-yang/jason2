# define nested dictionaries for storing data
class nested_dict(dict):
    """Initialize nested dict"""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value