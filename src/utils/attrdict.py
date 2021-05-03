class AttrDict(dict):
    """ Dictionary that also lets you get the entries as properties """
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)

    @staticmethod
    def from_dict(dict):
        attrdict =  AttrDict()
        for key, value in dict.items():
            attrdict[key] = value # Calls __setitem_
        return attrdict