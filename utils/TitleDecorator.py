'''Title Creator for plots'''

class TitleCreator:
    def __init__(self, keys, expected):
        mappings = {'alpha': AlphaTitle, 'beta': BetaTitle, 'g_dsr': GdsrTitle, 'g_dsa': GdsaTitle, 'l_dsr': LdsrTitle,
                    'l_dsa': LdsaTitle, 'wv': WvTitle, 'H_oz': HozTitle}
        self.decor = EmptyMessage()
        for key, value in zip(keys, expected):
            self.decor = mappings[key](value, self.decor)


    def built_title(self):
        params = self.decor.get_title()
        wrapper = r' %s ' % params
        return wrapper


class EmptyMessage:
    def __init__(self):
        pass

    def get_title(self):
        return ''


class HozTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$H_{oz}=%s$ ' % self.expected


class WvTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$wv=%s$ ' % self.expected


class LdsaTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$l_{dsa}=%s$ ' % self.expected


class LdsrTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$l_{dsr}=%s$ ' % self.expected


class AlphaTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$\alpha=%s$ ' % self.expected


class BetaTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$\beta=%s$ ' % self.expected


class GdsrTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$g_{dsr}=%s$ ' % self.expected


class GdsaTitle:
    def __init__(self, expected, base=EmptyMessage()):
        self.expected = expected
        self.base = base

    def get_title(self):
        return self.base.get_title() + r'$g_{dsa}=%s$ ' % self.expected


def test():
    keys = ['alpha', 'beta', 'g_dsr', 'g_dsa']
    expected = [1, 2, 3, 4]
    builder = TitleCreator(keys, expected)
    title = builder.built_title()
    print(title)


if __name__ == "__main__":
    test()