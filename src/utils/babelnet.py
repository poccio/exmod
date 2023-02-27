import functools

_wn2bn = {}
_bn2wn = {}


try:

    with open("data/resources/babelnet/bn2wn.txt") as f:
        for line in f:
            line = line.strip()
            parts = line.split("\t")
            _bn2wn[parts[0]] = parts[1]
            _wn2bn[parts[1]] = parts[0]

    def bn_id2wn_id(bn_id):
        return _bn2wn[bn_id]

    def wn_id2bn_id(wn_id):
        return _wn2bn[wn_id]

except FileNotFoundError:

    def bn_id2wn_id(bn_id):
        raise FileNotFoundError

    def wn_id2bn_id(wn_id):
        raise FileNotFoundError
