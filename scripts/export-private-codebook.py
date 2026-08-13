"""Convert a trusted research pickle to JSON for private evaluation only."""
import json, math, pickle, struct, sys
from pathlib import Path

class Dtype:
    def __init__(self, name, *_a, **_k): self.name, self.byteorder = str(name), "<"
    def __setstate__(self, state): self.byteorder = state[0] if isinstance(state, tuple) and state else "<"
class Array:
    def __setstate__(self, state): _, self.shape, self.dtype, self.fortran, self.raw = state
    def __iter__(self):
        code = "d" if "f8" in self.dtype.name else "f"
        yield from struct.unpack((">" if self.dtype.byteorder == ">" else "<") + code * (len(self.raw) // struct.calcsize(code)), self.raw)
    def __len__(self): return self.shape[0]
def reconstruct(*_args): return Array()
class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct": return reconstruct
        if module == "numpy" and name == "ndarray": return Array
        if module == "numpy" and name == "dtype": return Dtype
        if module == "builtins" and name in {"set", "frozenset", "slice"}: return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

source = Path(sys.argv[1]); target = Path(sys.argv[2]); book = Restricted(source.open("rb")).load(); refs = book["carrier_refs"]
dark = [(-5,-3),(5,3),(-5,3),(5,-3),(-3,-4),(3,4),(-3,4),(3,-4),(-4,-3),(4,3),(-4,3),(4,-3),(-5,-1),(5,1),(-5,1),(5,-1),(-5,-2),(5,2),(-5,2),(5,-2),(-2,-5),(2,5),(-2,5),(2,-5),(-1,-5),(1,5),(-1,5),(1,-5),(-4,-4),(4,4),(-4,4),(4,-4),(-1,-6),(1,6),(-3,-5),(3,5)]
white = [(0,-7),(0,7),(0,-8),(0,8),(0,-9),(0,9),(0,-10),(0,10),(0,-11),(0,11),(0,-12),(0,12),(0,-20),(0,20),(0,-21),(0,21),(0,-22),(0,22),(0,-23),(0,23)]
payload = {"imageSize": int(book.get("image_size", 512)), "sets": [{"name":"dark","carriers":dark,"phases":[float(x) for x in refs["dark_ref_phases"]]}, {"name":"white","carriers":white,"phases":[float(x) for x in refs["white_ref_phases"]]}], "calibration":{"center":0.78,"steepness":20}}
target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(payload), encoding="utf-8")
print(target)
