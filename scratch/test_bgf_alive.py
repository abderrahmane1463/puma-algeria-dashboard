import pickle
import pandas as pd
from pathlib import Path
from config import BGF_MODEL

if Path(BGF_MODEL).exists():
    with open(BGF_MODEL, 'rb') as f:
        bgf = pickle.load(f)
    res = bgf.conditional_probability_alive(5, 20, 52)
    print(f"Type: {type(res)}")
    print(f"Value: {res}")
    print(f"Shape/Length: {getattr(res, 'shape', 'No shape')}")
else:
    print("Model not found")
