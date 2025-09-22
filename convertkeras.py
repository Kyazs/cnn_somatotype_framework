# quick stats: check scale / possible unit mismatch or outliers
import sys
sys.path.append('src')

import numpy as np
from src.extractor.extractor_model_training import load_databases
from utils import UK_MEAS

df = load_databases()
stats = df[UK_MEAS].describe().T
print(stats[['count','mean','std','min','25%','50%','75%','max']])