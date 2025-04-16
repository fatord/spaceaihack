# export.py
import pandas as pd
import numpy as np

def export_trajectory(trajectory, filename):
    df = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
