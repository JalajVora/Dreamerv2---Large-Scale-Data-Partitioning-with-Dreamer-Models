from glob import glob
from scipy.interpolate import make_interp_spline, BSpline
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np

All_files = glob('./partition/*.csv')
headers = ('x','y')
name2 = "./partition/graph"
x_label = "steps"
y_label = "Return"
plt.xlabel(x_label)
plt.ylabel(y_label)

for file in All_files:
    name = file.split('.csv')[0]
    name = name.split('/')[-1]
    df = pd.read_csv(file,
                      names=headers,
                      header=0)
    x,y = df['x'], df['y']
    if 0==1:
        spl = make_interp_spline(x, y, k=3)
        x = np.linspace(x.min(), x.max(), 200)
        y = spl(x)
    else:
        y = gaussian_filter1d(y, sigma=20)

    plt.plot(x,y, label=name)
    plt.legend()

plt.tight_layout()
plt.savefig(name2+'.png', dpi=300)
