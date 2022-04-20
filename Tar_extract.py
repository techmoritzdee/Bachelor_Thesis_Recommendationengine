# Import from tar.gz to pandas dataframe
# Import packages
import tarfile
import pandas as pd

# settings matrixrange (plot view)
pd.options.display.max_columns = None
pd.options.display.max_rows = 20

# Open tar
def open_tar(x):
    return tarfile.open(x)

# Read Tar to PD

def read_to_pd(x, df):
    for member in x.getmembers():
        f = x.extractfile(member)
        try:
            df = df.append(pd.read_csv(f, nrows=10))
        except:
            print('ERROR: Continue, did not find %s in tar archive.' % member)