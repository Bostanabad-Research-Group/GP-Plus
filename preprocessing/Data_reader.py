# data_reader.py in the gpplus package
import pandas as pd
import pkg_resources

def data_reader(file_name):
    # Construct the resource name
    resource_name = pkg_resources.resource_filename('gpplus', 'datasets/' + file_name)

    # Determine the file type and read accordingly
    if file_name.endswith('.csv'):
        return pd.read_csv(resource_name)
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(resource_name)
    else:
        raise ValueError("Unsupported file format: Only CSV and Excel files are supported.")