import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def scale(array, scale_param=None):
    """
    :param array: input ndarray need to be scale or tranform
    :param method: Gauss distribute or kbins, parameter:'Gauss', 'KBins'
    :return: scale array & scaler
    """
    if len(array.shape) == 1:
        array = array.reshape((-1, 1))
    elif len(array.shape) > 2:
        raise ValueError("the shape of array is not correct, we need array shape in 2 dimension not {}".format(len(array.shape)))
    scaler = StandardScaler()
    if scale_param is None:
        scale_param = scaler.fit(array)
    return scaler.fit_transform(array, scale_param).reshpae((-1)), scale_param

def seprate_bins(array, bin_num, method='uniform'):
    """
    transform a array to one-hot array.
    Note that binning features generally has no beneficial effect for tree-based models,
    as these models can learn to split up the data anywhere.
    :param array: array need to be transform
    :param method: separate method, parameter: 'uniform', 'quantile', 'kmeans'
    :return: array
    """
    scaler = KBinsDiscretizer(n_bins=bin_num, encode='onehot', strategy=method)
    return scaler.fit_transform(array)

def read_csv(csv_path, key=None):
    """
    read csv return the whole file in panda forms or return the determined key-value.
    """
    if key == None:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path)[key]

def get_distribute(DataFrame, key):
    # return the nuber distribute of key-value
    return getattr(DataFrame, key).value_counts()

def plot_figure(x, y, x_label=None, y_label=None, legend=None, figure_type=None, save_path=None, title=None, dpi=200, grid=False):
    """
    :param figure_type: 'pi', 'scatter', 'hist'
    :param save_path: the path to save the figure
    :return:
    """
    fig = plt.figure(dpi=dpi)
    if legend != None:
        if figure_type == None:
            if len(y.shape) > 1:
                for i in range(len(y)):
                    plt.plot(x, y[i], legend[i])
            else:
                plt.plot(x, y, legend)
        elif figure_type == 'bar':
            if len(y.shape) > 1:
                for i in range(len(y)):
                    plt.bar(x, y[i], legend[i], stack=True)
            else:
                plt.bar(x, y, legend)
        elif figure_type == 'scatter':
            if len(y.shape) > 1:
                for i in range(len(y)):
                    plt.scatter(x, y[i], legend[i])
            else:
                plt.scatter(x, y, legend)
        elif figure_type == 'pie':
            if len(y.shape) > 1:
                raise ValueError("the shape of y must be one dimension")
            else:
                plt.pie(y, explode=[0 for i in range(len(y))], labels=legend, autopct="%1.1f%%")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
    else:
        if figure_type == None:
            if len(y.shape) > 1:
                for i in range(len(y)):
                    plt.plot(x, y[i])
            else:
                plt.plot(x, y)
        elif figure_type == 'bar':
            if len(y.shape) > 1:
                for i in range(len(y)):
                    plt.plot(x, y[i], stack=True)
            else:
                plt.bar(x, y)
        elif figure_type == 'scatter':
            if len(y.shape) > 1:
                for i in range(len(y)):
                    plt.scatter(x, y[i])
            else:
                plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    if save_path != None:
        plt.savefig(save_path)
    if title != None:
        plt.title(title)
    if grid:
        plt.grid(True)
    plt.show()