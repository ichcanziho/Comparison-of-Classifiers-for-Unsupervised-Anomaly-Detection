import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def get_box_plot_data(labels, bp):
    rows_list = []
    for i in range(len(labels)):
        dict1 = {'label': labels[i], 'lower_whisker': bp['whiskers'][i * 2].get_ydata()[1],
                 'lower_quartile': bp['boxes'][i].get_ydata()[1], 'median': bp['medians'][i].get_ydata()[1],
                 'upper_quartile': bp['boxes'][i].get_ydata()[2],
                 'upper_whisker': bp['whiskers'][(i * 2) + 1].get_ydata()[1]}
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)


def transform_data(results):
    base = results.copy()
    models = list(results.model)
    results = results.drop(columns=['model', 'Average'])
    results = results.values
    results = results.T
    bp = plt.boxplot(results)
    info = get_box_plot_data(models, bp)
    plt.close()
    base['median'] = list(info['median'])
    base = base.sort_values(by=['median'])
    models = list(base.model)
    base = base.drop(columns=['model', 'Average', 'median'])
    base = base.values
    base = base.T
    return base,models


def make_plot(paths, scalers):

    locs = [(0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1)]
    datas = []
    names = []

    for path in paths:
        a = pd.read_csv(path)
        data, name = transform_data(a)
        datas.append(data)
        names.append(name)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    tests = ['']*6
    i = 0
    for data, name, scaler in zip(datas, names, scalers):
        sns.boxplot(ax=axes[locs[i]], data=data)
        axes[locs[i]].set_title(scaler)
        axes[locs[i]].set_xticklabels(names[i], rotation=90)
        i += 1

    axes[(0, 1)].set_yticklabels(tests[0])
    axes[(1, 1)].set_yticklabels(tests[0])
    axes[(2, 1)].set_yticklabels(tests[0])

    plt.suptitle("Area Under the ROC Curve                                              Average Precision",fontweight="bold")
    plt.savefig('results/imgs/summarize.png', bbox_inches='tight')
    plt.show()


def main():
    paths = ['results/no/30_Models_AUC_NO_SCALER.csv',
             'results/minmax/30_Models_AUC_MINMAX.csv',
             'results/std/30_Models_AUC_STD_SCALER.csv',
             'results/no/30_Models_AVE_NO_SCALER.csv',
             'results/minmax/30_Models_AVE_MINMAX.csv',
             'results/std/30_Models_AVE_STD_SCALER.csv']

    scalers = ['Without scaler', 'Min max scaler', 'Standard scaler',
               'Without scaler', 'Min max scaler', 'Standard scaler']

    make_plot(paths, scalers)


if __name__ == "__main__":
    main()
