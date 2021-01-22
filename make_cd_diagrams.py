import Orange
from autorank import autorank
import pandas as pd
import matplotlib.pyplot as plt


def saveCD(data, name='test', title='title'):
    models = list(data.model)
    data = data.drop(columns=['model', 'Average'])
    values = data.values
    values = values.T
    data = pd.DataFrame(values, columns=models)
    result = autorank(data, alpha=0.05, verbose=False)
    print(result)
    critical_distance = result.cd
    rankdf = result.rankdf
    avranks = rankdf.meanrank
    ranks = list(avranks.values)
    names = list(avranks.index)
    names = names[:30]
    avranks = ranks[:30]
    Orange.evaluation.graph_ranks(avranks, names, cd=critical_distance, width=10, textspace=1.5, labels=True)
    plt.suptitle(title)
    plt.savefig('results/imgs/eps/' + name + ".eps", format="eps")
    plt.savefig('results/imgs/png/' + name + ".png", format="png")
    plt.show()
    plt.close()


def get_box_plot_data(labels, bp):
    rows_list = []
    for i in range(len(labels)):
        dict1 = {'label': labels[i], 'lower_whisker': bp['whiskers'][i * 2].get_ydata()[1],
                 'lower_quartile': bp['boxes'][i].get_ydata()[1], 'median': bp['medians'][i].get_ydata()[1],
                 'upper_quartile': bp['boxes'][i].get_ydata()[2],
                 'upper_whisker': bp['whiskers'][(i * 2) + 1].get_ydata()[1]}
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)


def main():

    resultsPaths = ['results/minmax/30_Models_AUC_MINMAX.csv', 'results/minmax/30_Models_AVE_MINMAX.csv',
                    'results/no/30_Models_AUC_NO_SCALER.csv', 'results/no/30_Models_AVE_NO_SCALER.csv',
                    'results/std/30_Models_AUC_STD_SCALER.csv', 'results/std/30_Models_AVE_STD_SCALER.csv']

    namesCD = ['cd auc minmax scale', 'cd ave minmax scale',
               'cd auc no scale', 'cd ave no scale',
               'cd auc std scale', 'cd ave std scale']

    titlesCD = ['Cd diagram - AUC with min max scaling', 'Cd diagram - Average precision with min max scaling',
                'Cd diagram - AUC without scaling', 'Cd diagram - Average precision without scaling',
                'Cd diagram - AUC with standard scaling', 'Cd diagram - Average precision with  standard scaling']

    for results, name, title in zip(resultsPaths, namesCD, titlesCD):
        r = pd.read_csv(results)
        saveCD(r, name, title)


if __name__ == "__main__":
    main()

