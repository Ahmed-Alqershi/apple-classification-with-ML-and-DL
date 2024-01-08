import pandas as pd
import matplotlib.pyplot as plt


def show_results():
    results ={
        "resnet50": 95.8,
        "svm": 87.5,
        "knn": 33.3,
        "decision_tree": 43.9,
    }

    df = pd.DataFrame.from_dict(results, orient='index').sort_values(by=0, ascending=False)
    ax = df.plot(kind='bar', legend=False, title="Accuracy of different models")

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.savefig("results.png")
    plt.show()

if __name__ == "__main__":
    show_results()
