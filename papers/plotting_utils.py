def plot_line_nested_epsilon_dict(data):
    import seaborn as sns
    import pandas as pd
    return sns.lineplot(data=pd.DataFrame(data).T)

def plot_box_map(data, title, vmin=0, vmax=5):
    import matplotlib.pyplot as plt
    import seaborn as sb
    fig, ax = plt.subplots(figsize=(12, 4))
    corr = data.copy()
    # color map
    cmap = sb.diverging_palette(10, 150, s=40, l=55, n=9)
    # plot heatmap
    ax = sb.heatmap(corr, annot=True, fmt=".2f", 
            linewidths=5, cmap=cmap, vmin=vmin, vmax=vmax, 
            cbar_kws={"shrink": .8}, square=True)
    ax.set(xlabel='finding #')
    plt.yticks(plt.yticks()[0], rotation=0)
    plt.xticks(plt.xticks()[0])
    plt.title(title, loc='left', fontsize=18)
    plt.show()
    return plt

def plot_box_simple_small(data, vmin=0, vmax=5):
    import matplotlib.pyplot as plt
    import seaborn as sb
    fig, ax = plt.subplots(figsize=(10, 4))
    corr = data.copy()
    # color map
    cmap = sb.diverging_palette(10, 150, s=40, l=55, n=9)
    # plot heatmap
    ax = sb.heatmap(corr, annot=False, fmt=".2f", 
            linewidths=5, cmap=cmap, vmin=vmin, vmax=vmax, 
            cbar_kws={"shrink": .8}, square=True)
    # ax.set(xlabel='finding #')
    # plt.yticks(plt.yticks()[0], rotation=0)
    # plt.xticks(plt.xticks()[0])
    plt.show()
    return plt

