def plot_line_nested_epsilon_dict(data):
    import seaborn as sns
    import pandas as pd
    return sns.lineplot(data=pd.DataFrame(data).T)

def plot_line_nested_epsilon_dict_error_bars(data, legend="brief"):
    import seaborn as sns
    import pandas as pd
    g = sns.lineplot(data=pd.DataFrame(data), x="str_eps", y="value", hue="synth", style="synth", 
                                err_style="bars", ci=95, err_kws={"capsize": 5}, legend=legend)
    if legend:
        g.get_legend().set_title(None)
    g.set(xlabel=None, ylabel=None)
    return g

def plot_box_map(data, title, vmin=0, vmax=5):
    import matplotlib.pyplot as plt
    import seaborn as sb
    fig, ax = plt.subplots(figsize=(24, 8))
    corr = data.copy()
    # color map
    cmap = sb.diverging_palette(240, 34, s=80, l=55, n=9)
    # plot heatmap
    ax = sb.heatmap(corr, annot=True, fmt=".0f", 
            linewidths=5, cmap=cmap, vmin=vmin, vmax=vmax, 
            square=True)
    ax.set(xlabel='finding #')
    plt.yticks(plt.yticks()[0], rotation=0, fontsize=20)
    plt.xticks(plt.xticks()[0], fontsize=14)
    sb.set(font_scale=2)
    plt.title(title, loc='left', fontsize=30)
    plt.show()
    return plt

def plot_box_simple_small(data, vmin=0, vmax=5):
    import matplotlib.pyplot as plt
    import seaborn as sb
    fig, ax = plt.subplots(figsize=(10, 4))
    corr = data.copy()
    # color map (colorblind friendly)
    cmap = sb.diverging_palette(240, 34, s=80, l=55, n=9)
    # plot heatmap
    ax = sb.heatmap(corr, annot=False, fmt=".2f", 
            linewidths=5, cmap=cmap, vmin=vmin, vmax=vmax, 
            cbar_kws={"shrink": .8}, square=True)
    # ax.set(xlabel='finding #')
    # plt.yticks(plt.yticks()[0], rotation=0)
    # plt.xticks(plt.xticks()[0])
    plt.show()
    return plt

