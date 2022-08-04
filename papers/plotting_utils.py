def plot_line_nested_epsilon_dict(data):
    import seaborn as sns
    import pandas as pd
    return sns.lineplot(data=pd.DataFrame(data).T)