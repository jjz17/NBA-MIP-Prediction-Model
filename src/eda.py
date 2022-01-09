import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv')

# Shape of Dataset
print(data.shape)

features = data.drop(['Season', 'Outcome', 'Player'], axis=1)
target = data['Outcome']


plot = sns.pairplot(data, y_vars="Outcome", x_vars=['PTS', 'TOV','AST', 'FG%', 'MP', '3P']).set(ylabel = "MIP-Shares")
plot.fig.suptitle("Features vs. MIP-Shares", y = 1.1)

sns.histplot(data['PTS']).set(title = "Frequency of Points Scored")


# instantiate the PCA object and request two components
pca = PCA(n_components= 2, random_state=3000)

# standardize the features so they are all on the same scale
features_standardized = StandardScaler().fit_transform(features)

reduced_data = pca.fit_transform(features_standardized)
reduced_df = pd.DataFrame(reduced_data, columns = ["Component1", "Component2"])
reduced_df["target"] = target

graph = px.scatter(reduced_df, x="Component1", y="Component2", color = "target", title = "Dimensionality Reduction with PCA")
graph.show()
# graph.write_html("pca_scatter.html")


tsne = TSNE(n_components=2, random_state=3000)

tsne_reduced_data = tsne.fit_transform(features_standardized)
tsne_reduced_df = pd.DataFrame(tsne_reduced_data, columns = ["Component1", "Component2"])
tsne_reduced_df["target"] = target

tsne_graph = px.scatter(tsne_reduced_df, x="Component1", y="Component2", color = "target",
                        title = "Dimensionality Reduction with t-SNE")
tsne_graph.show()
# tsne_graph.write_html("tsne_scatter.html")
