import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
from umap import UMAP

logging.basicConfig(level=logging.INFO)

# Sample documents
documents = [
    "Machine learning is fascinating.",
    "Artificial intelligence and machine learning are closely related.",
    "Natural language processing is a part of AI.",
    "Deep learning is a subset of machine learning.",
    "AI is transforming the world.",
    "Data science involves statistics and machine learning.",
    "Python is a popular language for data science.",
    "Big data and AI are revolutionizing industries.",
    "Machine learning models are used in predictive analytics.",
    "Machine learning requires a good understanding of statistics.",
    "AI is the future.",
    "Machine learning is the future.",
    "Deep learning is the future.",
    "Python is the best language for machine learning.",
    "Machine learning involves training models.",
    "AI can learn from data.",
    "Machine learning can be supervised or unsupervised.",
    "AI is used in recommender systems.",
    "Data science is a multidisciplinary field.",
    "Machine learning is a subset of AI.",
    "AI is used in self-driving cars.",
    "Machine learning is used in anomaly detection.",
    "AI is used in computer vision.",
    "Machine learning is used in natural language processing.",
    "AI is used in speech recognition.",
    "Machine learning is used in sentiment analysis.",
    "AI is used in facial recognition.",
    "Machine learning is used in image recognition.",
    "AI is used in object detection.",
    "Machine learning is used in clustering.",
    "AI is used in classification.",
    "Machine learning is used in regression.",
    "AI is used in dimensionality reduction.",
    "Machine learning is used in feature selection.",
    "AI is used in reinforcement learning.",
    "Machine learning is used in recommendation systems.",
    "AI is used in adversarial learning.",
    "Machine learning is used in generative models.",
    "AI is used in neural networks.",
    "Machine learning is used in decision trees.",
    "AI is used in random forests.",
    "Machine learning is used in gradient boosting.",
    "AI is used in support vector machines.",
    "Machine learning is used in clustering.",
    "AI is used in classification.",
    "Machine learning is used in regression.",
    "AI is used in dimensionality reduction.",
    "OpenAI is an AI research lab.",
    "Machine learning is used in feature selection.",
    "Some machine learning models are interpretable.",
    "Reasoning is an ongoing challenge in AI.",
    "The Turing Test is used to evaluate AI.",
    "IQ tests are used to evaluate human intelligence.",
    "Some AI systems can pass the Turing Test.",
    "AI is used in robotics.",
    "Some AI systems can perform tasks autonomously.",
    "Something wicked this way comes.",
    "Skynet is a fictional AI system.",
    "The Matrix is a science fiction movie.",
    "The Terminator is a science fiction movie.",
    "AI is used in science fiction.",
    "Mr Data is an android in Star Trek.",
    "The Borg are cyborgs in Star Trek.",
    "The Doctor is an EMH in Star Trek Voyager.",
    "The Cybermen are cyborgs in Doctor Who.",
    "The Daleks are cyborgs in Doctor Who.",
    "The TARDIS is a time machine in Doctor Who.",
    "Everybody lives, Rose. Just this once, everybody lives!",
    "The Silence are a religious order in Doctor Who.",
    "The Weeping Angels are quantum-locked aliens in Doctor Who.",
    "Octupi are intelligent creatures and are like aliens of the deep.",
    "The Kraken is a sea monster of legend.",
    "The Loch Ness Monster is a cryptid.",
    "Scientists have cloned many animals, including Dolly the sheep.",
    "The Large Hadron Collider is a particle accelerator.",
    "Dolly Parton is a country music singer.",
    "The Beatles were a famous rock band.",
    "Elvis Presley was the King of Rock and Roll.",
    "Michael Jackson was the King of Pop.",
    "Madonna is the Queen of Pop.",
    "Queen Elizabeth II is the Queen of England, the United Kingdom, and the Commonwealth.",
    "The Queen of Hearts is a character in Alice in Wonderland.",
    "Many sheep are raised for wool.",
    "Cows are raised for milk and beef.",
    "Pigs are raised for pork.",
    "Men who disrespect women are pigs.",
    "Dogs are loyal companions.",
    "Cats are independent creatures.",
    "Police are sometimes called pigs.",
    "The police are here to protect and serve.",
    "The police are investigating the crime.",
    "In the book 1984, Big Brother is watching you.",
    "The Thought Police monitor citizens in 1984.",
    "The Ministry of Truth rewrites history in 1984.",
    "Truth Social is a social media platform created by Donald Trump known for being a haven for misinformation.",
    "Hello Kitty is a popular character from Japan.",
    "Mickey Mouse is a famous cartoon character.",
    "Mice are small rodents that may have been the cause of the Black Death.",
    "The death of a black man named George Floyd sparked protests against police brutality.",
    "The Black Panther Party was a revolutionary group in the 1960s.",
    "Black Lives Matter is a movement against police violence.",
    "The Black Widow is a character in Marvel Comics.",
    "Trump supporters stormed the Capitol on January 6, 2021.",
    "The Capitol is the seat of the United States Congress.",
    "Congress is the legislative branch of the United States government.",
    "The President of the United States is the head of state and government.",
    "Trump supporters believe the election was stolen. It was not.",
    "The election was certified by Congress, as required by the Constitution.",
    "The Constitution is the supreme law of the United States.",
    "They are trying to steal the election from us, Trump said. Then he tried to steal the election by inciting a violent insurrection at the Capitol to stop the certification of the election results.",
    "The election was free and fair, and Joe Biden won.",
    "Someone who loses an election should gracefully concede and work for a smooth transition of power.",
    "The peaceful transfer of power is a hallmark of democracy.",
    "Democracy is a system of government by the people, for the people.",
    "The people have the power to choose their leaders through free and fair elections.",
    "Elections are the foundation of democracy.",
    "The rule of law is the foundation of democracy.",
    "Donald Trump has stated that he believes he is above the law. He is not, and no one is. He was convicted by a jury of his peers of 34 felony counts of fraud for paying off an adult film star to keep quiet about their affair. It was also deemed a violation of federal election laws, because it was done to influence the election.",
    "The law is the law, and no one is above it.",
    "The law is a complex and evolving system.",
    "The legal system is designed to provide justice for all.",
    "Justice is the foundation of a fair and equitable society.",
    "Society is made up of individuals and groups.",
    "Groups of people can work together to achieve common goals.",
    "Goals are important for motivation and success.",
    "Success is defined by achieving one's goals.",
    "Donald Trunp is a liar and a cheat who has never achieved anything on his own. He was born into wealth and squandered it on failed business ventures, leaving a trail of bankruptcies and lawsuits in his wake. He is despised by most of the world for his incompetence, corruption, and cruelty.",
    "Cruelty is the hallmark of a weak and insecure person.",
    "Insecurity is a common human trait.",
    "Human beings are complex creatures with a wide range of emotions and behaviors.",
    "Behavior is influenced by genetics, environment, and personal experiences.",
    "Experiences shape who we are and how we interact with the world.",
    "The disgraceful behavior of Donald Trump has tarnished the reputation of the United States and undermined its standing in the world.",
    "The discovery of fire was a major milestone in human history. It allowed early humans to cook food, stay warm, and protect themselves from predators. It also enabled the development of tools and the advancement of civilization. But the development of AI and machine learning is a much more significant milestone, as it has the potential to revolutionize every aspect of human life.",
    "The potential of AI and machine learning is limitless. It can be used to solve complex problems, improve efficiency, and enhance decision-making. It can also be used to create new products and services, drive innovation, and transform industries. The possibilities are endless, and the impact is profound.",
    "The impact of AI and machine learning is already being felt in many areas, including healthcare, finance, transportation, and entertainment. It is changing the way we live, work, and interact with the world. It is reshaping the economy, society, and culture. It is a transformative force that is reshaping the future of humanity.",
    "The future of AI and machine learning is bright. It holds the promise of a better world, where technology is used to improve human well-being, protect the environment, and advance knowledge. It offers the potential to address some of the most pressing challenges facing humanity, such as climate change, poverty, and disease. It is a powerful tool for positive change and progress.",
    "The development of AI and machine learning is a collective endeavor that requires collaboration, innovation, and ethical considerations. It is important to ensure that these technologies are developed and used responsibly, ethically, and equitably. It is essential to consider the social, economic, and environmental impacts of AI and machine learning and to address the challenges and opportunities they present.",
    "Sally sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "She sells seashells by the seashore.",
    "Unique New York.",
    "Toy boat. Toy boat. Toy boat.",
    "Red leather, yellow leather.",
    "Irish wristwatch.",
    "Six slippery snails slid slowly seaward.",
    "Black bug's blood.",
    "A proper copper coffee pot.",
    "Polices police police, but who polices the police?",
    "How can a clam cram in a clean cream can?",
    "Police are often called coppers. The term copper comes from the use of copper buttons on police uniforms in the 19th century.",
    "",
]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# TF-IDF Clustering
kmeans_tfidf = KMeans(n_clusters=3, random_state=42)
tfidf_clusters = kmeans_tfidf.fit_predict(tfidf_matrix)

# LDA Vectorization
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(documents)

# LDA Model
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_matrix = lda_model.fit_transform(count_matrix)

# LDA Clustering
kmeans_lda = KMeans(n_clusters=3, random_state=42)
lda_clusters = kmeans_lda.fit_predict(lda_matrix)

# Display Results
df = pd.DataFrame(
    {
        "Document": documents,
        "TF-IDF Cluster": tfidf_clusters,
        "LDA Cluster": lda_clusters,
    }
)

logging.info(df)

# Visualize LDA Topics
for topic_idx, topic in enumerate(lda_model.components_):
    logging.info("Topic {}".format(topic_idx))
    logging.info(
        " ".join(
            [count_vectorizer.get_feature_names()[i] for i in topic.argsort()[:-6:-1]]
        )
    )

# Visualize TF-IDF Features
logging.info("TF-IDF Features")
logging.info(tfidf_vectorizer.get_feature_names())

# Visualize LDA Features
logging.info("LDA Features")
logging.info(count_vectorizer.get_feature_names())

# Visualize LDA Clusters
logging.info("LDA Clusters")
logging.info(lda_clusters)

# Plot LDA Clusters using the most appropriate library for visualizing clusters. Then explain why you chose this method in the comments below the code.
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
tsne_lda = tsne.fit_transform(lda_matrix)

plt.scatter(
    tsne_lda[:, 0],
    tsne_lda[:, 1],
    c=lda_clusters,
    cmap="viridis",
    alpha=0.7,
    s=50,
)
plt.title("LDA Clusters")
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.show()
# TSNE is a good choice for visualizing LDA clusters because it is a nonlinear dimensionality reduction technique that can help visualize the clusters in a lower-dimensional space. It is also computationally efficient and easy to interpret.

# Plot TF-IDF Clusters using the most appropriate library for visualizing clusters. Then explain why you chose this method in the comments below the code.
tsne_tfidf = tsne.fit_transform(tfidf_matrix.toarray())

plt.scatter(
    tsne_tfidf[:, 0],
    tsne_tfidf[:, 1],
    c=tfidf_clusters,
    cmap="viridis",
    alpha=0.7,
    s=50,
)
plt.title("TF-IDF Clusters")
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.show()
# TSNE is a good choice for visualizing TF-IDF clusters because it is a nonlinear dimensionality reduction technique that can help visualize the clusters in a lower-dimensional space. It is also computationally efficient and easy to interpret.

# Plot LDA Clusters using an alternative library and method for visualizing clusters. Then explain why you chose this method in the comments below the code.
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_lda = pca.fit_transform(lda_matrix)

plt.scatter(
    pca_lda[:, 0],
    pca_lda[:, 1],
    c=lda_clusters,
    cmap="viridis",
    alpha=0.7,
    s=50,
)
plt.title("LDA Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
# PCA is a good choice for visualizing LDA clusters because it is a linear dimensionality reduction technique that can help visualize the clusters in a lower-dimensional space. It is also computationally efficient and easy to interpret.

# Plot TF-IDF Clusters using an alternative library and method for visualizing clusters. Then explain why you chose this method in the comments below the code.
pca_tfidf = pca.fit_transform(tfidf_matrix.toarray())

plt.scatter(
    pca_tfidf[:, 0],
    pca_tfidf[:, 1],
    c=tfidf_clusters,
    cmap="viridis",
    alpha=0.7,
    s=50,
)
plt.title("TF-IDF Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
# PCA is a good choice for visualizing TF-IDF clusters because it is a linear dimensionality reduction technique that can help visualize the clusters in a lower-dimensional space. It is also computationally efficient and easy to interpret.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import seaborn as sns

sns.pairplot(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    hue=lda_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing LDA clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different topics in a grid of scatterplots, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
sns.pairplot(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    hue=tfidf_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing TF-IDF clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different features in a grid of scatterplots, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.express as px

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
from mermaid import Mermaid

m = Mermaid()
m.node("Topic 0")
m.node("Topic 1")
m.node("Topic 2")
for i, cluster in enumerate(lda_clusters):
    m.edge(f"Document {i}", f"Topic {cluster}")
m.draw()
# Mermaid is a good choice for visualizing LDA clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and topics using a directed graph, where each document is connected to the corresponding topic cluster.

# Plot TF-IDF Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
m = Mermaid()
m.node("Feature 0")
m.node("Feature 1")
m.node("Feature 2")
for i, cluster in enumerate(tfidf_clusters):
    m.edge(f"Document {i}", f"Feature {cluster}")
m.draw()
# Mermaid is a good choice for visualizing TF-IDF clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and features using a directed graph, where each document is connected to the corresponding feature cluster.

# Plot LDA Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with a more advanced method. I will create a 3D scatter plot of the different topics, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.


# Plot TF-IDF Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with a more advanced method. I will create a 3D scatter plot of the different features, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with the most advanced method. I will create a scatter plot of the different topics, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with the most advanced method. I will create a scatter plot of the different features, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import seaborn as sns

sns.pairplot(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    hue=lda_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing LDA clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different topics in a grid of scatterplots, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
sns.pairplot(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    hue=tfidf_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing TF-IDF clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different features in a grid of scatterplots, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.express as px

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
from mermaid import Mermaid

m = Mermaid()
m.node("Topic 0")
m.node("Topic 1")
m.node("Topic 2")
for i, cluster in enumerate(lda_clusters):
    m.edge(f"Document {i}", f"Topic {cluster}")
m.draw()
# Mermaid is a good choice for visualizing LDA clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and topics using a directed graph, where each document is connected to the corresponding topic cluster.

# Plot TF-IDF Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
m = Mermaid()
m.node("Feature 0")
m.node("Feature 1")
m.node("Feature 2")
for i, cluster in enumerate(tfidf_clusters):
    m.edge(f"Document {i}", f"Feature {cluster}")
m.draw()
# Mermaid is a good choice for visualizing TF-IDF clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and features using a directed graph, where each document is connected to the corresponding feature cluster.

# Plot LDA Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with a more advanced method. I will create a 3D scatter plot of the different topics, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.


# Plot TF-IDF Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with a more advanced method. I will create a 3D scatter plot of the different features, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with the most advanced method. I will create a scatter plot of the different topics, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with the most advanced method. I will create a scatter plot of the different features, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import seaborn as sns

sns.pairplot(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    hue=lda_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing LDA clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different topics in a grid of scatterplots, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
sns.pairplot(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    hue=tfidf_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing TF-IDF clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different features in a grid of scatterplots, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.express as px

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
from mermaid import Mermaid

m = Mermaid()
m.node("Topic 0")
m.node("Topic 1")
m.node("Topic 2")
for i, cluster in enumerate(lda_clusters):
    m.edge(f"Document {i}", f"Topic {cluster}")
m.draw()
# Mermaid is a good choice for visualizing LDA clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and topics using a directed graph, where each document is connected to the corresponding topic cluster.

# Plot TF-IDF Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
m = Mermaid()
m.node("Feature 0")
m.node("Feature 1")
m.node("Feature 2")
for i, cluster in enumerate(tfidf_clusters):
    m.edge(f"Document {i}", f"Feature {cluster}")
m.draw()
# Mermaid is a good choice for visualizing TF-IDF clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and features using a directed graph, where each document is connected to the corresponding feature cluster.

# Plot LDA Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with a more advanced method. I will create a 3D scatter plot of the different topics, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.


# Plot TF-IDF Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with a more advanced method. I will create a 3D scatter plot of the different features, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with the most advanced method. I will create a scatter plot of the different topics, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with the most advanced method. I will create a scatter plot of the different features, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import seaborn as sns

sns.pairplot(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    hue=lda_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing LDA clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different topics in a grid of scatterplots, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
sns.pairplot(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    hue=tfidf_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing TF-IDF clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different features in a grid of scatterplots, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.express as px

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
from mermaid import Mermaid

m = Mermaid()
m.node("Topic 0")
m.node("Topic 1")
m.node("Topic 2")
for i, cluster in enumerate(lda_clusters):
    m.edge(f"Document {i}", f"Topic {cluster}")
m.draw()
# Mermaid is a good choice for visualizing LDA clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and topics using a directed graph, where each document is connected to the corresponding topic cluster.

combined_features = np.hstack((tfidf_matrix.toarray(), lda_matrix))


# Plot TF-IDF Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
m = Mermaid()
m.node("Feature 0")
m.node("Feature 1")
m.node("Feature 2")
for i, cluster in enumerate(tfidf_clusters):
    m.edge(f"Document {i}", f"Feature {cluster}")
m.draw()
# Mermaid is a good choice for visualizing TF-IDF clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and features using a directed graph, where each document is connected to the corresponding feature cluster.

# Plot LDA Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with a more advanced method. I will create a 3D scatter plot of the different topics, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.


# Plot TF-IDF Clusters using most appropriate visualization library (you choose the most fitting for the chosen method) with a more advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with a more advanced method. I will create a 3D scatter plot of the different features, with different colors representing different clusters.

fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing LDA clusters with the most advanced method. I will create a scatter plot of the different topics, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using the most appropriate visualization library (you choose the most fitting for the chosen method) with the most advanced method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
# Make it innovative, and tell me what it will be first
# I will use Plotly for visualizing TF-IDF clusters with the most advanced method. I will create a scatter plot of the different features, with different colors representing different clusters.

fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import seaborn as sns

sns.pairplot(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    hue=lda_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing LDA clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different topics in a grid of scatterplots, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this method in the comments below the code.
sns.pairplot(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    hue=tfidf_clusters,
)
plt.show()
# Seaborn is a good choice for visualizing TF-IDF clusters because it provides a high-level interface for creating informative and attractive visualizations. The pairplot function allows us to visualize the relationships between different features in a grid of scatterplots, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.express as px

fig = px.scatter_3d(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    x="Topic 0",
    y="Topic 1",
    z="Topic 2",
    color=lda_clusters,
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different topics, with different colors representing different clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with a more advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = px.scatter_3d(
    pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names()),
    x=tfidf_vectorizer.get_feature_names()[0],
    y=tfidf_vectorizer.get_feature_names()[1],
    z=tfidf_vectorizer.get_feature_names()[2],
    color=tfidf_clusters,
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The scatter_3d function allows us to create a 3D scatter plot of the different features, with different colors representing different clusters.

# Plot LDA Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(
            x=lda_matrix[:, 0],
            y=lda_matrix[:, 1],
            mode="markers",
            marker=dict(color=lda_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing LDA clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different topics, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot TF-IDF Clusters using an alternative visualization library (ie, not matplotlib) with the most advanced method for visualizing clusters. Then explain why you chose this method in the comments below the code.
fig = go.Figure(
    data=[
        go.Scatter(
            x=tfidf_matrix.toarray()[:, 0],
            y=tfidf_matrix.toarray()[:, 1],
            mode="markers",
            marker=dict(color=tfidf_clusters),
        )
    ]
)
fig.show()
# Plotly is a good choice for visualizing TF-IDF clusters because it provides interactive and high-quality visualizations that can be easily customized. The go.Scatter function allows us to create scatter plots of the different features, with different colors representing different clusters. The interactive features of Plotly make it easy to explore the data and identify patterns in the clusters.

# Plot LDA Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
from mermaid import Mermaid

combined_features = np.hstack((tfidf_matrix.toarray(), lda_matrix))


m = Mermaid()
m.node("Topic 0")
m.node("Topic 1")
m.node("Topic 2")
for i, cluster in enumerate(lda_clusters):
    m.edge(f"Document {i}", f"Topic {cluster}")
m.draw()
# Mermaid is a good choice for visualizing LDA clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and topics using a directed graph, where each document is connected to the corresponding topic cluster.

# Plot TF-IDF Clusters using an alternative visualization library (use Mermaid this time) with the most straightforward and interpretable method for visualizing clusters. Then explain why you chose this specific method in the comments below the code.
m = Mermaid()
m.node("Feature 0")
m.node("Feature 1")
m.node("Feature 2")
for i, cluster in enumerate(tfidf_clusters):
    m.edge(f"Document {i}", f"Feature {cluster}")
m.draw()
# Mermaid is a good choice for visualizing TF-IDF clusters because it provides a simple and intuitive way to create diagrams and flowcharts. In this case, we can represent the relationship between documents and features using a directed graph, where each document is connected to the corresponding feature cluster.

# Let's brainstorm some other ways to visualize LDA and TF-IDF clusters. Write down at least five ideas for each type of cluster visualization and explain why each idea would be useful.
# LDA Clusters
# 1. Network Graph: Visualize the relationships between topics as nodes and edges in a network graph. This can help identify the connections between different topics and how they are related to each other.
# 2. Radar Chart: Create a radar chart to compare the importance of different words in each topic. This can help identify the key terms that define each topic and how they differ from each other.
# 3. Heatmap: Generate a heatmap to visualize the distribution of topics across different documents. This can help identify the prevalence of different topics in the corpus and how they are distributed.
# 4. Tree Map: Create a tree map to visualize the hierarchy of topics in the corpus. This can help identify the main topics and subtopics and how they are organized.
# 5. Chord Diagram: Generate a chord diagram to visualize the relationships between topics and documents. This can help identify the connections between different topics and how they are distributed across the corpus.
# TF-IDF Clusters
# 1. Word Cloud: Create a word cloud to visualize the most important terms in the corpus based on TF-IDF scores. This can help identify the key terms that define the corpus and how they are weighted.
# 2. Bar Chart: Generate a bar chart to compare the TF-IDF scores of different terms in the corpus. This can help identify the most important terms and how they contribute to the overall TF-IDF scores.
# 3. Scatter Plot: Create a scatter plot to visualize the distribution of documents based on their TF-IDF scores. This can help identify the clusters of documents with similar TF-IDF scores and how they are related to each other.
# 4. Box Plot: Generate a box plot to compare the TF-IDF scores of different terms across the corpus. This can help identify the distribution of TF-IDF scores and how they vary between terms.
# 5. Violin Plot: Create a violin plot to compare the TF-IDF scores of different terms across the corpus. This can help identify the distribution of TF-IDF scores and how they vary between terms.

# Now, let's visualize the LDA and TF-IDF clusters using some of the ideas mentioned above. Write the code to implement the visualization ideas for each type of cluster and explain the insights gained from each visualization.
# LDA Clusters
# Network Graph
import networkx as nx

G = nx.Graph()

for i, topic in enumerate(lda_model.components_):
    G.add_node(f"Topic {i}")

for i, doc in enumerate(lda_matrix):
    for j, prob in enumerate(doc):
        G.add_edge(f"Document {i}", f"Topic {j}", weight=prob)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10)
plt.show()
# The network graph visualizes the relationships between topics and documents in the corpus. It shows how documents are connected to different topics based on the probability distribution of topics in each document.

# Radar Chart
fig = px.line_polar(
    pd.DataFrame(lda_model.components_, columns=count_vectorizer.get_feature_names())
)
fig.show()
# The radar chart compares the importance of different words in each topic. It shows the key terms that define each topic and how they differ from each other based on their weights.

# Heatmap
plt.imshow(lda_matrix, aspect="auto")
plt.colorbar()
plt.xlabel("Topic")
plt.ylabel("Document")
plt.show()
# The heatmap visualizes the distribution of topics across different documents. It shows the prevalence of different topics in the corpus and how they are distributed across the documents.

# Tree Map
import squarify

sizes = lda_model.components_.sum(axis=1)
labels = [f"Topic {i}" for i in range(lda_model.n_components)]

squarify.plot(sizes=sizes, label=labels, alpha=0.7)
plt.axis("off")
plt.show()
# The tree map visualizes the hierarchy of topics in the corpus. It shows the main topics and subtopics and how they are organized based on their weights.

# Chord Diagram
import matplotlib.pyplot as plt
from chord import Chord

doc_topic_matrix = lda_matrix.T
doc_topic_matrix = doc_topic_matrix / doc_topic_matrix.sum(axis=0)

chord = Chord(doc_topic_matrix, count_vectorizer.get_feature_names())
chord.to_html("chord.html")
# The chord diagram visualizes the relationships between topics and documents in the corpus. It shows the connections between different topics and how they are distributed across the documents.

# TF-IDF Clusters
# Word Cloud
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(
    dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_matrix.toarray().sum(axis=0)))
)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# The word cloud visualizes the most important terms in the corpus based on TF-IDF scores. It shows the key terms that define the corpus and how they are weighted.

# Bar Chart
plt.barh(tfidf_vectorizer.get_feature_names(), tfidf_matrix.toarray().sum(axis=0))
plt.xlabel("TF-IDF Score")
plt.ylabel("Term")
plt.show()
# The bar chart compares the TF-IDF scores of different terms in the corpus. It shows the most important terms and how they contribute to the overall TF-IDF scores.

# Scatter Plot
plt.scatter(
    tfidf_matrix.toarray()[:, 0], tfidf_matrix.toarray()[:, 1], c=tfidf_clusters
)
plt.xlabel("TF-IDF Feature 0")
plt.ylabel("TF-IDF Feature 1")
plt.show()
# The scatter plot visualizes the distribution of documents based on their TF-IDF scores. It shows the clusters of documents with similar TF-IDF scores and how they are related to each other.

# Box Plot
plt.boxplot(tfidf_matrix.toarray())
plt.xticks(
    range(1, len(tfidf_vectorizer.get_feature_names()) + 1),
    tfidf_vectorizer.get_feature_names(),
    rotation=90,
)
plt.xlabel("Term")
plt.ylabel("TF-IDF Score")
plt.show()
# The box plot compares the TF-IDF scores of different terms across the corpus. It shows the distribution of TF-IDF scores and how they vary between terms.

# Violin Plot
plt.violinplot(tfidf_matrix.toarray())
plt.xticks(
    range(1, len(tfidf_vectorizer.get_feature_names()) + 1),
    tfidf_vectorizer.get_feature_names(),
    rotation=90,
)
plt.xlabel("Term")
plt.ylabel("TF-IDF Score")
plt.show()
# The violin plot compares the TF-IDF scores of different terms across the corpus. It shows the distribution of TF-IDF scores and how they vary between terms.

# Are there any other ways you can think of to visualize LDA and TF-IDF clusters? Write down at least three more ideas for each type of cluster visualization and explain why each idea would be useful.
# LDA Clusters
# 1. Sankey Diagram: Create a Sankey diagram to visualize the flow of topics in the corpus. This can help identify the transitions between different topics and how they are connected.
# 2. Flow Chart: Generate a flow chart to visualize the progression of topics in the corpus. This can help identify the sequence of topics and how they are related to each other.
# 3. Word Network: Create a word network to visualize the relationships between words in different topics. This can help identify the connections between different words and how they are distributed across topics.
# TF-IDF Clusters
# 1. Cluster Map: Generate a cluster map to visualize the distribution of documents based on their TF-IDF scores. This can help identify the clusters of documents with similar TF-IDF scores and how they are related to each other.
# 2. Feature Network: Create a feature network to visualize the relationships between features in the corpus. This can help identify the connections between different features and how they are distributed across the corpus.
# 3. Term Frequency Plot: Generate a term frequency plot to compare the frequency of different terms in the corpus. This can help identify the most common terms and how they contribute to the overall TF-IDF scores.
# 4. Bubble Chart: Create a bubble chart to visualize the distribution of documents based on their TF-IDF scores. This can help identify the clusters of documents with similar TF-IDF scores and how they are related to each other.
# 5. Cluster Dendrogram: Generate a cluster dendrogram to visualize the hierarchy of clusters in the corpus. This can help identify the main clusters and subclusters and how they are organized.

# Let's implement some of the visualization ideas mentioned above for both LDA and TF-IDF clusters. Write the code to implement the visualization ideas for each type of cluster and explain the insights gained from each visualization.
# LDA Clusters
# Sankey Diagram
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[f"Topic {i}" for i in range(lda_model.n_components)],
            ),
            link=dict(
                source=[
                    i
                    for i in range(lda_model.n_components)
                    for _ in range(lda_model.n_components)
                ],
                target=[
                    j
                    for _ in range(lda_model.n_components)
                    for j in range(lda_model.n_components)
                ],
                value=lda_model.components_.flatten(),
            ),
        )
    ]
)
fig.show()


# Flow Chart
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

sankey = Sankey()

for i in range(lda_model.n_components):
    sankey.add(
        flows=[1] * lda_model.n_components,
        labels=[f"Topic {j}" for j in range(lda_model.n_components)],
    )

sankey.finish()
plt.show()

# Word Network
import networkx as nx

G = nx.Graph()

for i, topic in enumerate(lda_model.components_):
    for j, prob in enumerate(topic):
        G.add_edge(f"Topic {i}", count_vectorizer.get_feature_names()[j], weight=prob)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10)
plt.show()

# TF-IDF Clusters
# Cluster Map
import seaborn as sns

sns.clustermap(tfidf_matrix.toarray(), row_cluster=False, col_cluster=True)
plt.show()

# Feature Network
G = nx.Graph()

for i, doc in enumerate(tfidf_matrix.toarray()):
    for j, prob in enumerate(doc):
        G.add_edge(
            f"Document {i}", tfidf_vectorizer.get_feature_names()[j], weight=prob
        )

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10)
plt.show()

# Term Frequency Plot
plt.barh(tfidf_vectorizer.get_feature_names(), tfidf_matrix.toarray().sum(axis=0))
plt.xlabel("TF-IDF Score")
plt.ylabel("Term")
plt.show()

# Bubble Chart
plt.scatter(
    tfidf_matrix.toarray()[:, 0],
    tfidf_matrix.toarray()[:, 1],
    s=tfidf_matrix.toarray().sum(axis=1) * 100,
)
plt.xlabel("TF-IDF Feature 0")
plt.ylabel("TF-IDF Feature 1")
plt.show()

# Cluster Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(tfidf_matrix.toarray(), "ward")
dendrogram(Z, labels=tfidf_vectorizer.get_feature_names(), orientation="right")
plt.show()


# Which of the visualization methods would be useful for both LDA and TF-IDF clusters? Write down at least three ideas that could be applied to both types of clusters and explain why each idea would be useful.
# 1. Cluster Analysis: Perform cluster analysis to identify the optimal number of clusters in the corpus. This can help determine the appropriate number of clusters for both LDA and TF-IDF clusters.
# Thats not a visualization method
# Okay, ill start again
# 1. Network Graph: Visualize the relationships between topics and features as nodes and edges in a network graph. This can help identify the connections between different topics and features and how they are related to each other.
# 2. Radar Chart: Create a radar chart to compare the importance of different words in each topic and feature in the corpus. This can help identify the key terms and features that define each topic and how they differ from each other.
# 3. Heatmap: Generate a heatmap to visualize the distribution of topics and features across different documents. This can help identify the prevalence of different topics and features in the corpus and how they are distributed.
# 4. Word Cloud: Create a word cloud to visualize the most important terms and features in the corpus based on TF-IDF scores. This can help identify the key terms and features that define the corpus and how they are weighted.
# 5. Bar Chart: Generate a bar chart to compare the TF-IDF scores of different terms and features in the corpus. This can help identify the most important terms and features and how they contribute to the overall TF-IDF scores.

# Let's implement some of the visualization ideas mentioned above for both LDA and TF-IDF clusters. Write the code to implement the visualization ideas for each type of cluster and explain the insights gained from each visualization.
# Network Graph
# For this visualization, we will create a network graph to visualize the relationships between topics and features in the corpus.

import networkx as nx

G = nx.Graph()

# Add topics as nodes with weights based on their scores in the LDA matrix (rows) and features as nodes with weights based on their scores in the TF-IDF matrix (columns)
for i, topic in enumerate(lda_model.components_):
    G.add_node(f"Topic {i}", bipartite=0, weight=topic.sum())
    for j, prob in enumerate(topic):
        G.add_edge(f"Topic {i}", f"Feature {j}", weight=prob)

for i, feature in enumerate(tfidf_matrix.T):
    G.add_node(f"Feature {i}", bipartite=1, weight=feature.sum())

# Create a layout for the graph
pos = nx.bipartite_layout(G, nodes=[f"Topic {i}" for i in range(lda_matrix.shape[0])])

# Draw the graph with nodes colored by their bipartite set (topics in blue, features in red) and edges colored by their weights (probabilities) between topics and features based on the LDA and TF-IDF matrices respectively
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=["blue" if "Topic" in node else "red" for node in G.nodes],
    edge_color=[G[u][v]["weight"] for u, v in G.edges],
    edge_cmap=plt.cm.Blues,
    edge_vmin=0,
    edge_vmax=1,
    width=2,
    edge_color="black",
)
plt.show()
# The network graph visualizes the relationships between topics and features in the corpus. It shows how topics are connected to different features based on the probability distribution of features in each topic.
# an alternative to this might be to use a flow chart to visualize the progression of topics and features in the corpus. I'll code that too

# Flow Chart
# First, I'll create a flow chart to visualize the progression of topics and features in the corpus. Specifically, I'll create a Sankey diagram with topics and features as flows and labels to show the flow of topics to features based on their weights. This will help us identify the sequence of topics and features and how they are related to each other.

# Create a Sankey diagram
sankey = Sankey()

# Add topics as flows with weights based on their scores in the LDA matrix (rows) and features as labels with weights based on their scores in the TF-IDF matrix (columns)
for i, topic in enumerate(lda_model.components_):
    sankey.add(flows=[topic.sum()], labels=[f"Topic {i}"])

for i, feature in enumerate(tfidf_matrix.T):
    sankey.add(flows=[feature.sum()], labels=[f"Feature {i}"])

# Finish the Sankey diagram
sankey.finish()
plt.show()
# The flow chart visualizes the progression of topics and features in the corpus. It shows the sequence of topics and features and how they are related to each other based on their weights.
# an alternative to this might be to use a tree map to visualize the hierarchy of topics and features in the corpus. I'll code that too

# Tree Map
# First, I'll create a tree map to visualize the hierarchy of topics and features in the corpus. Specifically, I'll create a squarify plot with topics and features as labels and sizes based on their weights in the LDA and TF-IDF matrices respectively. This will help us identify the main topics and features and how they are organized.

# Create a squarify plot
sizes = np.concatenate((lda_model.components_.sum(axis=1), tfidf_matrix.sum(axis=0)))
labels = np.concatenate(
    (["Topic"] * lda_model.shape[0], ["Feature"] * tfidf_matrix.shape[1])
)

squarify.plot(sizes=sizes, label=labels, alpha=0.7)
plt.axis("off")
plt.show()
# The tree map visualizes the hierarchy of topics and features in the corpus. It shows the main topics and features and how they are organized based on their weights.


# Clustering Analysis
# We will be creating several sets of clusters using different clustering algorithms and visualizing the results.
# List of clustering algorithms to use: KMeans, Agglomerative Clustering, HDBSCAN, Spectral Clustering and Gaussian Mixture Model. Maybe also could try fuzzy clustering algorithms like Fuzzy C-Means or Fuzzy K-Means, and it might also be interesting to experiment with clustering algorithms that can handle mixed data types like K-Prototypes or KModes or even Categorical K-Means. Also K-nearest neighbors clustering or variant of KNN like KNNG, KNN-DTW, or KNN++.
# Note: We will be using the combined features (TF-IDF and LDA) for clustering analysis at first.
# I'll start by performing KMeans clustering on the combined features and visualizing the clusters using some of the ideas mentioned above. I'll also experiment with other clustering algorithms to see how they compare to KMeans in terms of cluster quality and interpretability. I'll code the clustering analysis and visualization for each algorithm and explain the insights gained from each clustering method.

# Perform KMeans clustering on the combined features first and then visualize the clusters using some of the ideas mentioned above.

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(combined_features)
kmeans_clusters = kmeans.predict(combined_features)

# Display Results
df_kmeans = pd.DataFrame({"Document": documents, "Cluster": kmeans_clusters})

logging.info(df_kmeans)

# Before visualizing the KMeans clusters, we'll also perform Agglomerative Clustering, HDBSCAN, Spectral Clustering, and Gaussian Mixture Model clustering on the combined features to compare the results and see how they differ in terms of cluster quality and interpretability. I'll code that too

# Agglomerative Clustering
# Planning: There are several parameters that can be tuned in Agglomerative Clustering, such as the number of clusters, the linkage criterion, and the distance metric. I'll set the number of clusters to 3, the linkage criterion to 'ward', and the distance metric to 'euclidean' to ensure that the clusters are well-separated and interpretable. I'll also set the affinity parameter to 'euclidean' to calculate the distance between instances based on the Euclidean distance metric. Other parameters like memory=None and compute_full_tree='auto' are set to ensure that the algorithm runs efficiently and scales well to large datasets. I'll create an Agglomerative Clustering object with these parameters and fit it to the combined features to perform clustering analysis. I'll also predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# Are you sure we should use euclidean distance? I think it would be better to use cosine distance since we are working with text data. I'll change that. Also, what about the linkage criterion? I think it would be better to use 'average' or 'complete' instead of 'ward'. I'll change that too
# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# also make a k-nearest neighbors graph to use as connectivity matrix
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(combined_features, n_neighbors=10, include_self=False)


agg = AgglomerativeClustering(
    n_clusters=3,
    affinity="cosine",
    linkage="average",
    compute_full_tree="auto",
    memory=None,  # This parameter is set to None to ensure that the algorithm runs efficiently and scales well to large datasets. It could be set to a string to specify the directory where the algorithm will cache the precomputed distances between instances, but in our case, it's not necessary.
    connectivity=connectivity,  # This parameter is set to the k-nearest neighbors graph to use as the connectivity matrix for the clustering algorithm. It could be set to None to use the Euclidean distance between instances as the connectivity matrix, but in our case, it's better to use the k-nearest neighbors graph to capture the local structure of the data.
)
agg_clusters = agg.fit_predict(combined_features)

# Display Results
df_agg = pd.DataFrame({"Document": documents, "Cluster": agg_clusters})

logging.info(df_agg)

# HDBSCAN Clustering
# Planning: There are several parameters that can be tuned in HDBSCAN, such as the minimum cluster size, the minimum samples, and the cluster selection method. I'll set the minimum cluster size to 5, the minimum samples to 5, and the cluster selection method to 'eom' to ensure that the clusters are well-separated and interpretable. I'll also set the metric parameter to 'euclidean' to calculate the distance between instances based on the Euclidean distance metric. Other parameters like algorithm='best' and core_dist_n_jobs=None are set to ensure that the algorithm runs efficiently and scales well to large datasets. I'll create an HDBSCAN object with these parameters and fit it to the combined features to perform clustering analysis. I'll also predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# HDBSCAN Clustering
from hdbscan import HDBSCAN


# As parameters to the HDBSCAN function, thinking step by step and considering our data is text, the following parameters make more sense:
# min_cluster_size: We should calculate this value based on the number of documents we have and the expected number of clusters. For example, if we have 100 documents and we expect 3 clusters, we could set min_cluster_size=30 to ensure that each cluster contains at least 30 documents. This would help us capture the main clusters in the data and ensure that they are well-separated and interpretable. It could be set to an integer to specify the minimum cluster size, but in our case, it's better to calculate this value based on the number of documents and the expected number of clusters.
# min_samples: We should set this value based on the number of samples in a neighborhood to ensure that the clusters are well-separated and interpretable. For example, if we have 100 documents and we expect 3 clusters, we could set min_samples=10 to ensure that each cluster has at least 10 samples. This would help us capture the main clusters in the data and ensure that they are well-separated and interpretable. It could be set to an integer to specify the minimum number of samples in a neighborhood, but in our case, it's better to calculate this value based on the number of documents and the expected number of clusters.
# cluster_selection_method='eom' to set the cluster selection method to 'eom' (excess of mass), which is a robust method for selecting clusters in high-dimensional data. It could be set to 'leaf' or 'leaf-dm' to use the leaf cluster selection method, but in our case, 'eom' is more suitable for text data because it considers the density of the clusters. Leaf cluster selection method is different from eom, it's based on the leaf nodes of the cluster hierarchy, while eom is based on the excess of mass of the clusters. In practice this means that leaf cluster selection method is more sensitive to the shape of the clusters, while eom is more robust to outliers and noise. In our case, eom is more suitable because it can handle the noise and outliers in the text data better than the leaf cluster selection method.
# metric='euclidean' to set the distance metric to 'euclidean' to calculate the distance between instances based on the Euclidean distance metric, which is suitable for text data. Is it really suitable for text data? How does cosine compare to euclidean distance for our use case? Well, in the difference in the results would be the following. Euclidean would be more sensitive to the magnitude of the vectors, while cosine would be more sensitive to the direction of the vectors. This means that Euclidean distance would be more affected by the length of the documents, while cosine distance would be more affected by the similarity of the terms in the documents. In our case, this would result, for example, in two documents with the same terms but different lengths being considered more similar by cosine distance than by Euclidean distance. In general, cosine distance is more suitable for text data because it captures the similarity of the terms in the documents regardless of their length. So, I'll change the metric parameter to 'cosine' to calculate the distance between instances based on the cosine distance metric, which is more suitable for text data.
# algorithm='best' to set the algorithm to 'best' to automatically choose the best algorithm based on the data and the parameters provided. This ensures that the clustering algorithm performs well on the given data and produces high-quality clusters.
# core_dist_n_jobs=None to set the number of parallel jobs to use for computing core distances to None to use all available CPU cores, which can speed up the computation of core distances for large datasets. It could be set to an integer to specify the number of CPU cores to use, but in our case, it's not necessary.
# I'll create an HDBSCAN object with these parameters and fit it to the combined features to perform clustering analysis. I'll also predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# HDBSCAN Clustering

num_documents = len(documents)

# Define the percentage range for min_cluster_size, using if branches to scale the value according to a few specified ranges of num_documents
ranges = [
    (
        1,
        10,
    ),  # This would be, according to multipliers_per_range, 0.50, or between 1 and 5
    (
        11,
        20,
    ),  # This would be, according to multipliers_per_range, 0.40, or between 6 and 8
    (
        21,
        50,
    ),  # This would be, according to multipliers_per_range, 0.30, or between 9 and 15
    (
        51,
        100,
    ),  # This would be, according to multipliers_per_range, 0.20, or between 16 and 20
    (
        101,
        1000,
    ),  # This would be, according to multipliers_per_range, 0.10, or between 21 and 100
    (1001, 2000),
    (2001, 3000),
    (3001, 5000),
    (5001, 10000),
    (10001, 12000),
    (12001, 15000),
    (15001, 20000),
    (20001, 100000),
    (100001, 200000),
    (200001, 300000),
    (300001, 400000),
    (400001, 500000),
    (500001, 600000),
    (600001, 700000),
    (700001, 800000),
    (800001, 900000),
    (900001, 1000000),
]
# multipliers_per_range = [0.
# Shouldnt it start kinda high, going down as the number of documents increase?
# Yes, you're right. I'll update the multipliers_per_range list to start high and decrease as the number of documents increase. I'll also add a placeholder value for the last range to ensure that the multipliers_per_range list has the same length as the ranges list. I'll code that too
# multipliers_per_range = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.0001]
# Hey wait I thought of someting, what about the number of features? Shouldn't we also consider the number of features in the dataset when calculating the min_cluster_size and min_samples? Yes, you're right. The number of features could also affect the clustering analysis, especially if the number of features is large. I'll update the ranges list to include the number of features as well and calculate the min_cluster_size and min_samples based on both the number of documents and the number of features. I'll code that too
# I'll calculate the min_cluster_size and min_samples based on the number of documents and the number of features. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
# No, I meant maybe we should calculate the multiplier based on the number of features as well. Remember, this will be used to determine the min_cluster_size. So, for example, if we have 100 documents and 100 features, the multiplier should be higher than if we have 100 documents and 10 features. Likewise, if we have 100 documents and 1000 features, we want it to depend on the number of features as well. So, we could calculate the multiplier based on the proportion of the number of documents to the number of features, and then use that multiplier to calculate the min_cluster_size and min_samples. I'll code that too
# So are you saying that it is more dependent on the number of features than the number of documents? Yes, that's correct. The number of features could have a significant impact on the clustering analysis, especially if the number of features is large. So, it's important to consider the number of features when calculating the min_samples and min_cluster_size. I'll update the multipliers_per_range list to include the number of features as well and calculate the min_cluster_size and min_samples based on both the number of documents and the number of features. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
# How should the number of documents factor in, then? The number of documents could also affect the clustering analysis, especially if the number of documents is large. So, it's important to consider the number of documents when calculating the min_samples and min_cluster_size. I'll update the multipliers_per_range list to include the number of documents as well and calculate the min_cluster_size and min_samples based on both the number of documents and the number of features. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
# Let's define those ratios. If there are more documents than features, we should prioritize the number of documents. If there are more features than documents, we should prioritize the number of features. If the number of documents is equal to the number of features, we should balance the number of documents and features. I'll code that too
#  multiplier = min(num_documents, num_features) / max(num_documents, num_features) if num_documents != num_features else 1.0. For example, if there are many more documents than features, like 1000 documents and 10 features, the multiplier would be 10 / 1000 = 0.01, resulting in clusters with a minimum size of 10 documents. If there are many more features than documents, like 10 documents and 1000 features, the multiplier would be 10 / 1000 = 0.01, resulting in clusters with a minimum size of 10 documents. That doesnt seem right, does it?
multipliers_per_range = [
    0.50,  # for 1-10 documents
    0.40,  # for 11-20 documents
    0.30,  # for 21-50 documents
    0.20,  # for 51-100 documents
    0.10,  # for 101-1000 documents
    0.05,  # for 1001-2000 documents
    0.04,  # for 2001-3000 documents
    0.03,  # for 3001-5000 documents
    0.02,  # for 5001-10000 documents
    0.01,  # for 10001-12000 documents
    0.005,  # for 12001-15000 documents
    0.004,  # for 15001-20000 documents
    0.003,  # for 20001-100000 documents
    0.002,  # for 100001-200000 documents
    0.001,  # for >200000 documents
]


min_percentage = 0.00  # placeholder
max_percentage = 0.00  # placeholder
# I'll iterate over the ranges to find the range that contains the number of documents and calculate the min_cluster_size and min_samples based on the range of num_documents. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
for i, (min_range, max_range) in enumerate(ranges):
    if min_range <= num_documents <= max_range:
        min_percentage = multipliers_per_range[i]
        max_percentage = multipliers_per_range[i + 1]
        break

# Calculate min and max values for min_cluster_size
min_cluster_size_min = int(num_documents * min_percentage)
min_cluster_size_max = int(num_documents * max_percentage)

# Define a modulus value to determine how much to subtract for min_samples
modulus = 5

# Calculate min_samples based on min_cluster_size
min_samples_min = min_cluster_size_min - (min_cluster_size_min % modulus)
min_samples_max = min_cluster_size_max - (min_cluster_size_max % modulus)

# Ensure min_samples is not less than 1, and if min_cluster_size is in a ratio of proportion of the number of documents of
min_samples_min = max(min_samples_min, 1)
min_samples_max = max(min_samples_max, 1)

# Print the calculated values
print(f"min_cluster_size range: {min_cluster_size_min} to {min_cluster_size_max}")
print(f"min_samples range: {min_samples_min} to {min_samples_max}")
hdbscan = HDBSCAN(
    min_cluster_size=min_cluster_size_min,
    min_samples=min_samples_min,
    cluster_selection_method="eom",
    metric="cosine",
    algorithm="best",
    core_dist_n_jobs=None,
    cluster_selection_epsilon=0.2,
)
# I'll fit the HDBSCAN object to the combined features to perform clustering analysis and predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# HDBSCAN Clustering

hdbscan.fit(combined_features)
hdbscan_clusters = hdbscan.labels_

# Display Results
df_hdbscan = pd.DataFrame({"Document": documents, "Cluster": hdbscan_clusters})

logging.info(df_hdbscan)

# Let's visualize the HDBSCAN clusters using some of the ideas mentioned above. Write the code to implement the visualization ideas for HDBSCAN clusters and explain the insights gained from each visualization.
# HDBSCAN Clusters
# Scatter Plot
plt.scatter(combined_features[:, 0], combined_features[:, 1], c=hdbscan_clusters)
plt.xlabel("TF-IDF Feature 0")
plt.ylabel("LDA Topic 0")
plt.show()
# The scatter plot visualizes the HDBSCAN clusters of documents based on the combined TF-IDF and LDA features. It shows the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
# an alternative to this might be to use a PCA scatter plot to visualize the HDBSCAN clusters in a lower-dimensional space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# PCA Scatter Plot (HDBSCAN). As arguments to the PCA function, I'll set n_components=2 to reduce the dimensionality of the combined features to 2D and visualize the clusters in a lower-dimensional space. I'll also normalize the combined features before applying PCA to ensure that the features are on the same scale. Other parameters like random_state=42 and svd_solver='full' are set to ensure reproducibility and stability of the results. Theres also a parameter called whiten=True which is set to True to ensure that the components are uncorrelated and have unit variance, which is a common practice in PCA, in our case, it might not be necessary, but it's good to have it there just in case. I'll create a PCA scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using PCA. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
pca = PCA(n_components=2, random_state=42, svd_solver="full", whiten=True)
pca_features = pca.fit_transform(combined_features)
# I'll create a PCA scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using PCA. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=hdbscan_clusters)
# I'll set the x-axis label to "PCA Feature 0" and the y-axis label to "PCA Feature 1" to indicate the components of the PCA features. I'll also display the scatter plot to visualize the HDBSCAN clusters of documents in a lower-dimensional space.
plt.xlabel("PCA Feature 0")
plt.ylabel("PCA Feature 1")
plt.show()
# The PCA scatter plot visualizes the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using PCA. It shows the clusters of documents with similar TF-IDF and LDA scores in a lower-dimensional space and how they are related to each other.
# an alternative to this might be to use a t-SNE scatter plot to visualize the HDBSCAN clusters in a non-linear space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# t-SNE Scatter Plot (HDBSCAN). As arguments to the TSNE function, I'll set n_components=2 to reduce the dimensionality of the combined features to 2D and visualize the clusters in a lower-dimensional space. I'll also set random_state=42 to ensure reproducibility of the results. Other parameters like perplexity=30 and n_iter=1000 are set to control the quality of the embedding and the number of iterations to perform. I'll create a t-SNE scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using t-SNE. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_features = tsne.fit_transform(combined_features)
# I'll create a t-SNE scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using t-SNE. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=hdbscan_clusters)
plt.xlabel("t-SNE Feature 0")
plt.ylabel("t-SNE Feature 1")
plt.show()
# The t-SNE scatter plot visualizes the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using t-SNE. It shows the clusters of documents with similar TF-IDF and LDA scores in a non-linear space and how they are related to each other.
# an alternative to this might be to use a UMAP scatter plot to visualize the HDBSCAN clusters in a lower-dimensional space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# UMAP Scatter Plot (HDBSCAN). As arguments to the UMAP function, I'll set n_components=2 to reduce the dimensionality of the combined features to 2D and visualize the clusters in a lower-dimensional space. I'll also set random_state=42 to ensure reproducibility of the results. Other parameters like n_neighbors=15 and min_dist=0.1 are set to control the number of neighbors to consider and the minimum distance between points in the embedding. I'll create a UMAP scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using UMAP. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_features = reducer.fit_transform(combined_features)
# I'll create a UMAP scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using UMAP. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
plt.scatter(umap_features[:, 0], umap_features[:, 1], c=hdbscan_clusters)
plt.xlabel("UMAP Feature 0")
plt.ylabel("UMAP Feature 1")
plt.show()
# The UMAP scatter plot visualizes the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using UMAP. It shows the clusters of documents with similar TF-IDF and LDA scores in a lower-dimensional space and how they are related to each other.
# an alternative to this might be to use a heatmap to visualize the combined features in a 2D space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# Heatmap
plt.imshow(combined_features, aspect="auto")
plt.colorbar()
plt.xlabel("Feature")
plt.ylabel("Document")
plt.show()
# The heatmap visualizes the combined TF-IDF and LDA features in a 2D space. It shows the distribution of features across different documents and how they are related to each other.
# an alternative to this might be to use a correlation matrix to identify any relationships between the TF-IDF and LDA features and how they are correlated. I'll code that too

# Correlation Matrix
correlation_matrix = np.corrcoef(combined_features.T)

plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar()
plt.xlabel("Feature")
plt.ylabel("Feature")
plt.show()
# The correlation matrix visualizes the relationships between the TF-IDF and LDA features. It shows the correlations between different features and how they are related to each other.
# an alternative to this might be to use a parallel coordinates plot to compare the importance of different words and topics in the corpus and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# Parallel Coordinates Plot
pd.plotting.parallel_coordinates(
    pd.DataFrame(
        combined_features,
        columns=["TF-IDF Feature 0", "TF-IDF Feature 1", "LDA Topic 0", "LDA Topic 1"],
    ),
    "Cluster",
)
plt.show()
# The parallel coordinates plot compares the TF-IDF and LDA scores of different terms and topics in the corpus. It shows how the TF-IDF and LDA features vary between clusters and how they are related to each other.
# an alternative to this might be to use a radar chart to compare the importance of different words in each topic and feature in the corpus. I'll code that too

# Radar Chart
fig = px.line_polar(
    pd.DataFrame(lda_model.components_, columns=count_vectorizer.get_feature_names())
)
fig.show()
# The radar chart compares the importance of different words in each topic. It shows the key terms that define each topic and how they differ from each other based on their weights.
# an alternative to this might be to use a sunburst plot to visualize the relationships between topics and features in the corpus. I'll code that too

# Sunburst Plot
fig = px.sunburst(
    pd.DataFrame(lda_matrix, columns=["Topic 0", "Topic 1", "Topic 2"]),
    path=["Topic 0", "Topic 1", "Topic 2"],
)
fig.show()
# The sunburst plot visualizes the relationships between topics and features in the corpus. It shows the connections between different topics and how they are distributed across the corpus.
# an alternative to this might be to use a chord diagram to visualize the relationships between topics and features in the corpus. I'll code that too

# Chord Diagram
import matplotlib.pyplot as plt
from chord import Chord

# I'll create a chord diagram to visualize the relationships between topics and features in the corpus. Specifically, I'll create a chord diagram with topics and features as nodes and edges connecting topics to features based on their weights. This will help us identify the connections between different topics and features and how they are related to each other.

# First, I'll transpose the LDA matrix to get the topics as rows and features as columns
doc_topic_matrix = lda_matrix.T
# Next, I'll normalize the matrix to get the probability distribution of features in each topic with respect to the total sum of features in each topic, so that the sum of each row is equal to 1. Summing across the rows (axis=1) will give the total sum of features in each topic
doc_topic_matrix = doc_topic_matrix / doc_topic_matrix.sum(
    axis=1
)  # Normalize the matrix to get the probability distribution of features in each topic

# Finally, I'll create a chord diagram with the normalized matrix and the feature names as labels
chord = Chord(doc_topic_matrix, count_vectorizer.get_feature_names())
# Save the chord diagram to an HTML file
chord.to_html("chord.html")

import webbrowser

try:
    # Display the chord diagram by opening the HTML file in a web browser automatically
    webbrowser.open("chord.html")
except Exception as e:
    print(f"An error occurred while trying to open the file: {e}")

# The chord diagram visualizes the relationships between topics and features in the corpus. It shows the connections between different topics and how they are distributed across the documents.
# an alternative to this might be to use a network graph to visualize the relationships between topics and features in the corpus. I'll code that too
# Network Graph
import networkx as nx

# I'll create a network graph to visualize the relationships between topics and features in the corpus. Specifically, I'll create a bipartite graph with topics and features as nodes and edges connecting topics to features based on their weights. This will help us identify the connections between different topics and features and how they are related to each other.

# Create a bipartite graph
G = nx.Graph()

# Add topics as nodes with weights based on their scores in the LDA matrix (rows) and features as nodes with weights based on their scores in the TF-IDF matrix (columns)
for i, topic in enumerate(lda_model.components_):
    G.add_node(f"Topic {i}", bipartite=0, weight=topic.sum())
    for j, prob in enumerate(topic):
        G.add_edge(f"Topic {i}", f"Feature {j}", weight=prob)

for i, feature in enumerate(tfidf_matrix.T):
    G.add_node(f"Feature {i}", bipartite=1, weight=feature.sum())

# Create a layout for the graph
pos = nx.bipartite_layout(G, nodes=[f"Topic {i}" for i in range(lda_matrix.shape[0])])

# Draw the graph with nodes colored by their bipartite set (topics in blue, features in red) and edges colored by their weights (probabilities) between topics and features based on the LDA and TF-IDF matrices respectively
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=["blue" if "Topic" in node else "red" for node in G.nodes],
    edge_color=[G[u][v]["weight"] for u, v in G.edges],
    edge_cmap=plt.cm.Blues,
    edge_vmin=0,
    edge_vmax=1,
    width=2,
    edge_color="black",
)
plt.show()
# The network graph visualizes the relationships between topics and features in the corpus. It shows how topics are connected to different features based on the probability distribution of features in each topic.
# an alternative to this might be to use a flow chart to visualize the progression of topics and features in the corpus. I'll code that too

# Flow Chart
# First, I'll create a flow chart to visualize the progression of topics and features in the corpus. Specifically, I'll create a Sankey diagram with topics and features as flows and labels to show the flow of topics to features based on their weights. This will help us identify the sequence of topics and features and how they are related to each other.

# Create a Sankey diagram
sankey = Sankey()

# Add topics as flows with weights based on their scores in the LDA matrix (rows) and features as labels with weights based on their scores in the TF-IDF matrix (columns)
for i, topic in enumerate(lda_model.components_):
    sankey.add(flows=[topic.sum()], labels=[f"Topic {i}"])

for i, feature in enumerate(tfidf_matrix.T):
    sankey.add(flows=[feature.sum()], labels=[f"Feature {i}"])

# Finish the Sankey diagram
sankey.finish()
plt.show()
# The flow chart visualizes the progression of topics and features in the corpus. It shows the sequence of topics and features and how they are related to each other based on their weights.
# an alternative to this might be to use a tree map to visualize the hierarchy of topics and features in the corpus. I'll code that too

# Tree Map
# First, I'll create a tree map to visualize the hierarchy of topics and features in the corpus. Specifically, I'll create a squarify plot with topics and features as labels and sizes based on their weights in the LDA and TF-IDF matrices respectively. This will help us identify the main topics and subtopics and how they are organized.

# Create a squarify plot
sizes = np.concatenate((lda_model.components_.sum(axis=1), tfidf_matrix.sum(axis=0)))
labels = np.concatenate(
    (["Topic"] * lda_model.shape[0], ["Feature"] * tfidf_matrix.shape[1])
)

squarify.plot(sizes=sizes, label=labels, alpha=0.7)
plt.axis("off")
plt.show()
# The tree map visualizes the hierarchy of topics and features in the corpus. It shows the main topics and features and how they are organized based on their weights.


# Clustering Analysis
# We will be creating several sets of clusters using different clustering algorithms and visualizing the results.
# List of clustering algorithms to use: KMeans, Agglomerative Clustering, HDBSCAN, Spectral Clustering and Gaussian Mixture Model. Maybe also could try fuzzy clustering algorithms like Fuzzy C-Means or Fuzzy K-Means, and it might also be interesting to experiment with clustering algorithms that can handle mixed data types like K-Prototypes or KModes or even Categorical K-Means. Also K-nearest neighbors clustering or variant of KNN like KNNG, KNN-DTW, or KNN++.
# Note: We will be using the combined features (TF-IDF and LDA) for clustering analysis at first.
# I'll start by performing KMeans clustering on the combined features and visualizing the clusters using some of the ideas mentioned above. I'll also experiment with other clustering algorithms to see how they compare to KMeans in terms of cluster quality and interpretability. I'll code the clustering analysis and visualization for each algorithm and explain the insights gained from each clustering method.

# Perform KMeans clustering on the combined features first and then visualize the clusters using some of the ideas mentioned above.

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(combined_features)
kmeans_clusters = kmeans.predict(combined_features)

# Display Results
df_kmeans = pd.DataFrame({"Document": documents, "Cluster": kmeans_clusters})

logging.info(df_kmeans)

# Before visualizing the KMeans clusters, we'll also perform Agglomerative Clustering, HDBSCAN, Spectral Clustering, and Gaussian Mixture Model clustering on the combined features to compare the results and see how they differ in terms of cluster quality and interpretability. I'll code that too

# Agglomerative Clustering
# Planning: There are several parameters that can be tuned in Agglomerative Clustering, such as the number of clusters, the linkage criterion, and the distance metric. I'll set the number of clusters to 3, the linkage criterion to 'ward', and the distance metric to 'euclidean' to ensure that the clusters are well-separated and interpretable. I'll also set the affinity parameter to 'euclidean' to calculate the distance between instances based on the Euclidean distance metric. Other parameters like memory=None and compute_full_tree='auto' are set to ensure that the algorithm runs efficiently and scales well to large datasets. I'll create an Agglomerative Clustering object with these parameters and fit it to the combined features to perform clustering analysis. I'll also predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# Are you sure we should use euclidean distance? I think it would be better to use cosine distance since we are working with text data. I'll change that. Also, what about the linkage criterion? I think it would be better to use 'average' or 'complete' instead of 'ward'. I'll change that too
# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

# also make a k-nearest neighbors graph to use as connectivity matrix
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(combined_features, n_neighbors=10, include_self=False)


agg = AgglomerativeClustering(
    n_clusters=3,
    affinity="cosine",
    linkage="average",
    compute_full_tree="auto",
    memory=None,  # This parameter is set to None to ensure that the algorithm runs efficiently and scales well to large datasets. It could be set to a string to specify the directory where the algorithm will cache the precomputed distances between instances, but in our case, it's not necessary.
    connectivity=connectivity,  # This parameter is set to the k-nearest neighbors graph to use as the connectivity matrix for the clustering algorithm. It could be set to None to use the Euclidean distance between instances as the connectivity matrix, but in our case, it's better to use the k-nearest neighbors graph to capture the local structure of the data.
)
agg_clusters = agg.fit_predict(combined_features)

# Display Results
df_agg = pd.DataFrame({"Document": documents, "Cluster": agg_clusters})

logging.info(df_agg)

# HDBSCAN Clustering
# Planning: There are several parameters that can be tuned in HDBSCAN, such as the minimum cluster size, the minimum samples, and the cluster selection method. I'll set the minimum cluster size to 5, the minimum samples to 5, and the cluster selection method to 'eom' to ensure that the clusters are well-separated and interpretable. I'll also set the metric parameter to 'euclidean' to calculate the distance between instances based on the Euclidean distance metric. Other parameters like algorithm='best' and core_dist_n_jobs=None are set to ensure that the algorithm runs efficiently and scales well to large datasets. I'll create an HDBSCAN object with these parameters and fit it to the combined features to perform clustering analysis. I'll also predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# HDBSCAN Clustering
from hdbscan import HDBSCAN


# As parameters to the HDBSCAN function, thinking step by step and considering our data is text, the following parameters make more sense:
# min_cluster_size: We should calculate this value based on the number of documents we have and the expected number of clusters. For example, if we have 100 documents and we expect 3 clusters, we could set min_cluster_size=30 to ensure that each cluster contains at least 30 documents. This would help us capture the main clusters in the data and ensure that they are well-separated and interpretable. It could be set to an integer to specify the minimum cluster size, but in our case, it's better to calculate this value based on the number of documents and the expected number of clusters.
# min_samples: We should set this value based on the number of samples in a neighborhood to ensure that the clusters are well-separated and interpretable. For example, if we have 100 documents and we expect 3 clusters, we could set min_samples=10 to ensure that each cluster has at least 10 samples. This would help us capture the main clusters in the data and ensure that they are well-separated and interpretable. It could be set to an integer to specify the minimum number of samples in a neighborhood, but in our case, it's better to calculate this value based on the number of documents and the expected number of clusters.
# cluster_selection_method='eom' to set the cluster selection method to 'eom' (excess of mass), which is a robust method for selecting clusters in high-dimensional data. It could be set to 'leaf' or 'leaf-dm' to use the leaf cluster selection method, but in our case, 'eom' is more suitable for text data because it considers the density of the clusters. Leaf cluster selection method is different from eom, it's based on the leaf nodes of the cluster hierarchy, while eom is based on the excess of mass of the clusters. In practice this means that leaf cluster selection method is more sensitive to the shape of the clusters, while eom is more robust to outliers and noise. In our case, eom is more suitable because it can handle the noise and outliers in the text data better than the leaf cluster selection method.
# metric='euclidean' to set the distance metric to 'euclidean' to calculate the distance between instances based on the Euclidean distance metric, which is suitable for text data. Is it really suitable for text data? How does cosine compare to euclidean distance for our use case? Well, in the difference in the results would be the following. Euclidean would be more sensitive to the magnitude of the vectors, while cosine would be more sensitive to the direction of the vectors. This means that Euclidean distance would be more affected by the length of the documents, while cosine distance would be more affected by the similarity of the terms in the documents. In our case, this would result, for example, in two documents with the same terms but different lengths being considered more similar by cosine distance than by Euclidean distance. In general, cosine distance is more suitable for text data because it captures the similarity of the terms in the documents regardless of their length. So, I'll change the metric parameter to 'cosine' to calculate the distance between instances based on the cosine distance metric, which is more suitable for text data.
# algorithm='best' to set the algorithm to 'best' to automatically choose the best algorithm based on the data and the parameters provided. This ensures that the clustering algorithm performs well on the given data and produces high-quality clusters.
# core_dist_n_jobs=None to set the number of parallel jobs to use for computing core distances to None to use all available CPU cores, which can speed up the computation of core distances for large datasets. It could be set to an integer to specify the number of CPU cores to use, but in our case, it's not necessary.
# I'll create an HDBSCAN object with these parameters and fit it to the combined features to perform clustering analysis. I'll also predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# HDBSCAN Clustering

num_documents = len(documents)

# Define the percentage range for min_cluster_size, using if branches to scale the value according to a few specified ranges of num_documents
ranges = [
    (
        1,
        10,
    ),  # This would be, according to multipliers_per_range, 0.50, or between 1 and 5
    (
        11,
        20,
    ),  # This would be, according to multipliers_per_range, 0.40, or between 6 and 8
    (
        21,
        50,
    ),  # This would be, according to multipliers_per_range, 0.30, or between 9 and 15
    (
        51,
        100,
    ),  # This would be, according to multipliers_per_range, 0.20, or between 16 and 20
    (
        101,
        1000,
    ),  # This would be, according to multipliers_per_range, 0.10, or between 21 and 100
    (1001, 2000),
    (2001, 3000),
    (3001, 5000),
    (5001, 10000),
    (10001, 12000),
    (12001, 15000),
    (15001, 20000),
    (20001, 100000),
    (100001, 200000),
    (200001, 300000),
    (300001, 400000),
    (400001, 500000),
    (500001, 600000),
    (600001, 700000),
    (700001, 800000),
    (800001, 900000),
    (900001, 1000000),
]
# multipliers_per_range = [0.
# Shouldnt it start kinda high, going down as the number of documents increase?
# Yes, you're right. I'll update the multipliers_per_range list to start high and decrease as the number of documents increase. I'll also add a placeholder value for the last range to ensure that the multipliers_per_range list has the same length as the ranges list. I'll code that too
# multipliers_per_range = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.0001]
# Hey wait I thought of someting, what about the number of features? Shouldn't we also consider the number of features in the dataset when calculating the min_cluster_size and min_samples? Yes, you're right. The number of features could also affect the clustering analysis, especially if the number of features is large. I'll update the ranges list to include the number of features as well and calculate the min_cluster_size and min_samples based on both the number of documents and the number of features. I'll code that too
# I'll calculate the min_cluster_size and min_samples based on the number of documents and the number of features. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
# No, I meant maybe we should calculate the multiplier based on the number of features as well. Remember, this will be used to determine the min_cluster_size. So, for example, if we have 100 documents and 100 features, the multiplier should be higher than if we have 100 documents and 10 features. Likewise, if we have 100 documents and 1000 features, we want it to depend on the number of features as well. So, we could calculate the multiplier based on the proportion of the number of documents to the number of features, and then use that multiplier to calculate the min_cluster_size and min_samples. I'll code that too
# So are you saying that it is more dependent on the number of features than the number of documents? Yes, that's correct. The number of features could have a significant impact on the clustering analysis, especially if the number of features is large. So, it's important to consider the number of features when calculating the min_samples and min_cluster_size. I'll update the multipliers_per_range list to include the number of features as well and calculate the min_cluster_size and min_samples based on both the number of documents and the number of features. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
# How should the number of documents factor in, then? The number of documents could also affect the clustering analysis, especially if the number of documents is large. So, it's important to consider the number of documents when calculating the min_samples and min_cluster_size. I'll update the multipliers_per_range list to include the number of documents as well and calculate the min_cluster_size and min_samples based on both the number of documents and the number of features. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
# Let's define those ratios. If there are more documents than features, we should prioritize the number of documents. If there are more features than documents, we should prioritize the number of features. If the number of documents is equal to the number of features, we should balance the number of documents and features. I'll code that too
#  multiplier = min(num_documents, num_features) / max(num_documents, num_features) if num_documents != num_features else 1.0. For example, if there are many more documents than features, like 1000 documents and 10 features, the multiplier would be 10 / 1000 = 0.01, resulting in clusters with a minimum size of 10 documents. If there are many more features than documents, like 10 documents and 1000 features, the multiplier would be 10 / 1000 = 0.01, resulting in clusters with a minimum size of 10 documents. That doesnt seem right, does it?
multipliers_per_range = [
    0.50,  # for 1-10 documents
    0.40,  # for 11-20 documents
    0.30,  # for 21-50 documents
    0.20,  # for 51-100 documents
    0.10,  # for 101-1000 documents
    0.05,  # for 1001-2000 documents
    0.04,  # for 2001-3000 documents
    0.03,  # for 3001-5000 documents
    0.02,  # for 5001-10000 documents
    0.01,  # for 10001-12000 documents
    0.005,  # for 12001-15000 documents
    0.004,  # for 15001-20000 documents
    0.003,  # for 20001-100000 documents
    0.002,  # for 100001-200000 documents
    0.001,  # for >200000 documents
]


min_percentage = 0.00  # placeholder
max_percentage = 0.00  # placeholder
# I'll iterate over the ranges to find the range that contains the number of documents and calculate the min_cluster_size and min_samples based on the range of num_documents. I'll also ensure that min_samples is not less than 1 and that min_cluster_size is in a ratio of proportion of the number of documents of 1/10. I'll code that too
for i, (min_range, max_range) in enumerate(ranges):
    if min_range <= num_documents <= max_range:
        min_percentage = multipliers_per_range[i]
        max_percentage = multipliers_per_range[i + 1]
        break

# Calculate min and max values for min_cluster_size
min_cluster_size_min = int(num_documents * min_percentage)
min_cluster_size_max = int(num_documents * max_percentage)

# Define a modulus value to determine how much to subtract for min_samples
modulus = 5

# Calculate min_samples based on min_cluster_size
min_samples_min = min_cluster_size_min - (min_cluster_size_min % modulus)
min_samples_max = min_cluster_size_max - (min_cluster_size_max % modulus)

# Ensure min_samples is not less than 1, and if min_cluster_size is in a ratio of proportion of the number of documents of
min_samples_min = max(min_samples_min, 1)
min_samples_max = max(min_samples_max, 1)

# Print the calculated values
print(f"min_cluster_size range: {min_cluster_size_min} to {min_cluster_size_max}")
print(f"min_samples range: {min_samples_min} to {min_samples_max}")
hdbscan = HDBSCAN(
    min_cluster_size=min_cluster_size_min,
    min_samples=min_samples_min,
    cluster_selection_method="eom",
    metric="cosine",
    algorithm="best",
    core_dist_n_jobs=None,
    cluster_selection_epsilon=0.2,
)
# I'll fit the HDBSCAN object to the combined features to perform clustering analysis and predict the clusters for each document based on the combined features and display the results to see how the documents are assigned to different clusters. I'll code that too
# HDBSCAN Clustering

hdbscan.fit(combined_features)
hdbscan_clusters = hdbscan.labels_

# Display Results
df_hdbscan = pd.DataFrame({"Document": documents, "Cluster": hdbscan_clusters})

logging.info(df_hdbscan)

# Let's visualize the HDBSCAN clusters using some of the ideas mentioned above. Write the code to implement the visualization ideas for HDBSCAN clusters and explain the insights gained from each visualization.
# HDBSCAN Clusters
# Scatter Plot
plt.scatter(combined_features[:, 0], combined_features[:, 1], c=hdbscan_clusters)
plt.xlabel("TF-IDF Feature 0")
plt.ylabel("LDA Topic 0")
plt.show()
# The scatter plot visualizes the HDBSCAN clusters of documents based on the combined TF-IDF and LDA features. It shows the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
# an alternative to this might be to use a PCA scatter plot to visualize the HDBSCAN clusters in a lower-dimensional space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# PCA Scatter Plot (HDBSCAN). As arguments to the PCA function, I'll set n_components=2 to reduce the dimensionality of the combined features to 2D and visualize the clusters in a lower-dimensional space. I'll also normalize the combined features before applying PCA to ensure that the features are on the same scale. Other parameters like random_state=42 and svd_solver='full' are set to ensure reproducibility and stability of the results. Theres also a parameter called whiten=True which is set to True to ensure that the components are uncorrelated and have unit variance, which is a common practice in PCA, in our case, it might not be necessary, but it's good to have it there just in case. I'll create a PCA scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using PCA. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
pca = PCA(n_components=2, random_state=42, svd_solver="full", whiten=True)
pca_features = pca.fit_transform(combined_features)
# I'll create a PCA scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using PCA. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=hdbscan_clusters)
# I'll set the x-axis label to "PCA Feature 0" and the y-axis label to "PCA Feature 1" to indicate the components of the PCA features. I'll also display the scatter plot to visualize the HDBSCAN clusters of documents in a lower-dimensional space.
plt.xlabel("PCA Feature 0")
plt.ylabel("PCA Feature 1")
plt.show()
# The PCA scatter plot visualizes the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using PCA. It shows the clusters of documents with similar TF-IDF and LDA scores in a lower-dimensional space and how they are related to each other.
# an alternative to this might be to use a t-SNE scatter plot to visualize the HDBSCAN clusters in a non-linear space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# t-SNE Scatter Plot (HDBSCAN). As arguments to the TSNE function, I'll set n_components=2 to reduce the dimensionality of the combined features to 2D and visualize the clusters in a lower-dimensional space. I'll also set random_state=42 to ensure reproducibility of the results. Other parameters like perplexity=30 and n_iter=1000 are set to control the quality of the embedding and the number of iterations to perform. I'll create a t-SNE scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using t-SNE. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_features = tsne.fit_transform(combined_features)
# I'll create a t-SNE scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using t-SNE. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=hdbscan_clusters)
plt.xlabel("t-SNE Feature 0")
plt.ylabel("t-SNE Feature 1")
plt.show()
# The t-SNE scatter plot visualizes the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using t-SNE. It shows the clusters of documents with similar TF-IDF and LDA scores in a non-linear space and how they are related to each other.
# an alternative to this might be to use a UMAP scatter plot to visualize the HDBSCAN clusters in a lower-dimensional space and identify any patterns or relationships between the TF-IDF and LDA features. I'll code that too

# UMAP Scatter Plot (HDBSCAN). As arguments to the UMAP function, I'll set n_components=2 to reduce the dimensionality of the combined features to 2D and visualize the clusters in a lower-dimensional space. I'll also set random_state=42 to ensure reproducibility of the results. Other parameters like n_neighbors=15 and min_dist=0.1 are set to control the number of neighbors to consider and the minimum distance between points in the embedding. I'll create a UMAP scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using UMAP. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_features = reducer.fit_transform(combined_features)
# I'll create a UMAP scatter plot to visualize the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using UMAP. I'll color the points based on the HDBSCAN clusters to identify the clusters of documents with similar TF-IDF and LDA scores and how they are related to each other.
plt.scatter(umap_features[:, 0], umap_features[:, 1], c=hdbscan_clusters)
plt.xlabel("UMAP Feature 0")
plt.ylabel("UMAP Feature 1")
plt.show()
# The UMAP scatter plot visualizes the HDBSCAN clusters of documents in a 2D space after reducing the dimensionality using UMAP. It shows the clusters of documents with similar TF-IDF and LDA scores in a lower-dimensional space and how they are related to each other.


# Let's brainstorm some modern/common useful cluster visualization methods to visualize the LDA and TF-IDF clusters in a way that will provide useful insights. Write down at least six ideas for visualizing the combined clusters and explain why each idea would be useful.
# Hmmm... I think we could start by visualizing the clusters in a scatter plot to see if there are any discernible patterns or relationships between the clusters. This could help us identify any similarities or differences between the clusters and how they are distributed in the data.
# Next, we could create a dendrogram of the clusters to visualize the hierarchy of clusters and how they are related to each other. This could help us identify any subclusters or groupings within the clusters and how they are organized.
# We could also generate a silhouette plot of the clusters to visualize the silhouette scores of the clusters and how well-separated they are from each other. This could help us evaluate the quality of the clustering and how distinct the clusters are from each other.
# Another useful visualization method could be to create a cluster map to visualize the distribution of documents based on their cluster assignments. This could help us identify the clusters of documents with similar features and how they are related to each other.
# We could also generate a cluster dendrogram to visualize the hierarchy of clusters in the corpus. This could help us identify the main clusters and subclusters and how they are organized.
# Finally, we could create a cluster network graph to visualize the relationships between clusters and documents in the corpus. This could help us identify the connections between different clusters and how they are distributed across the corpus.
# Some other ideas I didne't mention are a parallel coordinates plot, a radar chart, a sunburst plot, a chord diagram, a network graph, a flow chart, and a tree map. I'll code some of these ideas to visualize the combined clusters and explain the insights gained from each visualization.

# But first, I'll analyze (step by step) each of the visualization methods I mentioned above to see which ones would be most useful for understanding the data and identifying patterns in the clusters. I'll rate each method on a scale of 1 to 5 based on its usefulness and explain why I chose that rating.
# Scatter Plot: I think a scatter plot would be useful for visualizing the clusters and identifying any patterns or relationships between the clusters. I'll rate this method a 4 because it can provide valuable insights into the distribution of clusters and how they are related to each other.
# Dendrogram: I think a dendrogram would be useful for visualizing the hierarchy of clusters and how they are related to each other. I'll rate this method a 3 because it can help us identify any subclusters or groupings within the clusters and how they are organized, but since we're using KMeans, it may not be as informative. But if we were using hierarchical clustering, it would be more useful.
# Silhouette Plot: I think a silhouette plot would be useful for visualizing the silhouette scores of the clusters and how well-separated they are from each other. I'll rate this method a 2 because it can help us evaluate the quality of the clustering and how distinct the clusters are from each other, but it may not provide as much insight into the relationships between clusters.
# Cluster Map: I think a cluster map would be useful for visualizing the distribution of documents based on their cluster assignments. I'll rate this method a 5 because it can help us identify the clusters of documents with similar features and how they are related to each other.
# Cluster Dendrogram: I think a cluster dendrogram would be useful for visualizing the hierarchy of clusters in the corpus. I'll rate this method a 3 because it can help us identify the main clusters and subclusters and how they are organized, but since we're using KMeans, it may not be as informative. So I'll use this later with HDBSCAN or Agglomerative Clustering.
# Cluster Network Graph: I think a cluster network graph would be useful for visualizing the relationships between clusters and documents in the corpus. I'll rate this method a 4 because it can help us identify the connections between different clusters and how they are distributed across the corpus.
# Parallel Coordinates Plot: I think a parallel coordinates plot would be useful for visualizing the relationships between clusters and documents in the corpus. I'll rate this method a 4 because it can help us identify the connections between different clusters and how they are distributed across the corpus.
# Radar Chart: I think a radar chart would be useful for visualizing the relationships between clusters and documents in the corpus. I'll rate this method a 4 because it can help us identify the connections between different clusters and how they are distributed across the corpus. However, it may not be as informative as other methods because it's not as commonly used for cluster visualization.
# Sunburst Plot: I think a sunburst plot would be useful for visualizing the relationships between clusters and documents in the corpus. This ones usefulness rating would be a 4 because it can help us identify the connections between different clusters and how they are distributed across the corpus. However, it may not be as informative as other methods because it's not as commonly used for cluster visualization.
# Chord Diagram: I think a chord diagram would be useful for visualizing the relationships between clusters and documents in the corpus. I'll rate this method a 4 because it can help us identify the connections between different clusters and how they are distributed across the corpus.
# Network Graph: I analyzed network graph's usefulness for cluster visualization as around 4. I'll rate this method a 4 because it can help us identify the connections between different clusters and how they are distributed across the corpus. Of course, this is assuming that the clusters are connected in some way. Also, it's not as commonly used for cluster visualization as other methods, but it can still provide valuable insights into the relationships between clusters.
# Flow Chart: I analyzed flow chart's usefulness for cluster visualization as around 3. I'll rate this method a 3 because it can help us identify the connections between different clusters and how they are distributed across the corpus. Although it's not as commonly used for cluster visualization as other methods, it can still provide valuable insights into the relationships between clusters, but on the downside, it may not be as informative as other methods for this because we're iterating through the clusters and not the documents. I'll account for this in the code via mathematical magic.(I'll basically just use the cluster centers, but I'll explain that in the code)


# Let's implement some of the visualization ideas mentioned above for the combined clusters. Write the code to implement the visualization ideas for each type of cluster and explain the insights gained from each visualization.
# Cluster Map


# LDA with Gensim
dictionary = corpora.Dictionary([doc.split() for doc in documents])
corpus = [dictionary.doc2bow(doc.split()) for doc in documents]

lda_gensim = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

for idx, topic in lda_gensim.print_topics(-1):
    logging.info("Topic: {} Word: {}".format(idx, topic))

# LDA with Gensim Clustering
lda_matrix_gensim = np.zeros((len(documents), 3))

for i, doc in enumerate(corpus):
    for topic, prob in lda_gensim[doc]:
        lda_matrix_gensim[i, topic] = prob

# Combine TF-IDF and LDA features
combined_features_gensim = np.hstack((tfidf_matrix.toarray(), lda_matrix_gensim))

# Normalize combined features
combined_features_gensim = normalize(combined_features_gensim)

# Clustering
kmeans_gensim = KMeans(n_clusters=3, random_state=42)
clusters_gensim = kmeans_gensim.fit_predict(combined_features_gensim)

# Display Results
df = pd.DataFrame({"Document": documents, "Cluster": clusters_gensim})

logging.info(df)

# End of compare_lda_tfidf.py
