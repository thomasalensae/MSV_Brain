import dataset
import bert_embeddings
import dataset.config as config
import mlem_model
import visualisation_methods.umap as umap
import visualisation_methods.mds as mds

# Load dataset
df, X = dataset.import_dataset.load()

Y_layers = bert_embeddings.compute_embeddings(df, config.cfg)

models, fi_tables, score_objs = mlem_model.run_mlem_across_layers(X, Y_layers)

mlem_model.display_feature_importance(fi_tables)

umap.visualisation(df, Y_layers)
mds.visualisation(df, Y_layers)