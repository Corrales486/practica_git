import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


def build_tfidf(ctx):
    text_pipeline = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer())])

    joblib.dump(text_pipeline, ctx.obj['pipeline_output_file'])


def build_lsa(ctx):
    pass