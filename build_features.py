
import click
@click.command()
@click.pass_context

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

def build_tfidf(ctx):
    text_pipeline = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer())])

    joblib.dump(text_pipeline, ctx.obj['pipeline_output_file'])

@click.command()
@click.pass_context

def build_lsa(ctx):

    text_pipeline = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('truncated_svd', TruncatedSVD(n_components = 100))])

    joblib.dump(text_pipeline, ctx.obj['pipeline_output_file'])

@click.group()
@click.argument('pipeline_output_file',
                type=click.Path(exists=False, dir_okay=False))
@click.pass_context
def entry_point(ctx, pipeline_output_file):
    ctx.obj['pipeline_output_file'] = pipeline_output_file


if __name__ == '__main__':
    entry_point.add_command(build_tfidf)
    entry_point.add_command(build_lsa)
    entry_point(obj={})



