import os

import click
import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def read_dataset(dataset_file):
    df = pd.read_json(dataset_file, orient='records')
    return df['text'], df['target']


@click.command()
@click.pass_context
def train_multinomial_nb(ctx):
    features, target = read_dataset(ctx.obj['train_data'])
    pipeline = joblib.load(ctx.obj['features_pipeline'])

    nb = MultinomialNB()
    clf_pipeline = Pipeline([('features_construction', pipeline), ('cls', nb)])
    clf_pipeline.fit(features, target)
    joblib.dump(clf_pipeline, os.path.join(ctx.obj['model_output_file']))


@click.command()
@click.pass_context
@click.option('--max-iter', default=5, type=int)
def train_sgd(ctx, max_iter):
    features, target = read_dataset(ctx.obj['train_data'])
    pipeline = joblib.load(ctx.obj['features_pipeline'])

    sgd = SGDClassifier(loss='hinge',
                        penalty='l2',
                        alpha=1e-3,
                        random_state=42,
                        max_iter=max_iter,
                        tol=None)

    clf_pipeline = Pipeline([('features_construction', pipeline),
                             ('cls', sgd)])
    clf_pipeline.fit(features, target)
    joblib.dump(clf_pipeline, os.path.join(ctx.obj['model_output_file']))


@click.group()
@click.argument('train_data', type=click.Path(exists=True, dir_okay=False))
@click.argument('features_pipeline',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('model_output_file',
                type=click.Path(exists=False, dir_okay=False))
@click.pass_context
def entry_point(ctx, train_data, features_pipeline, model_output_file):
    ctx.obj['train_data'] = train_data
    ctx.obj['features_pipeline'] = features_pipeline
    ctx.obj['model_output_file'] = model_output_file


if __name__ == '__main__':
    entry_point.add_command(train_multinomial_nb)
    entry_point.add_command(train_sgd)
    entry_point(obj={})
