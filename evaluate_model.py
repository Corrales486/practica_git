import joblib
import pandas as pd
from sklearn.metrics import classification_report
import click


@click.command()
@click.argument('model_pipeline_file',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('data_file', type=click.Path(exists=True, dir_okay=False))
def evaluate(model_pipeline_file, data_file):
    pipeline = joblib.load(model_pipeline_file)
    data = pd.read_json(data_file, orient='records')
    target_hat = pipeline.predict(data['text'])
    print(classification_report(data['target'], target_hat))


if __name__ == '__main__':
    evaluate()