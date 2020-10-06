import click

@click.command()
@click.pass_context
def build_tfidf(ctx):
    pass

@click.command()
@click.pass_context
def build_lsa(ctx):
    pass

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

def build_tfidf(ctx):
    pass

def build_lsa(ctx):
    pass