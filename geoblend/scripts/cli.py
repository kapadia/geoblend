
import click
import geoblend

    
@click.command('geoblend')
@click.argument('src_path', type=click.Path(exists=True))
@click.argument('tar_path', type=click.Path(exists=True))
@click.argument('dst_path', type=click.Path(exists=False))
def geoblend(src_path, tar_path, dst_path):
    from geoblend.blend import blend
    blend(src_path, tar_path, dst_path)