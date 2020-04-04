import tarfile
from tqdm import tqdm
import requests

def download(url):
    buffer_size = 1024
    r = requests.get(url,stream=True)

    file_size = int(r.headers.get("Content-Length",0))
    filename = url.split('/')[-1]

    progress = tqdm(r.iter_content(buffer_size),f'Downloading {filename}',total=file_size,unit="B",unit_scale=True,unit_divisor=1024)
    with open(filename,'wb') as f:
        for data in progress:
            f.write(data)
            progress.update(len(data))

def untar(filename):
    tf = tarfile.open(filename)
    tf.extractall()

print('Beginning file download...')
url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
download(url)

print('Beginning file extraction...')
filename = 'genres.tar.gz'
untar(filename)
