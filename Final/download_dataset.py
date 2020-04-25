import tarfile
import zipfile
from tqdm import tqdm
import requests
# also requires AudioConverter

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
    tf.close()

def unzip(filename):
    zf = zipfile.ZipFile(filename, 'r')
    zf.extractall()
    zf.close()

def main():

    url = ''
    filename = ''

    urldictionary = {
        1 : 'http://opihi.cs.uvic.ca/sound/genres.tar.gz',
        2 : 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',
        3 : 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',
        4 : 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',
        5 : 'https://os.unil.cloud.switch.ch/fma/fma_full.zip'
    }

    filechoice = int(input(
        'Choose dataset to download:\n' + 
        '1 => GTZAN dataset (1.2 GB)\n' +
        '2 => FMA small dataset (7.2 GB)\n' +
        '3 => FMA medium dataset (22 GB)\n' +
        '4 => FMA large dataset (93 GB)\n' +
        '5 => FMA full dataset (879 GB)\n'
    ))

    if(filechoice != 1):
        print('Only GTZAN supported as of now.')
        filechoice = 1

    try:
        url = urldictionary[filechoice]
        filename = url.split('/')[-1]
    except:
        print('Please enter a valid number.')
        return -1
    
    print(f'Downloading {filename} from {url}')
    download(url)

    print(f'Unpacking {filename}')
    if(filechoice==1):
        untar(filename)
    else:
        unzip(filename)


if __name__ == '__main__':
    main()    


