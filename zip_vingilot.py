
import os
import zipfile

def my_app():
    with zipfile.ZipFile('results/vis.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        zipdir('results/vis', zipf)

    
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file),path))



if __name__ == '__main__':
    my_app()

