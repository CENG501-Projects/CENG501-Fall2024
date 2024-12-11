import shutil
import glob
import os
def moveToAll_Data(location):
    groups=['train','test','val']
    for group in groups:
        record_directory=f'../Datasets/{location}/{group}/all_data'
        if not os.path.exists(record_directory):
            os.makedirs(record_directory)
        folders=glob.glob(f'../Datasets/{location}/{group}/*')
        
        
        folder_items=[]
        for k in folders:
            if(os.listdir(k)):
                if k.endswith('all_data'):
                    pass
                else:
                    folder_items.append([k+'/'+t for t in os.listdir(k)])
        for k in folder_items:
            for v in k:
                shutil.move(v,f'../Datasets/{location}/{group}\\all_data/'+v.split('/')[-1])