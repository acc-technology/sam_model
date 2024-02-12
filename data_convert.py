from ultralytics.data.converter import convert_coco
import yaml
# convert_coco(labels_dir='blind_data/annotations/', use_segments=True)


config = {
    'train': 'train/images',
    'val': 'valid/images',
    'names':{0:'walk',1:'T',2:'L',3:'car',4:'bike',5:'cover',6:'column'}
   
}

with open('blind_data_split/data.yaml', 'w') as f:
    yaml.dump(config, f)