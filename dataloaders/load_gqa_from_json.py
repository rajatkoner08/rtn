import json
import pandas as pd
import numpy as np
import collections
import os
import pickle
from tqdm import tqdm
from config import PROJECT_PATH
ROOT_DIR = '/home/koner/projects/VQA' #Path(os.path.abspath(__file__)).parent.parent.parent # This is your Project Root
DATA_DIR = os.path.join(PROJECT_PATH,'data','gqa')

def create_triple_df(df, columns):
    """
    create triples for dataframe from (s, p, o) defined in columns
    df: pandas.DataFrame
    columns(list[str]): [subject, relation, object] columns names
    """
    return ['-'.join(x) for x in df[columns].values.tolist()]


def get_graph_df(graph_dict, graph_id):
    # object_df
    obj = pd.DataFrame.from_dict(graph_dict['objects'], orient='index')

    for img_attr in ['width', 'height', 'location', 'weather']:
        obj[img_attr] = graph_dict.get(img_attr, np.nan)
    obj['imageId'] = graph_id
    obj['srcObjectId'] = obj.index
    node_labels = obj['name'].to_dict()
    obj.rename(columns={'name': 'srcObjectName'}, inplace=True)
    obj['outDegree'] = obj.relations.apply(lambda x: len(set([i['object'] for i in x])))
    obj['outDegreeDup'] = obj.relations.apply(lambda x: len(x))

    # triple df
    obj_rel = obj.explode('relations')
    obj_rel = obj_rel.drop('attributes', axis=1)
    obj_rel = obj_rel.dropna(subset=['relations'])
    if obj_rel.shape[0] == 0:
        obj_rel['dstObjectId'] = np.nan
        obj_rel['dstObjectName'] = np.nan
        obj_rel['relationName'] = np.nan
        obj_rel['triple'] = np.nan
    else:
        obj_rel = pd.concat([obj_rel.drop('relations', axis=1),
                             pd.DataFrame(
                                 obj_rel.relations.tolist(),
                                 index=obj_rel.index).rename({'object': 'dstObjectId',
                                                              'name': 'relationName'},
                                                             axis=1)
                             ], axis=1)
        obj_rel['dstObjectName'] = [node_labels[x] for x in obj_rel.dstObjectId.values]
        obj_rel['triple'] = create_triple_df(obj_rel, ['srcObjectName', 'relationName', 'dstObjectName'])

    assert obj.outDegreeDup.sum() == obj_rel.shape[0]  # double check #edges for develop

    return obj_rel


def df_to_text(df, output_dir='haha/'):
    """
    df -> [['a', 'p t', 'b']] -> [['a\tpT\tb']]
    """
    kg = df[['srcObjectId', 'srcObjectName', 'relationName', 'dstObjectId', 'dstObjectName']].values.tolist()
    new = []
    for spo in kg:
        spo = [x.replace(' ', '_') for x in spo]
        new.append('\t'.join(spo))
    print('Length of graph triples:', len(kg))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = output_dir + '/graph.txt'
    with open(filename, 'w') as f:
        for line in new:
            f.write(line + '\n')
    print('Graph saved to:', filename)

def parse_sg(objects, ind_to_classes, ind_to_rels, ind_to_attr, valid_cls, valid_rels, valid_attr, im_width, im_height):
    sg_dict = {}
    classes = {}
    boxes = []
    rels = []
    attribtues = []
    keys = list(objects.keys())

    local_rels = {}
    cls_count = 0
    for key, obj in objects.items():
        sub = obj['name']
        if sub in valid_cls:
            #sanity check
            if obj['w'] >= 1 or obj['h'] >= 1:  # for line box
                if obj['x'] <= im_width and obj['y'] <= im_height:  # for boxes wihich are greater than image
                    x2 = obj['x'] + obj['w']
                    y2 = obj['y'] + obj['h']
                    if x2>obj['x'] and y2>obj['y']: # for those boxes whos x1==x2 or y1==y2
                        boxes.append([obj['x'], obj['y'], im_width if x2>im_width else x2, im_height if y2>im_height else y2])  # if (x2,y2) is bigger than width and height

                        attribs = obj['attributes']
                        local_rels[key] = obj['relations']

                        if sub not in ind_to_classes: #f not in class list put it
                            ind_to_classes.append(sub)
                        sub_idx = cls_count
                        cls_count += 1
                        classes[key] =[sub_idx, ind_to_classes.index(sub)]

                        for j, attr in enumerate(attribs):
                            if attr in valid_attr:
                                if attr not in ind_to_attr:
                                    ind_to_attr.append(attr)
                                attribtues.append([sub_idx, ind_to_attr.index(attr)])



    for sub_key, relations in local_rels.items():
        for relation in relations:
            rel_name = relation['name']
            object_name = objects[relation['object']]['name']
            sub_idx = classes[sub_key][0]
            if object_name in valid_cls:
                if rel_name in valid_rels:
                    if relation['object'] in classes.keys():  # for those object which are not in bbox discripency
                        if rel_name not in ind_to_rels:
                            ind_to_rels.append(rel_name)


                        rel_idx = ind_to_rels.index(rel_name)
                        obj_idx = classes[relation['object']][0]

                        #finally put the relation
                        rels.append([sub_idx, obj_idx, rel_idx])
            else:
                print(object_name,'Not in the class')

    sg_dict['classes']=list(classes.values())
    sg_dict['boxes'] = boxes
    sg_dict['rels'] = rels
    sg_dict['attributes']=attribtues

    return  sg_dict

if __name__ == '__main__':
    graph_path = os.path.join(DATA_DIR,'graph')
    #define numer of top class
    num_class = 800
    num_rels = 170

    ind_to_class = ['__background__']
    ind_to_rels = ['__background__']
    ind_to_attr = ['__background__']

    valid_cls = json.load(open(os.path.join(DATA_DIR, 'ent_dict.json'), 'r'))['sym2idx']
    valid_rels = json.load(open(os.path.join(DATA_DIR, 'pred_dict_171.json'), 'r'))['sym2idx']
    valid_attr = json.load(open(os.path.join(DATA_DIR, 'attr_dict.json'), 'r'))['sym2idx']


    for json_file in os.listdir(graph_path):
        file_name = os.fsdecode(json_file)
        print("parsing file :",file_name)
        graph_data = json.load(open(os.path.join(graph_path, file_name), 'r'))
        print('number of scene graphs: ', len(graph_data))

        # if file_name == 'train_sceneGraphs.json':
        #     class_list = []
        #     rel_list = []
        #     for i, scene in tqdm(enumerate(list(graph_data.items()))):
        #         img_name = scene[0]
        #         im_width = scene[1]['width']
        #         im_height = scene[1]['height']
        #         sg_dict = parse_sg(scene[1]['objects'], ind_to_class, ind_to_rels, ind_to_attr, valid_cls, valid_rels, valid_attr)
        #         class_in_sg = np.asarray(sg_dict['classes'])
        #         rels_in_sg = np.asarray(sg_dict['rels'])[:,2]
        #         attr_in_sg = np.asarray(sg_dict['attributes'])[:,1]

        # now the main parsing
        sg_dict_master = {}
        for i, scene in  tqdm(enumerate(list(graph_data.items()))):
            img_name = scene[0]
            im_width = scene[1]['width']
            im_height = scene[1]['height']
            sg_dict = parse_sg(scene[1]['objects'], ind_to_class, ind_to_rels, ind_to_attr, valid_cls, valid_rels, valid_attr, im_width, im_height)
            if len(sg_dict['rels'])>0:
                sg_dict_master[img_name] = sg_dict

        #save files after each file parseing is complete
        with open(os.path.join(DATA_DIR, os.path.splitext(file_name)[0]+ '.pkl'), 'wb') as f:
            pickle.dump(sg_dict_master, f)

    print('Total classes {}, rels {}, attributes {}'.format(len(ind_to_class), len(ind_to_rels), len(ind_to_attr)))


    #now save all classes, rels, attributes
    with open(DATA_DIR + '/ind_to_class .pkl', 'wb') as f:
        pickle.dump(ind_to_class, f)
    with open(DATA_DIR + '/ind_to_rels.pkl', 'wb') as f:
            pickle.dump(ind_to_rels, f)
    with open(DATA_DIR + '/ind_to_attr.pkl', 'wb') as f:
        pickle.dump(ind_to_attr, f)

    #graph_id, graph_dict = list(graph_data.items())[2]


    # df = get_graph_df(graph_dict, graph_id)
    # print(df)
    #df_to_text(df, output_dir)