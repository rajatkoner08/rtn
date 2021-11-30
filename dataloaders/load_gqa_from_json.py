import json
import pandas as pd
import numpy as np
import collections
from PIL import Image
import operator
import os
import pickle
from tqdm import tqdm
from config import PROJECT_PATH
DATA_DIR = os.path.join(PROJECT_PATH,'data','gqa')

def replace_joint_word(string, joint_wordlist):
    for joint_word in joint_wordlist:
        if joint_word in string:
            replaced_joint_word = joint_word.replace(" ","_")
            string = string.replace(joint_word,replaced_joint_word)
    return string

def create_dict(elements, till):
    count = 1
    element_dict = {}
    element_dict["__background__"]=0
    for elem in elements[:till]:
        element_dict[elem]=count
        count+=1
    return element_dict

def get_gqa_stat():
    #load train_scene graph
    if  not os.path.exists(os.path.join(DATA_DIR, 'gqa_stats.pkl')):
        print('parsing full stat on gqa')

        graph_data = json.load(open(os.path.join(DATA_DIR, "graph/train_sceneGraphs.json"), 'r'))
        classes = {}
        attributes = {}
        relations = {}
        for i, scene in tqdm(enumerate(list(graph_data.items()))):
            objects = scene[1]['objects']
            for key, obj in objects.items():
                sub = obj['name']
                if sub in classes:
                    classes[sub]+=1
                else:
                    classes[sub] = 1
                attribs = list(np.unique(obj['attributes']))  # avoid redundent attributes
                relation = obj['relations']
                for j, attr in enumerate(attribs):
                    if attr in attributes:
                        attributes[attr]+=1
                    else:
                        attributes[attr] = 1

                for rel in relation:
                    if rel['name'] in relations:
                        relations[rel['name']]+=1
                    else:
                        relations[rel['name']]=1

        classes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
        attributes = sorted(attributes.items(), key=operator.itemgetter(1), reverse=True)
        relations = sorted(relations.items(), key=operator.itemgetter(1), reverse=True)
        print('____________GQA Stat_______________')
        print('____________Classes_______________')
        print(classes)
        print('____________Attributes_______________')
        print(attributes)
        print('____________Relations_______________')
        print(relations)
        gqa_stat =[]
        gqa_stat.append(classes)
        gqa_stat.append(attributes)
        gqa_stat.append(relations)
        # save files after each file parseing is complete
        with open(os.path.join(DATA_DIR, 'gqa_stats.pkl'), 'wb') as f:
            pickle.dump(gqa_stat, f)
    else:
        print("GQA statistics already computed earlier")
        with open(os.path.join(DATA_DIR, 'gqa_stats.pkl'), 'rb') as pickle_file:
            gqa_stat = pickle.load(pickle_file)

    return gqa_stat

def parse_question_answer(all_questions, all_answers, classes, attributes, relations):
    # check if the file exisits
    if not os.path.exists(os.path.join(DATA_DIR, 'gqa_question_stats.pkl')): #
        print('parsing question wise related object and attributes')
        #create dict of imgaes and their question and answers
        joint_words = [j_w for j_w in list(classes)+list(attributes) if " " in j_w]
        replaced_classes = [cls_name.replace(" ","_") for cls_name in classes]
        replaced_attributes = [attr.replace(" ","_") for attr in attributes]

        question_classes = {}
        question_attributes = {}
        answer_classes = {}
        answer_attributes = {}
        for each_question in tqdm(all_questions):
            question = each_question.lower().replace('?','')
            question = replace_joint_word(question, joint_wordlist=joint_words)
            question = question.split()
            q_objects = list(set(question) & set(replaced_classes))
            q_attr = list(set(question) & set(replaced_attributes))

            #now count obj and attr bases on question
            for obj in q_objects:
                if obj in question_classes:
                    question_classes[obj]+=1
                else:
                    question_classes[obj] = 1

            for attr in q_attr:
                if attr in question_attributes:
                    question_attributes[attr] += 1
                else:
                    question_attributes[attr] = 1

        # now count obj and attr bases on answer
        if not all_answers==None:
            for answer in tqdm(all_answers):
                if answer not in ('yes', 'no'):
                    answer = replace_joint_word(answer, joint_wordlist=joint_words)
                    ans_objects = list(set([answer]) & set(replaced_classes))
                    ans_attr = list(set([answer]) & set(replaced_attributes))
                    for obj in ans_objects:
                        if obj in answer_classes:
                            answer_classes[obj] += 1
                        else:
                            answer_classes[obj] = 1

                    for attr in ans_attr:
                        if attr in answer_attributes:
                            answer_attributes[attr] += 1
                        else:
                            answer_attributes[attr] = 1


        #now sort dicts, then save it in a onne folder
        gqa_question_stat = []
        gqa_question_stat.append(sorted(question_classes.items(), key=operator.itemgetter(1), reverse=True))
        gqa_question_stat.append(sorted(question_attributes.items(), key=operator.itemgetter(1), reverse=True))
        gqa_question_stat.append(sorted(answer_classes.items(), key=operator.itemgetter(1), reverse=True))
        gqa_question_stat.append(sorted(answer_attributes.items(), key=operator.itemgetter(1), reverse=True))
        gqa_question_stat.append(joint_words)
        # save files after each file parseing is complete
        # with open(os.path.join(DATA_DIR, 'gqa_question_stats.pkl'), 'wb') as f:
        #     pickle.dump(gqa_question_stat, f)
    else:
        with open(os.path.join(DATA_DIR, 'gqa_question_stats.pkl'), 'rb') as pickle_file:
            gqa_question_stat = pickle.load(pickle_file)

    return gqa_question_stat

def check_question_coverage(all_questions, all_answers, cls_list, attr_list, all_classes, all_attributes, joint_words):
    # create dict of imgaes and their question and answers
    total_questions = len(all_questions)
    miss_q_obj = len(all_questions)
    miss_q_attr = len(all_questions)
    miss_a_obj = len(all_questions)
    miss_a_attr = len(all_questions)
    for each_question in tqdm(all_questions):
        question = each_question.lower().replace('?', '')
        question = replace_joint_word(question, joint_words).split()
        q_objects = list(set(question) & set(cls_list))
        q_attr = list(set(question) & set(attr_list))
        all_q_objects = list(set(question) & set(all_classes))
        all_q_attr = list(set(question) & set(all_attributes))
        if len(all_q_objects)>len(q_objects):
            miss_q_obj -=1
        if len(all_q_attr)>len(q_attr):
            miss_q_attr-=1

        # now count obj and attr bases on answer
    if not all_answers==None:
        for answer in tqdm(all_answers):
            answer = replace_joint_word(answer, joint_words)
            if answer not in ('yes', 'no'):
                all_ans_objects = list(set([answer]) & set(all_classes))
                all_ans_attr = list(set([answer]) & set(all_attributes))
                ans_objects = list(set([answer]) & set(cls_list))
                ans_attr = list(set([answer]) & set(attr_list))
                if len(all_ans_objects) > len(ans_objects):
                    miss_a_obj -= 1
                if len(all_ans_attr) > len(ans_attr):
                    miss_a_attr -= 1
    print("\t Missed questions for objects ",(miss_q_obj/total_questions))
    print("\t Missed questions for attributes ", (miss_q_attr / total_questions))
    print("\t Missed answers for objects ", (miss_a_obj / total_questions))
    print("\t Missed answer for attributes ", (miss_a_attr / total_questions))

def parse_sg(objects, ind_to_classes, ind_to_rels, ind_to_attr, valid_cls, valid_rels, valid_attr, im_width, im_height):
    sg_dict = {}
    classes = {}
    boxes = []
    rels = []
    attribtues = {}
    keys = list(objects.keys())

    local_rels = {}
    cls_count = 0
    for key, obj in objects.items():
        sub = obj['name']
        if sub in valid_cls:
            #sanity check
            x1 = obj['x']
            y1 = obj['y']
            w = obj['w']
            h = obj['h']
            if w > 1 and h > 1:  # for line box
                if x1 < im_width and y1 < im_height:  # for boxes wihich are greater than image
                    x2 = x1+w
                    y2 = y1+h
                    if x2>x1 and y2>y1: # for those boxes whos x1==x2 or y1==y2
                        boxes.append([x1, y1, im_width if x2>im_width else x2, im_height if y2>im_height else y2])  # if (x2,y2) is bigger than width and height

                        attribs = list(np.unique(obj['attributes'])) #avoid redundent attributes
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
                                if sub_idx in attribtues.keys():
                                    attribtues[sub_idx].append(ind_to_attr.index(attr))
                                else:
                                    attribtues[sub_idx] = [ind_to_attr.index(attr)]



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
                test =False
                #print(object_name,'Not in the class')

    sg_dict['classes']=list(classes.values())
    sg_dict['boxes'] = boxes
    sg_dict['rels'] = rels
    sg_dict['attributes']=attribtues

    return  sg_dict

if __name__ == '__main__':

    # Set these variable as per the experiments or data
    compute_stas = True
    take_from_question = True
    parse_save_sg_format = True

    #parse initial data from sg and question
    gqa_stats_sg = get_gqa_stat()

    all_classes = np.asarray(gqa_stats_sg[0])[:, 0]
    all_attributes = np.asarray(gqa_stats_sg[1])[:, 0]
    all_rels = np.asarray(gqa_stats_sg[2])[:, 0]

    file_list = ['train_balanced_questions.json','val_balanced_questions.json','test_balanced.txt','challenge_balanced.txt',]
    for file_name in file_list:
        if file_name.endswith('.txt'):
            file_contnt = np.genfromtxt(os.path.join(DATA_DIR,file_name),delimiter='\t',dtype=str)
        else:
            file_contnt = json.load(open(os.path.join(DATA_DIR, file_name), 'r'))
        questions = file_contnt[:,2] if file_name.endswith('.txt') else file_contnt['question']
        answers = None if file_name.endswith('.txt') else file_contnt['answer']
        gqa_stats_question = parse_question_answer(questions, answers, all_classes, all_attributes, all_rels)

        #gqa_stats_question = parse_question_answer(file_contnt, all_classes, all_attributes, all_rels)
        all_classes_from_q = np.asarray(gqa_stats_question[0])[:, 0]
        all_attributes_from_q = np.asarray(gqa_stats_question[1])[:, 0]

        # do experiments :
        if compute_stas:
            print("Computing statistics from %s set" % file_name)
            if take_from_question:
                print("\tTaking most frequent classes from question")
                stat_cls = all_classes_from_q
                stat_attr = all_attributes_from_q
            else:
                print("\tTaking most frequent classes from scene graph")
                stat_cls = all_classes
                stat_attr = all_attributes
            class_at = [800]  #,600,700,800,900
            attribute_at = [200]  #180, 200, 250, 300, 400
            for cls_at in class_at:
                for atr_at in attribute_at:
                    cls_list = [cls.replace(" ","_") for cls in stat_cls[:cls_at]]
                    attr_list = [attr.replace(" ","_") for attr in  stat_attr[:atr_at]]
                    print('\t\tFor classes till ',cls_at,' and  attributes till ', atr_at)
                    check_question_coverage(questions, answers, cls_list, attr_list, all_classes_from_q, all_attributes_from_q, np.asarray(gqa_stats_question[4]))

    if parse_save_sg_format:
        #now parse scene graph based the earlier computed class
        graph_path = os.path.join(DATA_DIR,'graph')

        #define numer of top class
        num_class = 800
        num_rels = 170
        num_attr = 200

        ind_to_class = ['__background__']
        ind_to_rels = ['__background__']
        ind_to_attr = ['__background__']

        if take_from_question:
            valid_cls = create_dict([cls.replace("_"," ") for cls in all_classes_from_q], num_class)
            valid_attr = create_dict([attr.replace("_"," ") for attr in all_attributes_from_q], num_attr)
        else:  #fixme these files are not needed
            valid_cls = json.load(open(os.path.join(DATA_DIR, 'ent_dict.json'), 'r'))['sym2idx']
            valid_attr = json.load(open(os.path.join(DATA_DIR, 'attr_dict.json'), 'r'))['sym2idx'] #fixme create func for count
            with open(os.path.join(DATA_DIR, 'ind_to_attr.pkl'), 'rb') as pickle_file:
                valid_attr = pickle.load(pickle_file)

        valid_rels = json.load(open(os.path.join(DATA_DIR, 'pred_dict_171.json'), 'r'))['sym2idx']

        for json_file in os.listdir(graph_path):
            file_name = os.fsdecode(json_file)
            print("parsing file :",file_name)
            graph_data = json.load(open(os.path.join(graph_path, file_name), 'r'))
            print('number of scene graphs: ', len(graph_data))

            # now the main parsing
            sg_dict_master = {}
            for i, scene in  tqdm(enumerate(list(graph_data.items()))):
                img_name = scene[0]
                if not img_name in ('713545'): #for image which have alternate height, width
                    im_width = scene[1]['width']
                    im_height = scene[1]['height']
                    image_unpadded = Image.open(os.path.join(DATA_DIR,"VG_100K",img_name+".jpg")).convert('RGB')
                    img_w, img_h = image_unpadded.size
                    if img_w == im_width and img_h == im_height:
                        sg_dict = parse_sg(scene[1]['objects'], ind_to_class, ind_to_rels, ind_to_attr, valid_cls, valid_rels, valid_attr, im_width, im_height)
                        if len(sg_dict['rels'])>0:
                            sg_dict['width']=im_width
                            sg_dict['height']=im_height
                            sg_dict_master[img_name] = sg_dict
                    else:
                        print("Image size mismatch for : % s in sg (% d,% d) and in img (% d,% d) [w,h] manner"%(img_name, im_width, im_height,img_w,img_h))

            #save files after each file parseing is complete
            with open(os.path.join(DATA_DIR, os.path.splitext(file_name)[0]+ '.pkl'), 'wb') as f:
                pickle.dump(sg_dict_master, f)
            print("Saved : ",os.path.join(DATA_DIR, os.path.splitext(file_name)[0]+ '.pkl'))

        print('Total classes {}, rels {}, attributes {}'.format(len(ind_to_class), len(ind_to_rels), len(ind_to_attr)))


        #now save all classes, rels, attributes
        with open(DATA_DIR + '/ind_to_class.pkl', 'wb') as f:
            pickle.dump(ind_to_class, f)
        with open(DATA_DIR + '/ind_to_rels.pkl', 'wb') as f:
                pickle.dump(ind_to_rels, f)
        with open(DATA_DIR + '/ind_to_attr.pkl', 'wb') as f:
            pickle.dump(ind_to_attr, f)

    #graph_id, graph_dict = list(graph_data.items())[2]
    #
    #
    # # df = get_graph_df(graph_dict, graph_id)
    # # print(df)
    # #df_to_text(df, output_dir)