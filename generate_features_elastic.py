import argparse
import json
import helper
import pandas as pd
import math
import os
from tqdm import tqdm
from multiprocessing import Pool
from collections import OrderedDict
import time
from elasticsearch import Elasticsearch
from time import sleep

# NUM_OF_WIKI_FILES_MEM = 30
wiki_file_name_lst = []
wiki_file_access_count_lst = []
wiki_files_lst = []
NUM_OF_DOC = 10
NUM_OF_SENT = 5


def get_file_from_index(wiki_pages_path, wiki_filename):
    file_index = wiki_file_name_lst.index(wiki_filename)
    return wiki_files_lst[file_index]


def load_wiki_pages_index(wiki_pages_path):
    for dir_path, _, dir_file_names in os.walk(wiki_pages_path):
        for file_name in dir_file_names:
            if helper.is_wiki_file(file_name):
                loc = os.path.join(dir_path, file_name)
                raw_file = open(loc)
                wiki_file = raw_file.read().splitlines()
                raw_file.close()
                wiki_file_name_lst.append(file_name)
                wiki_files_lst.append(wiki_file)


def get_sentence_from_wiki(wiki_pages_id_index, wiki_pages_path, doc_id, doc_line_num):
    wiki_file_dict = wiki_pages_id_index[doc_id]
    wiki_filename = wiki_file_dict["filename"]
    wiki_file = get_file_from_index(wiki_pages_path, wiki_filename)

    if doc_line_num == -1:
        sentence = []
        for wiki_line_num in wiki_file_dict["wiki_line_nums"]:
            line = wiki_file[wiki_line_num]
            data = line.split(" ")
            sentence.append([int(data[1]), data[2:]])
    else:
        line_nums = wiki_file_dict["doc_line_nums"]
        doc_line_num_idx = line_nums.index(doc_line_num)
        wiki_line_num = wiki_file_dict["wiki_line_nums"][doc_line_num_idx]
        line = wiki_file[wiki_line_num]
        data = line.split(" ")
        sentence = data[2:]

    return sentence


def get_claim_from_dict(dict):
    return dict["claim"]

def get_label_from_dict(dict):
    return dict.get("label", None)


def get_evidence(key, dict):
    evidence = dict.get("evidence", [])
    if not evidence == []:
        evidence = [[helper.unicode_encode(evid_id), line_num] for evid_id, line_num in evidence]
        return evidence

    claim = helper.unicode_encode(get_claim_from_dict(dict))
    claim = claim.replace("-COLON-", ":").replace("(", "-LRB-").replace(")", "-RRB-")
    processed_claim = " ".join(helper.process_tokens([claim.split(" ")]))

    body = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "raw_doc_id": {
                            "query": claim,
                            "boost": 1.2
                        }
                    }

                },
                "should": [
                    {
                        "match": {
                            "line": claim
                        }
                    },
                    {
                        "match": {
                            "processed_line": {
                                "query": processed_claim,
                                "boost": 1.05
                            }
                        }

                    }
                ]
            }
        }
    }
    while_i = 3
    while while_i > 0:
        while_i -= 1
        try:
            top_best_sents = es.search(index="wiki-pages-index", body=body)
            break
        except:
            sleep(0.03)
            if while_i == 0:
                wrt_txt = str(key) + "||" + claim + "\n"
                file1 = open("failed_ids.txt", "a")
                file1.write(wrt_txt)
                file1.close()
                print("time-out:", key)
                return []

    for hit in top_best_sents['hits']['hits'][:NUM_OF_SENT]:
        evidence.append([hit["_source"]["doc_id"], hit["_source"]["line_no"]])

    return evidence


def feature_gen_process(key, value, is_test_dataset):
    class_labels = []
    new_data_dict = {}
    claim = get_claim_from_dict(value)
    if not is_test_dataset:
        class_labels.append(get_label_from_dict(value))
    if is_test_dataset:
        value["evidence"] = []
    evidences = get_evidence(key,value)

    if evidences == []:
        return [],[],[],{}

    new_data_dict[key] = {"claim": claim, "evidence": evidences[:3]}
    sents = [get_sentence_from_wiki(wiki_pages_id_index, wiki_pages_path, doc_ev[0], doc_ev[1]) for doc_ev in evidences]
    claim = claim.replace("-COLON-", ":").replace("(", "-LRB-").replace(")", "-RRB-")
    unigram_pairs = helper.unigram_word_pair(claim, sents)
    bigram_pairs = helper.bigram_word_pair(claim, sents)
    return unigram_pairs, bigram_pairs, class_labels, new_data_dict



def generate_features(data_dict, is_test_dataset=False):

    class_labels = []
    unigram_features = []
    bigram_features = []
    new_data_dict = OrderedDict()
    data_dict_len = len(data_dict.items())

    pool = Pool(processes=5)
    process_pool_len = 12 if is_test_dataset else 20
    raw_temp_max_i = int(process_pool_len/2)
    temp_max_i = int(process_pool_len/2)
    temp_max_inc = int(process_pool_len * 1.8)
    process_pool_index = 0
    procs = []

    procs_limit = 25 if is_test_dataset else 50

    for key, value in tqdm(data_dict.items(), total=data_dict_len, unit="items",
                           desc='Generating features for dataset'):
        if process_pool_index < process_pool_len:
            process_pool_index += 1
            procs.append(pool.apply_async(feature_gen_process, (key, value, is_test_dataset,)))
            sleep(0.01)

        if len(procs) > procs_limit:
            temp_max_i = temp_max_inc

        if process_pool_index == process_pool_len:
            process_pool_index = 0
            temp_i=0
            while temp_i < temp_max_i:
                temp_i += 1
                t_proc = procs.pop(0)
                unigram_pairs, bigram_pairs, labels, temp_data_dict = t_proc.get()
                if unigram_pairs == []:

                    continue
                class_labels = class_labels + labels
                new_data_dict.update(temp_data_dict)
                unigram_features.append(unigram_pairs)
                bigram_features.append(bigram_pairs)
        temp_max_i = raw_temp_max_i

    for proc in procs:
        unigram_pairs, bigram_pairs, labels, temp_data_dict = proc.get()
        if unigram_pairs == []:
            continue
        class_labels = class_labels + labels
        new_data_dict.update(temp_data_dict)
        unigram_features.append(unigram_pairs)
        bigram_features.append(bigram_pairs)


    try:
        raw_file = open("failed_ids.txt")
        file_lines = raw_file.read().splitlines()
        os.remove("failed_ids.txt")
        print("Retrying failed ids")
        for raw_line in file_lines:
            line = raw_line.split("||")
            key = line[0]
            temp_data = {"claim": line[1], "evidence": []}
            if not is_test_dataset:
                temp_data.update({"label": "NOT ENOUGH INFO"})
            unigram_pairs, bigram_pairs, labels, temp_data_dict = feature_gen_process(key, temp_data, is_test_dataset)
            if unigram_pairs == []:
                file_lines.append(raw_line)
                continue
            class_labels = class_labels + labels
            new_data_dict.update(temp_data_dict)
            unigram_features.append(unigram_pairs)
            bigram_features.append(bigram_pairs)
        os.remove("failed_ids.txt")
    except:
        pass


    feat_set_dict = {'X_unigram_features': unigram_features, 'X_bigram_features': bigram_features}
    if not is_test_dataset:
        feat_set_dict.update({'y': class_labels})

    return feat_set_dict, new_data_dict



def read_json_file(path):
    return json.loads(open(path).read())


def parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help='location of train dataset')
    parser.add_argument('dev_path', type=str, help='location of dev dataset')
    parser.add_argument('test_path', type=str, help='location of test dataset')
    parser.add_argument('wiki_pages_path', type=str, help='location of wiki pages')
    parser.add_argument('wiki_dict_path', type=str, help='location of wiki dictionary file')
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    print("Start Time: ", time.strftime('%H:%M%p %Z on %b %d, %Y'))
    hash_size = int(math.pow(2, 24))

    args = parsed_arguments()

    wiki_pages_path = args.wiki_pages_path

    es = Elasticsearch()
    es.indices.refresh(index="wiki-pages-index")
    if es.ping():
        print('Yay Connected')
    else:
        raise Exception("Elasticsearch error. Start the server and build index")


    print("-----Loading wiki pages id index-----")
    wiki_pages_id_index = helper.load_dict_json(args.wiki_dict_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


    print("-----Loading wiki pages-----")
    temp_time = time.time()
    load_wiki_pages_index(wiki_pages_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))


    print("-----processing train set-----")
    temp_time = time.time()
    try:
        os.remove("failed_ids.txt")
    except:
        pass
    raw_file = read_json_file(args.train_path)
    data_set_features, data_set_dict = generate_features(raw_file)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    # helper.save_dict_json(data_set_features, "train_set_features.json")
    helper.save_dict_pickle(data_set_features, "elastic_train_set_features.pickle")
    helper.save_dict_json(data_set_dict, "elastic_train_set_dict.json")

    print("-----processing dev set-----")
    temp_time = time.time()
    try:
        os.remove("failed_ids.txt")
    except:
        pass
    raw_file = read_json_file(args.dev_path)
    data_set_features, data_set_dict = generate_features(raw_file, True)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    helper.save_dict_pickle(data_set_features, "elastic_dev_set_features.pickle")
    helper.save_dict_json(data_set_dict, "elastic_dev_set_dict.json")


    print("-----processing test set-----")
    temp_time = time.time()
    try:
        os.remove("failed_ids.txt")
    except:
        pass
    raw_file = read_json_file(args.test_path)
    data_set_features, data_set_dict = generate_features(raw_file, True)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    helper.save_dict_pickle(data_set_features, "elastic_test_set_features.pickle")
    helper.save_dict_json(data_set_dict, "elastic_test_set_dict.json")

    print("-----Total time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


