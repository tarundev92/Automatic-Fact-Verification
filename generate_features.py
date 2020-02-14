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


wiki_file_name_lst = []
wiki_file_access_count_lst = []
wiki_files_lst = []
NUM_OF_DOC = 10
NUM_OF_SENT = 3

def get_top_matching_docs(query, num_of_docs):

    mul_qry_model = ir_model_matrix.transpose().dot(helper.get_sentence_matrix(query, hash_size).transpose())
    removed_1d_data = mul_qry_model.toarray().squeeze()
    index_list = pd.Series(removed_1d_data).nlargest(num_of_docs).index
    doc_dict = context["doc_dict"][1]
    return [doc_dict[idx] for idx in index_list]


def get_best_sents_from_doc_ids(query, doc_ids, num_of_sents):
    docid_line_list = []
    texts = []
    for doc_id in doc_ids:
        sents = get_sentence_from_wiki(wiki_pages_id_index, wiki_pages_path, doc_id, -1)
        for line_num, sent in sents:
            docid_line_list.append([doc_id, line_num])
            texts.append(sent)

    chosen_sents = helper.closest_sentences(query, texts, hash_size, num_of_sents=num_of_sents)
    return [docid_line_list[i] for i, sent in chosen_sents.items()]


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


def get_evidence(dict):
    evidence = dict.get("evidence", [])
    if not evidence == []:
        evidence = [[helper.unicode_encode(evid_id), line_num] for evid_id, line_num in evidence]
        return evidence

    claim = [helper.regex_tokenizer_string(helper.unicode_encode(get_claim_from_dict(dict)))]

    top_matching_docs = get_top_matching_docs(claim, NUM_OF_DOC)

    add_num_sent = 0
    if get_label_from_dict(dict) == "NOT ENOUGH INFO":
        add_num_sent = 2

    return get_best_sents_from_doc_ids(claim, top_matching_docs, NUM_OF_SENT+add_num_sent)


def feature_gen_process(key, value, is_test_dataset):
    class_labels = []
    new_data_dict = {}
    claim = get_claim_from_dict(value)
    if not is_test_dataset:
        class_labels.append(get_label_from_dict(value))
    if is_test_dataset:
        value["evidence"] = []
    evidences = get_evidence(value)

    new_data_dict[key] = {"claim": claim, "evidence": evidences}
    sents = [get_sentence_from_wiki(wiki_pages_id_index, wiki_pages_path, doc_ev[0], doc_ev[1]) for doc_ev in evidences]
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
    process_pool_len = 8
    raw_temp_max_i = int(process_pool_len/2)
    temp_max_i = int(process_pool_len/2)
    temp_max_inc = int(process_pool_len * 1.8)
    process_pool_index = 0
    procs = []

    for key, value in tqdm(data_dict.items(), total=data_dict_len, unit="items",
                           desc='Generating features for dataset'):
        if process_pool_index < process_pool_len:
            process_pool_index += 1
            procs.append(pool.apply_async(feature_gen_process, (key, value, is_test_dataset,)))

        if len(procs) > 35:
            temp_max_i = temp_max_inc

        if process_pool_index == process_pool_len:
            process_pool_index = 0
            temp_i=0
            while temp_i < temp_max_i:
                temp_i += 1
                unigram_pairs, bigram_pairs, labels, temp_data_dict = procs.pop(0).get()
                class_labels = class_labels + labels
                new_data_dict.update(temp_data_dict)
                unigram_features.append(unigram_pairs)
                bigram_features.append(bigram_pairs)
        temp_max_i = raw_temp_max_i

    for proc in procs:
        unigram_pairs, bigram_pairs, labels, temp_data_dict = proc.get()
        class_labels = class_labels + labels
        new_data_dict.update(temp_data_dict)
        unigram_features.append(unigram_pairs)
        bigram_features.append(bigram_pairs)

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
    parser.add_argument('ir_model_path', type=str, help='location of wiki IR model file')
    parser.add_argument('wiki_pages_path', type=str, help='location of wiki pages')
    parser.add_argument('wiki_dict_path', type=str, help='location of wiki dictionary file')
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    print("Start Time: ", time.strftime('%H:%M%p %Z on %b %d, %Y'))
    hash_size = int(math.pow(2, 24))

    args = parsed_arguments()

    wiki_pages_path = args.wiki_pages_path


    print("-----Loading wiki pages id index-----")
    wiki_pages_id_index = helper.load_dict_json(args.wiki_dict_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print("-----Loading IR model-----")
    temp_time = time.time()
    ir_model_matrix, context = helper.load_sparse_csr(args.ir_model_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    print("-----Loading wiki pages-----")
    temp_time = time.time()
    load_wiki_pages_index(wiki_pages_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    print("-----processing train set-----")
    temp_time = time.time()
    raw_file = read_json_file(args.train_path)
    data_set_features, data_set_dict = generate_features(raw_file)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    helper.save_dict_pickle(data_set_features, "train_set_features.pickle")
    helper.save_dict_json(data_set_dict, "train_set_dict.json")

    print("-----processing dev set-----")
    temp_time = time.time()
    raw_file = read_json_file(args.dev_path)
    data_set_features, data_set_dict = generate_features(raw_file, True)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    helper.save_dict_pickle(data_set_features, "dev_set_features.pickle")
    helper.save_dict_json(data_set_dict, "dev_set_dict.json")

    print("-----processing test set-----")
    temp_time = time.time()
    raw_file = read_json_file(args.test_path)
    data_set_features, data_set_dict = generate_features(raw_file, True)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    helper.save_dict_pickle(data_set_features, "test_set_features.pickle")
    helper.save_dict_json(data_set_dict, "test_set_dict.json")

    print("-----Total time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


