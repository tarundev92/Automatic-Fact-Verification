from elasticsearch import Elasticsearch
from tqdm import tqdm
import argparse
import helper
import os
import time
from elasticsearch import helpers as elastic_helper
from multiprocessing import Pool


def wiki_clean_process(p_file, p_file_len, p_batch, es_id):
    p_wiki_pages_id_index = {}
    p_wiki_line_num = -1
    p_actions = []
    for line in tqdm(p_file, total=p_file_len, unit="lines", desc='Indexing ' + p_batch):
        p_wiki_line_num += 1
        line = helper.unicode_encode(line)
        data = line.split(" ")

        try:
            doc_line_num = int(data[1])
        except ValueError:
            continue

        doc_id = data[0]
        doc_line_tokens = data[2:]
        doc_line = " ".join(doc_line_tokens)


        if not p_wiki_pages_id_index.get(doc_id):
            p_wiki_pages_id_index[doc_id] = {
                "filename": p_batch.split("/")[-1],
                "wiki_line_nums": [],
                "doc_line_nums": []
            }

        es_id += 1

        raw_doc_id = procc_doc_id = doc_id.replace("_", " ").replace("-COLON-", ":")


        p_action = {
            "_index": "wiki-pages-index",
            "_type": "wiki-pages",
            "_id": es_id,
            "_source": {
                'doc_id': doc_id,
                'raw_doc_id': raw_doc_id,
                'processed_doc_id': procc_doc_id,
                'line': doc_line,
                'processed_line': " ".join(helper.process_tokens([doc_line_tokens])),
                'line_no': doc_line_num,
            }
        }
        p_wiki_pages_id_index[doc_id]["doc_line_nums"].append(doc_line_num)
        p_wiki_pages_id_index[doc_id]["wiki_line_nums"].append(p_wiki_line_num)


        p_actions.append(p_action)
    return p_actions, p_wiki_pages_id_index


def parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('wiki_pages_path', type=str, help='path to wiki pages directory')
    parser.add_argument('output_path', type=str, help='output path to save indexes')
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    print("Start Time: ", time.strftime('%H:%M%p %Z on %b %d, %Y'))

    args = parsed_arguments()
    wiki_pages_path = args.wiki_pages_path
    output_index_path = args.output_path

    if not os.path.exists(output_index_path):
        os.makedirs(output_index_path)

    batche_limit = 27
    file_batches = helper.get_file_batches_from_path(wiki_pages_path, batche_limit)

    es = Elasticsearch()
    if es.ping():
        print('Yay Connected')
    else:
        print('Awww it could not connect!')


    pool = Pool(processes=4)
    process_pool_len = 3
    raw_temp_max_i = int(process_pool_len / 2)
    temp_max_i = int(process_pool_len / 2)
    temp_max_inc = 2
    process_pool_index = 0
    procs = []


    es_id_start = 0
    wiki_pages_id_index = {}

    process_len = len(file_batches)
    for batch_files in tqdm(file_batches, total=process_len, unit="files", desc='Reading wiki pages'):
        actions = []
        for batch in batch_files:
            if os.path.isfile(batch):
                file = open(batch).read().splitlines()
                file_len = len(file)

                es_id_start += len(file)

                if process_pool_index < process_pool_len:
                    process_pool_index += 1
                    procs.append(pool.apply_async(wiki_clean_process, (file, file_len, batch, es_id_start,)))

                if len(procs) > 4:
                    temp_max_i = process_pool_len

                if process_pool_index == process_pool_len:
                    process_pool_index = 0
                    temp_i = 0
                    while temp_i < temp_max_i:
                        temp_i += 1
                        t_actions, t_wiki_pages_id_index = procs.pop(0).get()
                        actions += t_actions
                        wiki_pages_id_index.update(t_wiki_pages_id_index)
                temp_max_i = raw_temp_max_i

        for proc in procs:
            t_actions, t_wiki_pages_id_index = proc.get()
            actions += t_actions
            wiki_pages_id_index.update(t_wiki_pages_id_index)

        procs = []


        temp_time = time.time()
        print("bulk index batch")
        temp_success, temp_fail = elastic_helper.bulk(es, actions)
        print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))
    es.indices.refresh(index="wiki-pages-index")
    json_filename = "wiki_pages_id_index.json"
    dict_path = os.path.join(output_index_path, json_filename)
    temp_time = time.time()
    print("saving wiki id index as json file")
    helper.save_dict_json(wiki_pages_id_index, dict_path)
    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - temp_time)))

    print("-----Total time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
