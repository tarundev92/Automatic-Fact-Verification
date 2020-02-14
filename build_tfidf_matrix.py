import sys
import helper
import os
import math
from collections import Counter
import scipy.sparse as sp
import numpy as np
import string
import random
import time
from tqdm import tqdm


def term_count(doc_id, wiki_dict, index_doc):
    processed_tokens = helper.process_tokens(wiki_dict[doc_id])
    counts = Counter([helper.hash(gram, hash_size) for gram in processed_tokens])
    row = list(counts.keys())
    col = [index_doc[doc_id]] * len(counts)
    data = list(counts.values())
    return row, col, data

def get_wiki_id_tokens_dict(file_batch):
    wiki_dict = {}
    process_len = len(file_batch)
    for batch in tqdm(file_batch, total=process_len, unit="files", desc = 'Reading wiki pages'):
        if os.path.isfile(batch):
            wiki_line_num = -1
            for line in open(batch).read().splitlines():
                wiki_line_num += 1
                line = helper.unicode_encode(line)
                data = line.split(" ")
                try:
                    doc_line_num = int(data[1])
                except ValueError:
                    continue
                if not wiki_pages_id_index.get(data[0]):
                    wiki_pages_id_index[data[0]] = {
                        "filename": batch.split("/")[-1],
                        "wiki_line_nums": [],
                        "doc_line_nums": []
                    }

                wiki_pages_id_index[data[0]]["doc_line_nums"].append(doc_line_num)
                wiki_pages_id_index[data[0]]["wiki_line_nums"].append(wiki_line_num)

                wiki_dict[data[0]] = wiki_dict.get(data[0], []) + [data[2:]]
    return wiki_dict



def get_inverted_index(file_batch):
    wiki_dict = get_wiki_id_tokens_dict(file_batch)

    doc_ids = list(wiki_dict.keys())
    index_doc = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    row, col, data = [], [], []

    process_len = len(doc_ids)

    for doc_id in tqdm(doc_ids, total=process_len, unit="documents", desc = 'Processing wiki pages'):
        b_row, b_col, b_data = term_count(doc_id, wiki_dict, index_doc)
        row.extend(b_row)
        col.extend(b_col)
        data.extend(b_data)


    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (index_doc, doc_ids)


def merge_all_counts(counts_path, basename):
    index_file_paths = helper.get_file_path_list(counts_path)
    matrix, context = helper.load_sparse_csr(index_file_paths[0])
    del index_file_paths[0]

    index_doc, doc_ids = context['doc_dict']
    process_len = len(index_file_paths)
    for file_path in tqdm(index_file_paths, total=process_len, unit="files", desc = 'Merging files'):
        print("-" * 20, "Merging: ", file_path.split("/")[-1])
        next_matrix, next_context = helper.load_sparse_csr(file_path)
        matrix = sp.hstack([matrix, next_matrix])
        context['doc_freqs'] += next_context['doc_freqs']

        next_index_doc, next_doc_ids = next_context['doc_dict']

        if set(doc_ids).intersection(next_doc_ids):
            raise RuntimeError('overlapping doc id in %s file' % file_path)

        for key in next_index_doc.keys():
            next_index_doc[key] += len(index_doc)

        index_doc = {**index_doc, **next_index_doc}
        doc_ids += next_doc_ids

    context['doc_dict'] = (index_doc, doc_ids)
    merged_matrix_filename = os.path.join(output_index_path, basename)
    matrix = matrix.tocsr()
    print("-" * 20, "saving to %s.npz" % basename)
    helper.save_sparse_csr(merged_matrix_filename, matrix, context)

    print("merged_matrix_filename:", merged_matrix_filename + ".npz")
    return matrix, context


def get_tfidf_matrix(count_matrix):
    doc_freq = get_doc_freqs(count_matrix)
    doc_count = count_matrix.shape[1]
    idfreq = np.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
    idfreq[idfreq < 0] = 0
    idfreq = sp.diags(idfreq, 0)
    term_freq = count_matrix.log1p()
    tf_idf = idfreq.dot(term_freq)
    return tf_idf


def get_doc_freqs(counts):
    binary = (counts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


if __name__ == '__main__':
    start_time = time.time()
    #Read command line arguments
    cmd_args = sys.argv

    # wiki-pages-text path
    wiki_pages_path = cmd_args[1]

    #output path to store index
    output_index_path = cmd_args[2]

    #get files in batches of 4
    batche_limit = 27
    file_batches = helper.get_file_batches_from_path(wiki_pages_path, batche_limit)

    wiki_pages_id_index = {}

    hash_size = int(math.pow(2, 24))

    var_char_list = list(string.ascii_lowercase) + list(map(str, range(1, 10)))
    temp_folder_name = random.sample(var_char_list, 20)
    temp_folder_name = "".join(temp_folder_name)

    temp_path = output_index_path + ("" if output_index_path[-1] == '/' else "/") + temp_folder_name


    for batch in file_batches:

        basename = [name.split("/")[-1].split("-")[1].split(".")[0] for name in batch]
        basename = "wiki-" + basename[0] + "-to-" + basename[-1]
        print("-" * 10, "processing:", basename)
        count_matrix, doc_dict = get_inverted_index(batch)

        freqs = get_doc_freqs(count_matrix)
        hash_size = int(math.pow(2, 24))

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        filename = os.path.join(temp_path, basename)

        context = {
            'doc_freqs': freqs,
            'hash_size': hash_size,
            'doc_dict': doc_dict
        }
        print("-" * 10, "saving to %s.npz" % filename)
        helper.save_sparse_csr(filename, count_matrix, context)
        print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


    json_filename = "wiki_pages_id_index.json"
    dict_path = os.path.join(output_index_path, json_filename)
    print("saving wiki id index as json file")
    helper.save_dict_json(wiki_pages_id_index, dict_path)


    basename = 'wiki_processed_count_matrix'

    count_matrix, context = merge_all_counts(temp_path, basename)

    matrix = get_tfidf_matrix(count_matrix)

    tfidf_filename = basename + "_tfidf"

    filename = os.path.join(output_index_path, tfidf_filename)

    print("-"*10,"saving to %s.npz" % filename)
    helper.save_sparse_csr(filename, matrix, context)
    print("Total time taken: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


