# import packages

import pandas as pd
import numpy as np
from readJson import readJson
from processdata import processData
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering

# get binary matrix and sigmatrue matrix for LSH

def getBinary(unique, id):
    
    binary_matrix = pd.DataFrame(0, columns=unique, index=id.index)
    
    for unique_word in unique:
        binary_matrix[unique_word] = id.apply(lambda x: 1 if unique_word in x else 0)
    
    binary_matrix = binary_matrix.transpose().reset_index(drop=True)
    
    return binary_matrix

def get_signature_matrix(binMat, numbBands, numbRows):
    
    numbPerm = numbBands * numbRows
    numbWords, numbSig = binMat.shape
    sigMat = np.full((numbPerm, numbSig), np.inf)
    
    def shufllPermu(numbWords):
        sequence = np.arange(1, numbWords + 1)
        np.random.shuffle(sequence)
        return sequence
     
    for i in range(numbPerm): 
        
        signature = np.full((1, numbSig), np.inf)
        permuntation = shufllPermu(numbWords)
        # print(permuntation)
        for row in range(numbWords):
            nonzero_indices = np.where(binMat.iloc[row,:] == 1)[0]
            # print(nonzero_indices)
            if len(nonzero_indices) == 0:
                continue
            for col in nonzero_indices:
                #if len(nonzero_indices) == 0:
                    #break
                if signature[0, col] > permuntation[row]:
                    signature[0, col] = permuntation[row]
    
        sigMat[i, :] = signature

    return sigMat

  
# apply LSH
def lsh_cal(signatureMat, num_bands):

    signatureMat = pd.DataFrame(signatureMat)
    candidate = []
    # Use numpy.array_split to split the DataFrame into subgroups
    for q, subset in enumerate(np.array_split(signatureMat, num_bands, axis=0)):
        bucket = []
        for col in subset.columns:
                modified_col = [str(int(value)) for value in subset.iloc[:,col]]
                hashword = ''.join(modified_col)
                hashvalue = hash(hashword)
                # hashvalue1 = hashvalue % 7777
                bucket.append(hashvalue)

        for i in range(len(bucket)-1):
            for j in  range(i+1,len(bucket)):
                if bucket[i] == bucket[j] :
                    candidate.append((i,j))
                    
    candidate_list = list(set(candidate)) # here i should divide the length by 2
    return candidate_list

def cosine_distance(productA, productB):
    
    cosin = np.dot(productA, productB)/(np.linalg.norm(productA)*np.linalg.norm(productB))
    return 1 - cosin
    
# get dissimilarity matrix
def disMatrix(candidate, binaryMatrix, brand, refresh, size):
    
    numbWord, numbSig = binaryMatrix.shape
    dissim_matrix = np.ones((numbSig, numbSig))
    
    for row in candidate:
        
        if brand[row[0]] == brand[row[1]] or (pd.isna(brand[row[0]]) and pd.isna(brand[row[1]])):
            if refresh[row[0]] == refresh[row[1]] or (pd.isna(refresh[row[1]]) or pd.isna(refresh[row[1]])):
                if size[row[0]] == size[row[1]] or (pd.isna(size[row[0]]) or pd.isna(size[row[1]])):
                
                    dist = cosine_distance(np.array(binaryMatrix.iloc[:, row[0]]), np.array(binaryMatrix.iloc[:, row[1]]))
                        
                    dissim_matrix[row[0],row[1]] = dist
                    dissim_matrix[row[1],row[0]] = dist
        
    return dissim_matrix


# clustering 

def cluster_algorithm(threshold, dissimilarity_matrix):
    # Hierarchical clustering
    cluster = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=threshold, compute_full_tree=True)
    clusters = cluster.fit_predict(dissimilarity_matrix)

    pairs = []

    for i in range(cluster.n_clusters_):
        products_cluster = np.where(clusters == i)[0]

        if len(products_cluster) > 1:
            pairs += list(combinations(products_cluster, 2))

    return pairs

def getTruePairs (df):
    unique_values = df['modelID'].unique()
    index_dict = {value: df.index[df['modelID'] == value].tolist() for value in unique_values}
    duplicate_list = []
    for key,value in index_dict.items():
        if len(value) >= 2:
            for i in range(len(value)-1):
                for j in range(i+1,len(value)):
                    duplicate_list.append((value[i],value[j]))
            
    return duplicate_list

def F1(precision, recall):
    f1 = (2*precision*recall)/(precision+recall)
    return f1
        
def tuning_cluster(dissimilarity_matrix, truePairs, candidate_pairs, tuning_parameters):
    
    ntruePairs = len(truePairs)
    
    savelist = {}
    f1_scores = []
    for para in tuning_parameters:
        pairs = cluster_algorithm(para, dissimilarity_matrix)
        # Calculate TP, TN, FP, FN
        set_candidate_pairs = set(map(tuple, pairs))
        set_true_duplicates = set(map(tuple, truePairs))
        npair = len(pairs)
        if npair == 0:
            continue
        all_pairs = set(combinations(range(dissimilarity_matrix.shape[0]), 2))
        non_duplicate_pairs = all_pairs - set_true_duplicates
        TP = len(set_candidate_pairs.intersection(set_true_duplicates))
        FP = len(set_candidate_pairs - set_true_duplicates)
        FN = len(set_true_duplicates - set_candidate_pairs)
        # TN = len(non_duplicate_pairs) - FP
        
        precision_PQ = TP/(TP+FP)
        recall_PC = TP/(TP + FN)
        F1_value = F1(precision_PQ, recall_PC)

        fraction_comp = len(candidate_pairs) / len(all_pairs)
        savelist[F1_value] =  [precision_PQ, recall_PC, fraction_comp, para, npair]
        f1_scores.append(F1_value)
        
    f1_score = max(f1_scores)
    rest = savelist[f1_score]
    
    return {'pair_quality(precision)': rest[0], 'pair_completeness(recall)': rest[1], 'F1': f1_score, 'fraction_comp' : rest[2], 'threshold': rest[3], 'numbPair': npair}
    
# clean data and get classifiction conditions.
patterns = {'newegg.com': ' ',
            'thenerds.net': ' ',
            ' - best buy': ' ',
            ' - thenerds.net':' ',
            '-inch': 'inch ',
            '"':'inch ',
            '\'':'inch ',
            'inches':'inch ',
            ' inch':'inch ',
            '\bin\.\b': 'inch ',
            'hertz': 'hz ',
            ' hz':'hz ',
            '-hz':'hz ',
            'ledlcd': 'led lcd ',
            'led-lcd':'led lcd ',
            'diag.':' ',
            '/':' ',
            '\(':' ',
            '\) ':' ',
            '-': ' ',
            '/]':' '
}

brands =  ["philips", "supersonic", "sharp", "samsung", 
           "toshiba", "hisense", "sony", "lg",  "sanyo",
            "coby", "panasonic", "rca", "vizio", "naxa",
            "sansui", "viewsonic", "avue", "insignia",
            "sunbritetv", "magnavox", "jvc", "haier", 
            "optoma", "nec", "proscan", "venturer", 
            "westinghouse", "pyle", "dynex", "magnavox", 
            "sceptre", "tcl", "mitsubishi", 
            "curtisyoung", "compaq", "hannspree", 
            "upstar", "azend", "seiki", "craig",
            "contex", "affinity", "hiteker", "epson", 
            "elo", "pyle", "gpx", "sigmac", 
            "venturer", "elite"]


# read data and also generate Booststrap samples
numbBoot = 5
ratio = 0.63
tOfFeature = 0.5
file_name = 'TVs-all-merged.json'
tuning_parameters = [0.1, 0.3, 0.5, 0.75, 0.9]
b = 20
r = 5

# run an example to test the code
def example_code(file_name, numbBoot, ratio, tuning_parameters, b, r, patterns, brands, tOfFeature):
    readData = readJson(file_name)
    Booststrap, test = readData.getBootstrapSamples(numbBoot, ratio)
    result_list = []
    for i in range(len(Booststrap)):
        
        clean = processData(Booststrap[i], patterns, brands, tOfFeature)
        cleanedDat, uniqueWords, id = clean.preProcessData()
        brand = clean.getBrand(cleanedDat)
        size = clean.getScreenSize(cleanedDat)
        refresh = clean.getRefresh(cleanedDat)
        
        binMat = getBinary(uniqueWords, id)
        signMat = get_signature_matrix(binMat, b, r)
        candidate_pairs = lsh_cal(signMat, b)
        dissimilarity_matrix = disMatrix(candidate_pairs, binMat, brand, refresh, size)
        truePairs = getTruePairs(cleanedDat)
        result = tuning_cluster(dissimilarity_matrix, truePairs, candidate_pairs, tuning_parameters)
        
        result_list.append(result)
    
    return result_list

print(example_code(file_name, numbBoot, ratio, tuning_parameters, b, r, patterns, brands,tOfFeature))