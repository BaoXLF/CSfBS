processdata class includes the data cleaning procedure. I remove the feature that includes a lot of Nan and get the feature I want to have. 
The getBinary(unique, id) is to get binary matrix, the input should be unique words and the products.
get_signature_matrix(binMat, numbPerm) is to get signiture matrix, the input of this should be binary matrix and number of permutation. the output of it is the signature matrix
lsh_cal(signatureMat, num_bands) is applying LSH take the signiture matrix as input and number of bands, return the candidate pairs
disMatrix(candidate, binaryMatrix, shop, brand, refresh, size) is to generate distance matrix for clustering
cluster_algorithm(threshold, dissimilarity_matrix) using the hierarchical algorithm to get the result
 getTruePairs (df) is for getting the true pairs, you need to input the data frame
 tuning_cluster(dissimilarity_matrix, truePairs, candidate_pairs) is to test the performance and also based on the performance to get the
 result 
