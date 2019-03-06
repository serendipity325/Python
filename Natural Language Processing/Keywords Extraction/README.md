# Keywords Extraction

Select 3-7 keywords that are somewhat representative of what the content of the document is. 

1. TF-IDF 

    The TF-IDF method applies Term-Frequency-Inverse-Document-Frequency (TF-IDF) normalization to a sparse matrix of occurrence counts. 
    In other words, it is a method to extract words with low frequency in training data but high frequency in test data. 
    Text files that are not mostly associated with the test data are used as training data. 
    Stop-words and special characters are removed. Seven words with the highest frequency were extracted.
    
    Result with TF-IDF
    Extracted Keywords:
    fuel, oil, said, contract, diesel, defense, austerity


    *Frequency table*
                  Word  Local_Frequency  Universal_Frequency
    0             fuel         0.388453                  512
    1              oil         0.341772                 5937
    2             said         0.306548                41346
    3         contract         0.284539                  795
    4           diesel         0.200617                  110
    5          defense         0.169947                  975
    6        austerity         0.150463                   29
    7               lt         0.142269                 9196
    8          awarded         0.142269                  148
    9          barrels         0.137476                  682
    10        measures         0.137476                  604
    11          energy         0.132683                 1295
    12           sales         0.100309                 3422
    13            sale         0.100309                 2487
    14          agency         0.100309                  891
    15           shell         0.097113                  633
    16          gallon         0.097113                   84
    17           quake         0.094846                   40
    18      television         0.093088                  354
    19        kerosene         0.093088                   66
    20        ministry         0.091651                 1051
    21           mines         0.089384                  282
    22              50         0.088456                 4173
    23       announced         0.088456                 1174
    24          prices         0.087625                 3773
    25   international         0.087625                 2635
    26          barrel         0.087625                  564
    27        minister         0.086188                 1252
    28        bringing         0.084974                  250
    29        products         0.084430                 1946
    30             gas         0.082567                 1997
    31          medium         0.082163                  373
    32         foreign         0.079511                 2595
    33           heavy         0.077753                  810
    34         company         0.071438                 6446
    35         imports         0.050154                 2150
    36              51         0.050154                 1124
    37              57         0.050154                  730
    38             ban         0.050154                  279
    39      suspension         0.050154                  169
    40         linking         0.050154                   65
    41       declaring         0.050154                   43
    42        holidays         0.050154                   38
    43      indefinite         0.050154                   30
    44        conserve         0.050154                   29
    45            0600         0.050154                   20
    46         volcano         0.050154                   17
    47       paralyzed         0.050154                   17
    48              19         0.048557                 2999
    49         exports         0.048557                 2724
    50              75         0.048557                 1607
    51              90         0.048557                 1497
    52              55         0.048557                 1331
    53      registered         0.048557                  303
    54        refining         0.048557                  109
    55        aviation         0.048557                  109
    56          routes         0.048557                   49
    57        weekends         0.048557                   34
    58          posted         0.047423                 1063
    59           joint         0.047423                  538
    60         venture         0.047423                  355
    61           basin         0.047423                  106
    62         pumping         0.047423                   59
    63              14         0.046544                 4088
    64         product         0.046544                 1179
    65           crude         0.046544                 1049
    66          raises         0.046544                  360
    67        thursday         0.046544                  320
    68           owned         0.045825                  948
    69              12         0.045218                 4708
    70         january         0.045218                 2630
    71          tender         0.045218                 1051
    72           scale         0.045218                  465
    73         houston         0.045218                  438
    74         adopted         0.045218                  216
    75          repair         0.045218                  184
    76           ocean         0.045218                  152
    77          grades         0.045218                  141
    78         floated         0.045218                   34
    79        domestic         0.043813                 1467
    80        superior         0.043813                  247
    81          deputy         0.043813                  199
    82      earthquake         0.043813                   95
    83      production         0.043437                 3208
    84      increasing         0.043437                  476
    85              15         0.043094                 5415
    86          member         0.043094                  937
    87            gets         0.043094                  932
    88        possibly         0.043094                  614
    89       completed         0.043094                  605
    90         limited         0.042779                  938
    91        november         0.042779                  550
    92         raising         0.042779                  271
    93            june         0.042215                 1524
    94           month         0.041961                 2760
    95        recently         0.041961                 1205
    96         pacific         0.041961                  597
    97              30         0.041722                 4577
    98           offer         0.041722                 2866
    99       effective         0.041722                 1066
    100      available         0.041497                 3230
    101      statement         0.041497                 1974
    102          march         0.041284                 3200
    103          speed         0.041284                 1414
    104         fields         0.041284                  481
    105          ships         0.041284                  430
    106          coast         0.041284                  377
    107     government         0.040889                 6137
    108          lines         0.040363                20054
    109        program         0.040363                 4098
    110         raised         0.040202                  814
    111        earlier         0.040048                 2090
    112          force         0.039617                 1384
    113           main         0.039229                 1209
    114         killed         0.039229                 1012
    115       expected         0.039108                 2828
    116            000         0.038876                14877
    117           east         0.038765                 1354
    118         strong         0.038450                 1513
    119     department         0.038350                 3405
    120            cut         0.038350                 2277
    121          miles         0.037393                  820
    122          today         0.037024                 3845
    123          point         0.036623                 4085
    124        country         0.035719                 2144
    125           near         0.035669                 1309
    126           open         0.034701                 2140
    127           high         0.033798                 3761
    128           told         0.033499                 3637
    129           work         0.033090                 5007
    130            day         0.032603                 4167
    131         people         0.031911                10643


2. word2vec

    The model takes word2vec representations of words in a vector space. 
    I used Google News corpora which provided by Google which consist of 3 million word vectors. 
    Due to the large size of the data, it takes about 30 minutes to train the embedding model.
    The second step is to find the PageRank value of each word. In the model PageRank algorithm takes word2vec representations of words. 
    The cosine distance and similarity is used to calculate edge weights between nodes. 
    After PageRank values of words are found, words which have the highest PageRank values will be selected as keywords.
    
    
3. non-overlapping-unigram-bigram model

    The algorithm is as follows:

    1. Using CountVectorize from Scikit-Learn, erase special characters and stop-words, 
       then extract both unigrams and bigrams and put them in a table.
    2. Find the smallest frequency in the table. Then make a hash-table (dictionary) that represents 
       the frequency of each word from all unigrams and bigrams which have the smallest frequency.   
    3. Remove the used unigrams and bigrams from the table, and use the remaining table as a data. 
    4. Extract 10 words with the highest frequency from the hash-table.
    5. After extracting bigrams again from the remaining data, leave only bigrams containing 
       the selected 10 most frequent words and erase the remaining bigrams from the data. 
       The 10 most frequent words are also deleted from the remaining data.
    6. Extract unigrams from the remaining data. 
       If an extracted unigram is included in any bigrams which are still in the data, the unigram is removed.
    7. By sorting the frequency of the remaining words, words with the highest frequency are extracted as keywords.





