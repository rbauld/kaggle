
def clean1(text):
    import re
    from string import punctuation
    from nltk.corpus import stopwords
    import pandas as pd
    
    if pd.isnull(text):
        return ''

    stops = set(stopwords.words("english"))
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()

    text = text.split()
    text = [w for w in text if not w in stops]
    text = ' '.join(text)

    # Return a list of words
    return(text)

def magic1(train_in, test_in):
    # https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
    import numpy as np
    import pandas as pd
    import timeit

    train_orig = train_in.copy()
    test_orig  = test_in.copy()
    
    df1 = train_orig[['question1']].copy()
    df2 = train_orig[['question2']].copy()
    df1_test = test_orig[['question1']].copy()
    df2_test = test_orig[['question2']].copy()
    
    df2.rename(columns = {'question2':'question1'},inplace=True)
    df2_test.rename(columns = {'question2':'question1'},inplace=True)
    
    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    #train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
    train_questions.drop_duplicates(subset = ['question1'],inplace=True)
    
    train_questions.reset_index(inplace=True,drop=True)
    
    questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
    train_cp = train_orig.copy()
    test_cp = test_orig.copy()
    train_cp.drop(['qid1','qid2'],axis=1,inplace=True)
    
    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id':'id'},inplace=True)
    comb = pd.concat([train_cp,test_cp])
    
    comb['q1_hash'] = comb['question1'].map(questions_dict)
    comb['q2_hash'] = comb['question2'].map(questions_dict)
    
    q1_vc = comb.q1_hash.value_counts().to_dict()
    q2_vc = comb.q2_hash.value_counts().to_dict()
    
    def try_apply_dict(x,dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0
    #map to frequency space
    comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    
    # Calculate derivative features
    
    comb['freq_mean'] = (comb['q1_freq']+comb['q2_freq'])/2
    comb['freq_cross'] = comb['q1_freq']*comb['q2_freq']
    comb['q1_freq_sq'] = comb['q1_freq']*comb['q1_freq']
    comb['q2_freq_sq'] = comb['q2_freq']*comb['q2_freq']
    
    ret_cols = ['id', 'q1_freq', 'q2_freq','freq_mean', 'freq_cross', 'q1_freq_sq', 'q2_freq_sq']
    
    train_comb = comb[comb['is_duplicate'] >= 0][ret_cols]
    test_comb = comb[comb['is_duplicate'] < 0][ret_cols]
    
    return (train_comb[ret_cols], test_comb[ret_cols])

def wordmatch1(train_in, test_in, qcolumns = ['question1', 'question2'], append = ''):
    train_df = train_in.copy()
    test_df = test_in.copy()
    
    from nltk.corpus import stopwords
    
    stops = set(stopwords.words("english"))
    
    def word_match_share(row):
        q1words = {}
        q2words = {}
        for word in str(row[qcolumns[0]]).lower().split():
            if word not in stops:
                q1words[word] = 1
        for word in str(row[qcolumns[1]]).lower().split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R
    
    train_df['wordmatch1'+append] = train_df.apply(word_match_share, axis=1)
    test_df['wordmatch1'+append] = test_df.apply(word_match_share, axis=1)
    
    return (train_df, test_df)

def ngram_stats1(train_in, test_in, qcolumns = ['question1', 'question2'], append=''):
    
    train_in = train_in.copy()
    test_in = test_in.copy()   
    
    # Character length
    train_in['q1_len'+append] = train_in.apply(lambda x: len(x[qcolumns[0]]), axis = 1)
    train_in['q2_len'+append] = train_in.apply(lambda x: len(x[qcolumns[1]]), axis = 1)
    train_in['len_diff'+append] = abs(train_in['q1_len'+append] - train_in['q2_len'+append])
    
    test_in['q1_len'+append] = test_in.apply(lambda x: len(x[qcolumns[0]]), axis = 1)
    test_in['q2_len'+append] = test_in.apply(lambda x: len(x[qcolumns[1]]), axis = 1)
    test_in['len_diff'+append] = abs(test_in['q1_len'+append] - test_in['q2_len'+append])
    
    
    # n-gram statistics
    from nltk import ngrams
    from collections import Counter
    import pandas as pd
    from nltk.metrics import distance
    
    def get_ngram_stats(row, n, qcolumns, char=False, append=''):
        
        if char==True:
            q1 = ''.join(row[qcolumns[0]].split())
            q2 = ''.join(row[qcolumns[1]].split())
        else:
            q1 = row[qcolumns[0]].split()
            q2 = row[qcolumns[1]].split()    
        
        q1_ngram_list = list(ngrams(q1, n))
        q2_ngram_list = list(ngrams(q2, n))
        
        q1_ngram_set = set(q1_ngram_list)
        q2_ngram_set = set(q2_ngram_list)
        
        q1_sum = len(q1_ngram_list)
        q2_sum = len(q2_ngram_list)
        
        diff = abs(q1_sum - q2_sum)
        
        if q1_sum+q2_sum!=0:
            diff_norm = diff/(q1_sum+q2_sum)*2
        else:
            diff_norm = -1
        maximum = max([q1_sum, q2_sum])
        minimum = min([q1_sum, q2_sum])
        
        q1_unique = len(q1_ngram_set)
        q2_unique = len(q2_ngram_set)
        
        diff_unique = abs(q1_unique-q2_unique)
        
        intersect_r = Counter(q1_ngram_list) & Counter(q2_ngram_list)
        
        if q1_sum+q2_sum!=0:
            intersect_r = sum(intersect_r.values())/(q1_sum+q2_sum)*2
            intersect_unique_r = len(q1_ngram_set.intersection(q2_ngram_set))/(q1_unique+q2_unique)*2
        else:
            intersect_r = -1
            intersect_unique_r = -1
        
        if 0!=len(q1_ngram_set.union(q2_ngram_set)):
            jaccard_dist = (len(q1_ngram_set.union(q2_ngram_set))-len(q1_ngram_set.intersection(q2_ngram_set)))/len(q1_ngram_set.union(q2_ngram_set))
        else:
            jaccard_dist = 1
        
        bin_dist = distance.binary_distance(q1_ngram_set, q2_ngram_set)
        masi_dist = distance.masi_distance(q1_ngram_set, q2_ngram_set)
        
        listout = [q1_sum  , q2_sum  , diff  ,diff_norm   ,maximum, minimum, 
                   q1_unique, q2_unique, diff_unique, intersect_r, 
                   intersect_unique_r, jaccard_dist, bin_dist, masi_dist]
        
        keys    = ['q1_sum', 'q2_sum', 'diff', 'diff_norm', 'max', 'min', 
                   'q1_uni', 'q2_uni', 'diff_uni','intersect_r', 'inter_uni_r',
                   'jaccard_dist', 'bin_dist', 'masi_dist']
        keys = [x+str(n)+append for x in keys]
        dictout = dict(zip(keys, listout))
    
        return pd.Series(dictout)
    
    for n in range(1,4):
        print(n)
        ngram_stats = train_in.apply(lambda x:get_ngram_stats(x, n=n, qcolumns = qcolumns, char=False, append=append), axis=1)
        train_in = train_in.combine_first(ngram_stats)
        
    
    for n in range(1,4):
        print(n)
        ngram_stats = test_in.apply(lambda x:get_ngram_stats(x, n=n, qcolumns = qcolumns, char=False, append=append), axis=1)
        test_in = test_in.combine_first(ngram_stats)
        
    return (train_in, test_in)

def ngram_stats2(train_in, test_in, qcolumns = ['question1', 'question2'], append='', char=False):
    
    train_in = train_in.copy().loc[:,qcolumns]
    test_in = test_in.copy().loc[:,qcolumns]  
    
    # Character length
    train_in['q1_len'+append] = train_in.apply(lambda x: len(x[qcolumns[0]]), axis = 1)
    train_in['q2_len'+append] = train_in.apply(lambda x: len(x[qcolumns[1]]), axis = 1)
    train_in['len_diff'+append] = abs(train_in['q1_len'+append] - train_in['q2_len'+append])
    
    test_in['q1_len'+append] = test_in.apply(lambda x: len(x[qcolumns[0]]), axis = 1)
    test_in['q2_len'+append] = test_in.apply(lambda x: len(x[qcolumns[1]]), axis = 1)
    test_in['len_diff'+append] = abs(test_in['q1_len'+append] - test_in['q2_len'+append])
    
    
    # n-gram statistics
    from nltk import ngrams
    from collections import Counter
    import pandas as pd
    from nltk.metrics import distance
    import numpy as np
    
    def get_ngram_stats(row, n, qcolumns, char=False):
        
        if char==True:
            q1 = ''.join(row[qcolumns[0]].split())
            q2 = ''.join(row[qcolumns[1]].split())
        else:
            q1 = row[qcolumns[0]].split()
            q2 = row[qcolumns[1]].split()    
        
        q1_ngram_list = list(ngrams(q1, n))
        q2_ngram_list = list(ngrams(q2, n))
        
        q1_ngram_set = set(q1_ngram_list)
        q2_ngram_set = set(q2_ngram_list)
        
        q1_sum = len(q1_ngram_list)
        q2_sum = len(q2_ngram_list)
        
        diff = abs(q1_sum - q2_sum)
        
        if q1_sum+q2_sum!=0:
            diff_norm = diff/(q1_sum+q2_sum)*2
        else:
            diff_norm = -1
        maximum = max([q1_sum, q2_sum])
        minimum = min([q1_sum, q2_sum])
        
        q1_unique = len(q1_ngram_set)
        q2_unique = len(q2_ngram_set)
        
        diff_unique = abs(q1_unique-q2_unique)
        
        intersect_r = Counter(q1_ngram_list) & Counter(q2_ngram_list)
        
        if q1_sum+q2_sum!=0:
            intersect_r = sum(intersect_r.values())/(q1_sum+q2_sum)*2
            intersect_unique_r = len(q1_ngram_set.intersection(q2_ngram_set))/(q1_unique+q2_unique)*2
            masi_dist = distance.masi_distance(q1_ngram_set, q2_ngram_set)
        else:
            intersect_r = -1
            intersect_unique_r = -1
            masi_dist = -1
        
        if 0!=len(q1_ngram_set.union(q2_ngram_set)):
            jaccard_dist = (len(q1_ngram_set.union(q2_ngram_set))-len(q1_ngram_set.intersection(q2_ngram_set)))/len(q1_ngram_set.union(q2_ngram_set))
        else:
            jaccard_dist = 1
        
        bin_dist = distance.binary_distance(q1_ngram_set, q2_ngram_set)

        
        listout = [q1_sum  , q2_sum  , diff  ,diff_norm   ,maximum, minimum, 
                   q1_unique, q2_unique, diff_unique, intersect_r, 
                   intersect_unique_r, jaccard_dist, bin_dist, masi_dist]
    
        return listout
    
    keys    = ['q1_sum', 'q2_sum', 'diff', 'diff_norm', 'max', 'min', 
           'q1_uni', 'q2_uni', 'diff_uni','intersect_r', 'inter_uni_r',
           'jaccard_dist', 'bin_dist', 'masi_dist']    
    
    for n in range(1,4):
        print(n)
        ngram_stats = train_in.apply(lambda x:get_ngram_stats(x, n=n, qcolumns = qcolumns, char=char), axis=1)
        ngram_stats = np.vstack(ngram_stats.values)
        keys_tmp = [x+str(n)+append for x in keys]
        ngram_stats = pd.DataFrame(ngram_stats, columns = keys_tmp, index=train_in.index)
        train_in = train_in.combine_first(ngram_stats)
        
    
    for n in range(1,4):
        print(n)
        ngram_stats = test_in.apply(lambda x:get_ngram_stats(x, n=n, qcolumns = qcolumns, char=char), axis=1)
        ngram_stats = np.vstack(ngram_stats.values)
        keys_tmp = [x+str(n)+append for x in keys]
        ngram_stats = pd.DataFrame(ngram_stats, columns = keys_tmp, index=test_in.index)
        test_in = test_in.combine_first(ngram_stats)
        
    return (train_in, test_in)

def edit_distance(train_in, test_in, qcolumns = ['question1', 'question2'], append=''):

    train = train_in.copy().loc[:,qcolumns]
    test = test_in.copy().loc[:,qcolumns]
    
    import editdistance
    
    def my_fun(row, qcolumns):
        return editdistance.eval(row[qcolumns[0]], row[qcolumns[1]])
    
    key = 'edit_dist'+append
    train[key] = train.apply(lambda x: my_fun(x, qcolumns=qcolumns), axis=1)
    test[key]  = test.apply(lambda x: my_fun(x, qcolumns=qcolumns), axis=1)
    
    return (train, test)
    
def fuzzy_feats(train_in, test_in, qcolumns = ['question1', 'question2'], append=''):
    from fuzzywuzzy import fuzz
    import pandas as pd
    
    train = train_in.copy().loc[:,qcolumns]
    test = test_in.copy().loc[:,qcolumns]
    
    train['fuzz_r'+append] = train.apply(lambda x: fuzz.ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)
    train['fuzz_pr'+append] = train.apply(lambda x: fuzz.partial_ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)
    train['fuzz_tsr'+append] = train.apply(lambda x: fuzz.partial_token_set_ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)
    train['fuzz_tsor'+append] = train.apply(lambda x: fuzz.partial_token_sort_ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)    
    
    test['fuzz_r'+append] = test.apply(lambda x: fuzz.ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)
    test['fuzz_pr'+append] = test.apply(lambda x: fuzz.partial_ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)
    test['fuzz_tsr'+append] = test.apply(lambda x: fuzz.partial_token_set_ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)
    test['fuzz_tsor'+append] = test.apply(lambda x: fuzz.partial_token_sort_ratio(x[qcolumns[0]],x[qcolumns[1]]), axis = 1)     
    
    return (train, test)    
