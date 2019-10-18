import pandas as pd
import spacy
from collections import defaultdict
import re
import datetime as dt
import gensim
import numpy as np
import pickle
import sys,os
import codecs
import random
from scipy import stats

from rake_nltk import Rake

r = Rake()


np.random.seed(8)
nb_test = 1000
nb_dev = 1000


def find_substring(subarray, array):
 
    indices = []
    index = 0  # Begin at -1 so index + 1 is 0
    while True:
        if index == len(array):
            break

        for i in range(index, len(array)):
            index = i+1
            if array[i:i + len(subarray)] == subarray:

                indices.append(i)
                break

    return indices

 


train_crp = [line.rstrip('\n') for line in open('mimicIII/final-test/test/'+sys.argv[1])]
#dev_crp = [line.rstrip('\n') for line in open('mimicIII/final-test/dev.keys')]
#test_crp = [line.rstrip('\n') for line in open('mimicIII/final-test/test.keys')]


gen_id_list = []

gen_id_list.extend(train_crp)
#print(gen_id_list)
#gen_id_list.extend(dev_crp)
#gen_id_list.extend(test_crp)

time_span_dict = {0: '0', 182.6: '0.5', 365.25: '1', 730.5: '2', 1095.75: '3', 1461: '4', 1826.25: '5'}

#heading_dict = defaultdict(int)

heading_dict = pickle.load(open("heading.pickle", "rb" ))



null_str = 'NULL'



keys_text = {}
keys_keys = {}
keys_meta = {}


keyword_lenghts =[]


in_file = 'mimicIII/NOTEEVENTS.csv'

#in_file = '/exps/jive/mimicIII/NE-sample.csv'

in_patients = 'mimicIII/PATIENTS.csv'

in_diagnoses = 'mimicIII/DIAGNOSES_ICD.csv'

in_admissions = 'mimicIII/ADMISSIONS.csv'

in_phenotype_data = 'mimicIII/phenotyping/data/annotations.csv'

#w2v_model = gensim.models.Word2Vec.load('mimicIII/mimicIII.w2v')



vocab = pickle.load(open("vocab-final.pickle", "rb" ))

in_path = 'mimicIII/out_train'



c = 0

for word in vocab.keys():

    if vocab[word] > 1:

        c+=1



print(c)





#vocab = defaultdict(int)





df_raw = pd.read_csv(in_file)

df_raw = df_raw.fillna(null_str)



df_patients = pd.read_csv(in_patients)

df_patients = df_patients.fillna(null_str)



df_admissions = pd.read_csv(in_admissions)

df_admissions = df_admissions.fillna(null_str)



df_diagnoses = pd.read_csv(in_diagnoses)

df_diagnoses = df_diagnoses.fillna(null_str)



df_phenotype = pd.read_csv(in_phenotype_data)

df_phenotype = df_phenotype.fillna(null_str)





d_text = df_raw.groupby('SUBJECT_ID')['TEXT'].apply(list).to_dict()

d_cat = df_raw.groupby('SUBJECT_ID')['CATEGORY'].apply(list).to_dict()



#print(df_raw['CATEGORY'].unique())



d_desc = df_raw.groupby('SUBJECT_ID')['DESCRIPTION'].apply(list).to_dict()

d_error = df_raw.groupby('SUBJECT_ID')['ISERROR'].apply(list).to_dict()

d_admission_id = df_raw.groupby('SUBJECT_ID')['HADM_ID'].apply(list).to_dict()



df_raw['CHARTDATE'] = df_raw['CHARTDATE'].apply(lambda x: pd.to_datetime(x, dayfirst=True,errors='coerce'))

d_date = df_raw.groupby('SUBJECT_ID')['CHARTDATE'].apply(list).to_dict()



df_admissions['ADMITTIME'] = df_admissions['ADMITTIME'].apply(lambda x: pd.to_datetime(x, dayfirst=True,errors='coerce'))

d_adm_date = df_admissions.groupby('HADM_ID')['ADMITTIME'].apply(list).to_dict()



# patient info



d_sex = df_patients.groupby('SUBJECT_ID')['GENDER'].apply(list).to_dict()

d_dead_bool = df_patients.groupby('SUBJECT_ID')['EXPIRE_FLAG'].apply(list).to_dict()



now = pd.Timestamp(dt.datetime.now())

df_patients['DOB'] = df_patients['DOB'].apply(lambda x: pd.to_datetime(x, dayfirst=True,errors='coerce'))

df_patients['AGE'] = (abs(now-df_patients['DOB'])).astype('<m8[Y]')

d_age = df_patients.groupby('SUBJECT_ID')['AGE'].apply(list).to_dict()



# diagnoses



df_diagnoses.sort_values(['HADM_ID', 'SEQ_NUM'], ascending=[True, True], inplace=True)

d_diag = df_diagnoses.groupby('HADM_ID')['ICD9_CODE'].apply(list).to_dict()





list_target_dia = defaultdict(int)



d_phenotype = df_phenotype.groupby('subject.id')['cohort'].apply(list).to_dict()

d_pheno_admid = df_phenotype.groupby('Hospital.Admission.ID')['subject.id'].apply(list).to_dict()

'''
for x in train_crp:
    x_fl = float(x)
    cohort_id = str(d_pheno_admid[x_fl][0]) + '@' + x
    gen_id_list.append(cohort_id)
    #print(cohort_id)
'''



for pheno_adm_id in d_pheno_admid:



    dia_list = d_diag[pheno_adm_id]

    main_dia_list = []



    for i in range(3):

        if len(dia_list) > i:

            # if i == 1:

            #     main_dia_list.append(dia_list[i][0:3])

            # else:

            main_dia_list.append(dia_list[i][0:2])



    main_dia_str = ''.join(main_dia_list)

    list_target_dia[main_dia_str]+=1


for w in sorted(list_target_dia, key=list_target_dia.get, reverse=True):



    print('%s;%d' %(w, list_target_dia[w]))


#print(len(d_raw.keys()))



nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

nlp.add_pipe(nlp.create_pipe('sentencizer'))

par_lengths = []

#subject.id


gen_id_count=0


def get_keyword_str(init_str, in_keyword_list, one_key=False):
    #print(in_keyword_list)
    init_keyword_list = [keyword[1].split(' ') for keyword in in_keyword_list]

    init_keyword_list = sorted(init_keyword_list, key=len, reverse=True)

    result =[]

    final_keyword_dict_local = defaultdict(list)
    key_scores = []

    for n,keyword_array in enumerate(init_keyword_list):


        keyword_array_final = []

        for keyword_part in keyword_array:

            if not keyword_part.isdigit():
                keyword_array_final.append(keyword_part)

        position_list = find_substring(keyword_array, init_str.split())

        for position in position_list:

            keyword_str_final = ' '.join(keyword_array_final)

            if len(keyword_str_final) > 0:
                key_tuple = in_keyword_list[n]
                final_keyword_dict_local[position].append((keyword_str_final,key_tuple[0]))
                key_scores.append(key_tuple[0])

                for x in range(position+1, position+1+len(keyword_array)):
                    final_keyword_dict_local[x].append(('', 0.0))

   
    if len(np.array(key_scores))>0:
        
        max_score = np.max(np.array(key_scores))
    else:
        max_score = 0.0
   

    for position in range(len(init_str.split(' '))):

        if position in final_keyword_dict_local:
            # print(final_keyword_dict_local[position][0])

            key_phrase = final_keyword_dict_local[position][0]
            #  print(key_phrase)
            if one_key:

                if len(key_phrase[0]) > 0 and key_phrase[1]==max_score:
                    result.append(key_phrase[0])
                    print(init_str)
                    print(in_keyword_list)
                    print(key_scores)
                    print(max_score)
                    print(key_phrase[0])
                    break

            else:
                if len(key_phrase[0]) > 0:
                   result.append(key_phrase[0])
    return result


            # if token.text.lower() in w2v_model:

            #

            #     final_par_str.append(w2v_model.wv.most_similar(positive=[token.text.lower()], topn=1)[0][0])

            #

            # else:

            #

            #     final_par_str.append(token.text)


list_train_dia = defaultdict(int)
adm_ids = []
key_ids = []
adm_sent_dict = defaultdict(int)
keyphrase_lengths = []
sent_lengths = []
keyphrase_per_adm = []
keyphrase_per_adm_50 = []
keyphrase_per_adm_one = []

for key in d_text:



    record_list = []
    keys_list = []
    meta_list = []



    #if key in d_phenotype:

    #    continue



    main_info_str_array = []



    main_info_str_array.append(d_sex[key][0])

    main_info_str_array.append(str(d_age[key][0])[0:-3])

    main_info_str_array.append(str(d_dead_bool[key][0]))

    '''
    dia_list = d_diag[key]

    main_dia_list = []



    for i in range(3):

        if len(dia_list) > i:

            main_dia_list.append(dia_list[i][0:2])



    main_dia_str = ''.join(main_dia_list)
    '''



    '''
    if main_dia_str not in list_target_dia:



        continue

    '''

    # for t in range(len(dia_list)):

    #

    #     if dia_list[t] in list_target_dia:

    #         dia_counter += 1

    #

    # if dia_counter < 3:

    #     continue



    '''

    for i in range(5):

        if len(dia_list) > i:

            main_info_str_array.append(dia_list[i][0:3])

    '''

    pars_list = d_text[key]

    for event_counter, pars in enumerate(pars_list):

        adm_id = d_admission_id[key][event_counter]

        final_id = str(key)+'@'+str(adm_id)

        #print(final_id)
        if final_id not in gen_id_list:
            #exit()
            continue

        gen_id_count += 1
        print(gen_id_count)

        category = (d_cat[key][event_counter]).lower()



        if category != 'discharge summary':



            continue


        #par_keyword_list = keywords.keywords(pars).split('\n')

        par_list = pars.split('\n\n')



        par_keyword_list = []
        par_keyword_list2 = []

        par_sent_list = []

        for par in par_list:

            par = re.sub('\n', ' ', par)

            par = re.sub('\s+', ' ', par)

            par = re.sub('^\s', '', par)

            par = re.sub('\d{4}-\d{1,2}-\d{1,2}', 'date', par)

            # par = re.sub(r'\*\*', '', par)

            par = re.sub(r'\[\*\*([^\(\*\]\s]+) ?\(?[^\*]*\*\*\]', r'\1', par)

            r.extract_keywords_from_text(par)

            par_keyword_tuple_list = r.get_ranked_phrases_with_scores()

            score_med = np.median(np.array([k[0] for k in par_keyword_tuple_list]))

            par_keyword_list_prep = [k for k in par_keyword_tuple_list if k[0] > score_med]
            par_keyword_list_prep2 = [k for k in par_keyword_tuple_list]

            #print(par_keyword_tuple_list)


            # par_keyword_list_prep = r.get_ranked_phrases()
            par_keyword_list.extend(par_keyword_list_prep)
            par_keyword_list2.extend(par_keyword_list_prep2)

            spacy_par = nlp(par)

            par_sent_list.extend(spacy_par.sents)



        #par_list = [pars]

        prev_heading = ''



        for par_number,spacy_doc in enumerate(par_sent_list):



            heading_array = []

            heading_end =-1

            heading = ''

            heading_counter = 0



            for i in range(len(spacy_doc)):



                if len(spacy_doc) > i:



                    tok_text = spacy_doc[i].text



                    if spacy_doc[i].text == ':' and heading_counter<6:



                        heading = (' '.join(heading_array)).lower()

                        if heading_dict[heading] >= 3:

                            

                            if heading_end == -1:

                                prev_heading = heading

                            else:

                                prev_heading += ' '+heading

                           

                            heading_end = i

                            heading_array = []

                            heading_counter = 0

                        #      #break

                        #heading_dict[heading] += 1

                        #heading_array = []

                        #heading_counter = 0





                    else:



                        heading_array.append(spacy_doc[i].text)

                        heading_counter+=1



            '''

            final_par_str = [token.text for token in spacy_doc]

            for token in spacy_doc:

                vocab[token.text.lower()] += 1

            '''





            final_par_str = []

            final_keyword_str = []

            #final_par_str.append(token.text)
            # r.extract_keywords_from_text(spacy_doc_str)
            #
            # par_keyword_list = r.get_ranked_phrases()

            # par_keyword_tuple_list = r.get_ranked_phrases_with_scores()
            #
            # #score_med = np.median(np.array([k[0] for k in par_keyword_tuple_list]))
            # keyword_scores = np.array([k[0] for k in par_keyword_tuple_list])
            #
            # score_med =0.0
            #
            # if (len(keyword_scores)) > 0:
            #
            #     score_med = np.percentile(keyword_scores, 25)
            #
            # print(par_keyword_tuple_list)
            # print(score_med)
            #
            # par_keyword_list = [k[1] for k in par_keyword_tuple_list if k[0] > score_med]
            # par_keyword_list2 = [k[1] for k in par_keyword_tuple_list if k[0] >= score_med]

            #print(spacy_doc_str)
            #print(r.get_ranked_phrases_with_scores())

            #par_keyword_list2 = [k[1] for k in par_keyword_tuple_list if k[0] >= score_med]

            #print(par_keyword_list)


            for m, token in enumerate(spacy_doc):

                if m > heading_end:

                #if m > heading_end and m <= heading_end + 92:

                #if True:

                    if token.text.lower() not in vocab:

                        final_par_str.append('unk')

                    else:

                        if vocab[token.text.lower()] > 1:

                            final_par_str.append(token.text)

                        else:

                            final_par_str.append('unk')

            final_par_str_local = (' '.join(final_par_str)).lower()

            final_keyword_str= get_keyword_str(final_par_str_local, par_keyword_list)
            final_keyword_str_full = get_keyword_str(final_par_str_local, par_keyword_list2)
            final_keyword_str_one = get_keyword_str(final_par_str_local, par_keyword_list2, one_key=True)
            keyphrase_lengths.extend([len(key_phrase.split(' ')) for key_phrase in final_keyword_str_full])

            if len(final_keyword_str_full) ==0:

                if len(final_par_str) >= 5:

                    final_keyword_str_full.append(final_par_str[0])

                else:

                    final_keyword_str_full.append('null')

            if len(final_keyword_str) == 0 and len(final_keyword_str_full)>0:
                final_keyword_str = final_keyword_str_full
            if len(final_keyword_str_one) == 0 and len(final_keyword_str_full)>0:
                final_keyword_str_one = [final_keyword_str_full[0]]

            #print(len(final_par_str))
            if len(final_par_str) >= 5:

                

                info_str_array = []
                info_str_array2 = []



                info_str_array.extend(main_info_str_array)

                #info_str_array.append(d_cat[key][event_counter])

                info_str_array.append(d_desc[key][event_counter])



                event_date = d_date[key][0]


                if adm_id in d_diag:

                    dia_list = d_diag[adm_id]

                    main_dia_list = []
                    
                    
                    
                    for i in range(3):
                    
                        if len(dia_list) > i:
                    
                            main_dia_list.append(dia_list[i][0:2])
                   
                    main_dia_str = ''.join(main_dia_list)
                    list_train_dia[main_dia_str] += 1
                    #
                    #
                    #
                    #
                    #
                    # if main_dia_str not in list_target_dia:
                    #
                    #
                    #
                    #     continue

                    for i in range(5):

                        if len(dia_list) > i:

                            info_str_array.append(dia_list[i][0:3])



                # else:
                #
                #     continue



                if adm_id in d_adm_date:



                    adm_date = d_adm_date[adm_id][0]



                    event_span = (event_date - adm_date).days



                    for day_count in time_span_dict:



                        if event_span <= day_count:

                            info_str_array.append(time_span_dict[day_count])

                            break

                

                info_str_array.append(str(par_number))

                if len(prev_heading)>0:

                    info_str_array2.append(prev_heading.lower())

                else:

                    info_str_array2.append('null')


                info_str_array2.append(' '.join(final_keyword_str_full).lower())

                keyphrase_per_adm.append(len(final_keyword_str_full))

                info_str_array2.append(' '.join(final_keyword_str).lower())

                keyphrase_per_adm_50.append(len(final_keyword_str))
                
                info_str_array2.append(' '.join(final_keyword_str_one).lower())
                

                meta_list.append((' '.join(info_str_array)).lower() +'\t'+('\t'.join(info_str_array2)).lower())

                keys_list.append(str(key)+'@'+str(adm_id))

                adm_ids.append(adm_id)
                key_ids.append(key)

                record_list.append((' '.join(final_par_str)).lower())

                sent_lengths.append(len(final_par_str))

                adm_sent_dict[adm_id] +=1


    if len(record_list) > 0:

        keys_text[key] = record_list

        keys_meta[key] = meta_list

        keys_keys[key] = keys_list


print('**************************')
print('Stat patient -- admission')

print('Patients')
print(len(set(key_ids)))
print('Adm')
print(len(set(adm_ids)))

print('**************************')
print('Stat admission --  sent')

adm_sent_dict = np.array(list(adm_sent_dict.values()))
print(stats.describe(adm_sent_dict))


print('**************************')
print('Stat sent_lengths')

sent_lengths = np.array(sent_lengths)
print(stats.describe(sent_lengths))


print('**************************')
print('Stat keyphrase_lengths')

keyphrase_lengths = np.array(keyphrase_lengths)
print(stats.describe(keyphrase_lengths))

print('**************************')
print('Stat keyphrase_per_sent')

keyphrase_per_adm = np.array(keyphrase_per_adm)
print(stats.describe(keyphrase_per_adm))

print('**************************')
print('Stat keyphrase_per_sent_50')

keyphrase_per_adm_50 = np.array(keyphrase_per_adm_50)
print(stats.describe(keyphrase_per_adm_50))

print('**************************')
print('Heading Stat')

total_diag = np.sum(np.array(list(heading_dict.values())))

for w in sorted(heading_dict, key=heading_dict.get, reverse=True):



    print('%s;%.2f' %(w, heading_dict[w]/total_diag))



print('**************************')
print('Dia Stat')

for w in sorted(list_train_dia, key=list_train_dia.get, reverse=True):



    print('%s;%d' %(w, list_train_dia[w]))


#pickle.dump(heading_dict, open("heading.pickle", "wb"))



# for key in heading_dict:
#
#
#
#
#
#     print('%s;%d' %(key, heading_dict[key]))
#
#
#
#
#
# pickle.dump(heading_dict, open("heading.pickle", "wb"))
#
# pickle.dump(vocab, open("vocab.pickle", "wb"))

# keys_list = list(keys_text.keys())
#
#
#
# np.random.shuffle(keys_list)
#
# data = np.arange(len(keys_list))
#
# dev_test = keys_list[:nb_test+nb_dev]
#
# dev = dev_test[:nb_dev]
#
# test = dev_test[nb_dev:]
#
# train = keys_list[nb_test+nb_dev:]



with open(in_path + '/'+sys.argv[2]+'.txt', 'w') as ftr:

    ftr.writelines(["%s\n" % '\n'.join(keys_text[i])  for i in keys_text.keys()])



with open(in_path + '/'+sys.argv[2]+'.keys', 'w') as ftr:

    ftr.writelines(["%s\n" % '\n'.join(keys_keys[i])  for i in keys_keys.keys()])



with open(in_path + '/'+sys.argv[2]+'.meta', 'w') as ftr:

    ftr.writelines(["%s\n" % '\n'.join(keys_meta[i])  for i in keys_meta.keys()])





