# Load Packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import to_tex
import scipy
from winsor import winsor

## Set dropbox_pathpath
dropbox_path = 'C:\\.......\\Dropbox\\NLP_project'



#%%
""" 
LOAD IN THE TEXT DATA
"""

## Only input is the path to the dropbox folder
def load_in_data(dropbox_path):
    
    os.chdir(dropbox_path)

    ## Set the text-file path
    tf_path = 'Inputs\\Task 1. Hypotheses retrieval\\text_data\\done'
    
    ## List the file names in the folder
    files = [x for x in os.listdir(tf_path) if '.txt' in x]
    
    ## Screen out the files that didn't transfer correctly
    errors = pd.read_excel('Inputs\\Task 1. Hypotheses retrieval\\docs_with_transfer_errors.xlsx')['Doc'].tolist()
    
    ## Need to append .txt 
    errors = [x+'.txt' for x in errors]
    
    
    ## Get the set difference
    files = list(set(files) - set(errors))
    files.sort()
    
    
    text_data = ['']*len(files)
    
    i = 0
    for j in files:
        with open('{}\\{}'.format(tf_path,j), encoding='utf-8') as file:
            text_data[i] = file.read()
        i = i+1
    
    ## Create a dict that matches file names to their text counterparts
    
    text_files_dict = {files[i]: text_data[i] for i in range(len(files))} 
    
    return files, text_data, text_files_dict, errors

files, text_data, text_files_dict, error_files = load_in_data(dropbox_path)

#%%

"""
REMOVE PATTERNS, COMMON TEXT
"""
import re
from nltk.tokenize import sent_tokenize
import functools
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams

""" John's cleaning"""
def clean_and_convert_to_sentence(text_file):
    
    j = text_file
    ## First, remove everything post the word REFERENCES or BIBLIOGRAPHY.
    ## Attach \n in case it appears in text 
    seps = ['REFERENCES\n','References\n','BIBLIOGRAPHY\n',
            'REFERENCES \n','References \n','BIBLIOGRAPHY \n']
    for sep in seps:
        if sep in j:
            j = j.replace(sep,sep.replace(' ',''))
            j = j.rsplit(sep, 1)[0]
    
    j = j.splitlines()

  
    ## Load in the patterns file
    xl = pd.ExcelFile('Inputs\\Task 1. Hypotheses retrieval\\patterns.xlsx')
    remove = xl.parse('Sheet1', header=None).rename(columns = {0:'remove',1:'comment'})\
             ['remove'].astype(str).tolist()    
    
    ## This parses the items in remove into lines.
    i = 0 
    for item in remove:
        if i == 0:
            lines_to_remove = item.splitlines()
        else:
            lines_to_remove = lines_to_remove + item.splitlines()
        i = i + 1
        
    ## Remove empty lines
    lines_to_remove = [x for x in lines_to_remove]
    
    ## Just remove these lines by list comprehension
    j2 = [x for x in j if x not in lines_to_remove]
    
    ## Drop any line that is just a number or number and symbols
    j3 = [i for i in j2 if re.search('[a-zA-Z]', i.strip())]
    ## Drop any line if it is just a single character
    j3 = [i for i in j3 if len(i)>1]
    
    ## Drop any line that BEGINS with a month
    months = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']
    
    months = [i.upper() for i in months]
    
    j3 = [i for i in j3 if i.split()[0].upper() not in months]
    
    
    ## Check for lines that end in a -
    
    for k in j3:
        if k.strip().endswith('-'):
            num = j3.index(k)
            # the end of the word is at the start of next line
            end = j3[num+1].split()[0]
            # we remove the - and append the end of the word
            j3[num] = k[:-1] + end
            # following space from the next line
            j3[num+1] = j3[num+1][len(end)+1:]
    ## Remove lines that contain "This content downloaded" and other download things
    
    for string in ['This content downloaded','http','jstor','DOI','doi']:
        j4 = [x for x in j3 if string not in x]
    
    
    
    
    ## 1. Remove lines IP addresses 
    regex = re.compile(r'(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})')
    j5 = [i for i in j4 if not regex.search(i)]
    
    
    
    ## 3. Just remove everything in parentheses??
    ## Also need to worry about parentheses splitting over lines. 
    ## This bit of code inserts a linesplit placeholder and merges the text, 
    ## and then removes anything within parentheses
    j6 = j5
    for k in j6:
        j6[j6.index(k)] = k + " {LINESPLIT}"
    
    j6 = ''.join(i for i in j6)
    regex = r'\(.*?\)'
    j6 = re.sub(regex, '', j6)
    j6 = j6.split('{LINESPLIT}')
    j6 = [i for i in j6 if not i == '' ]
    j6 = [i for i in j6 if re.search('[a-zA-Z]', i.strip())]

    j7 = ''.join(i+' ' for i in j6)
    j7 = sent_tokenize(j7)
    
    ## replace double-spaces 
    for k in j7:
        j7[j7.index(k)] = k.replace('  ',' ')
        
    ## Remove lines that contain any of these strings
    for string in ['This content downloaded','http','jstor','DOI']:
        j7 = [x for x in j7 if string not in x]
    
    
    j7 = ''.join(i+' ' for i in j7)
    
    ## Replace cases of h1. or Hypothesis 1. with Hypo 1:
    ## Add a <split> signifier, so that when we extract hypotheses later
    ## we can split new lines based on <split>
    for num in np.arange(100):
        for regex in [r'h{}[a-zA-Z]\:'.format(num),r'H{}[a-zA-Z]\:'.format(num),
                      r'h{}[a-zA-Z]\.'.format(num),r'H{}[a-zA-Z]\.'.format(num),
                      r'h{}[a-zA-Z]'.format(num),r'H{}[a-zA-Z]'.format(num),
                      r'hypothesis {}[a-zA-Z]\:'.format(num),r'Hypothesis {}[a-zA-Z]\:'.format(num),
                      r'hypothesis {}[a-zA-Z]\.'.format(num),r'Hypothesis {}[a-zA-Z]\.'.format(num),
                      r'hypothesis {}[a-zA-Z]'.format(num),r'Hypothesis {}[a-zA-Z]'.format(num),
                      r'h{}\:'.format(num),r'H{}\:'.format(num),
                      r'h{}\.'.format(num),r'H{}\.'.format(num),
                      r'h{}'.format(num),r'H{}'.format(num),
                      r'hypothesis {}\:'.format(num),r'Hypothesis {}\:'.format(num),
                      r'hypothesis {}\.'.format(num),r'Hypothesis {}\.'.format(num),
                      r'hypothesis {}'.format(num),r'Hypothesis {}'.format(num)]:
            j7 =re.sub(regex,'<split>Hypo {}: '.format(num),j7)


    ## Just get rid of some annoying strings
    j7 = j7.replace(':  :',': ')
    ## Replace multiple white spaces with a single white space
    j7 = re.sub('\s+',' ',j7)
    ## Remove this string as it causes new sentences and funks up hypothesis extraction
    j7 = j7.replace(': .',':')

    return j7
    



""" Venetia and Felipe's cleaning"""

def preprocess(sent):
	sent = nltk.word_tokenize(sent)
	sent = nltk.pos_tag(sent)
	return sent


def rm_breaks(text, beta):


	#Convert to lower
    text = text.lower()
	#Remove commas
    text = text.replace(',', '')
	#Remove DOIs
    text = re.sub(r'\d+\.\d+/\w+', '', text)
    text = re.sub(r'doi:*', '', text)
    #Replace 'hypothesis 1' with 'h1'
    text = text.replace('hypotheses', 'hypothesis')
    text = re.sub(r'hypothesis (?=\d+)', 'h', text)
    #Remove numbers that dont have a character immediately before them (since H0 indicates hypothesis)
    text = re.sub(r'\W+\d+', '', text)
    text = re.sub(r'\d{2,4}', '', text)
    #Replace jstor link with 'jstor' placeholder, then delete
    text = re.sub(r'https?://.+', 'jstor.', text)
    # text = re.sub(r'\S+\.jstor\.\S+', 'jstor.', text))
    text = re.sub(r'\.{2,}|:', '', text)
    text = re.sub(r'this\scontent.+', '', text)
    #Delete jstor placeholder
    check = re.sub(r'.*jstor.*', '', text)
    if check != '':
        text = re.sub(r'.*jstor.*', '', text)
    #Remove word interruptions
    text = re.sub(r'-\s*\n\s*', '', text)
    #Remove line breaks
    text = re.sub(r'\n', '', text)
    #Do NER and remove sentences with too many named entities
    sent = preprocess(text)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    Owords = 0
    wordCount = 0
    maintext = []
    holder = []
    for i in iob_tagged:
        holder.append(i[0])
        wordCount += 1
        if i[0] == '.':
            score = Owords/wordCount
            if 'hypothesis' in holder:
                maintext += holder
                holder.clear()
            elif 'jstor' in holder:
                Owords = 0
                wordCount = 0
                holder.clear()
                continue
            elif score >= beta:
                maintext += holder
            Owords = 0
            wordCount = 0
            holder.clear()
        if i[2] == 'O':
            Owords += 1
    if maintext != []:
        maintext = functools.reduce(lambda a,b : a+' '+b,maintext)
    return maintext

#%%
text_cleaned_JB = ['']*len(text_data)

for i in np.arange(0,len(text_data)):
    print('{}% done '.format(round(i/len(text_data)*100),1))
    text_cleaned_JB[i] = clean_and_convert_to_sentence(text_data[i])    


## Save cleaned text
text_cleaned_dict_JB = {files[i]:text_cleaned_JB[i] for i in range(len(text_cleaned_JB))}
import codecs
for key in text_cleaned_dict_JB.keys():
    text = text_cleaned_dict_JB[key]
    text = text.lower()
    text_file = codecs.open('Outputs/cleaned_text_data_jb/{}'.format(key),mode='a',encoding='utf-8')
    text_file.write(text)
    text_file.close()
    
"""
text_cleaned_VF = ['']*len(text_data)
for i in np.arange(0,len(text_data)):
    print('{}% done '.format(round(i/len(text_data)*100),1))
    text_cleaned_VF[i] = rm_breaks(text_data[i],beta=0.3)    

text_cleaned_dict_VF = {files[i]:text_cleaned_VF[i] for i in range(len(text_cleaned_VF))}
"""

#%%
def extract_hyps(files,text_cleaned_dict):
    
    ## Loop through the files and organize into a dataframe
    i = 0
    for file in files:
        ## Pull text from the dictionary
        text = text_cleaned_dict[file]
        text = text.lower()
        
        ## Use NLTK tokenize to parse text into sentences
        document = sent_tokenize(text)
        ## convert all text to lower
        document = [x.lower() for x in document]
        
        
        ## Split based on the occurrence of "hypothesis:" Want these on a new line,
        ## pfls is the document-specific list of hypothesis sentences
        pfls = ['']
        for pg in document:
            ## check if sentence contains a hypothesis-like sentence
            if 'hypo' in pg or 'hypothesis' in pg or 'hypotheses' in pg or bool(re.match(r'h\d+', pg)):
                pfls.append(pg)
        pfls.remove(pfls[0])

        hyp_statements = ['']
        ## split the hyp_statements based in occurrence of hypo: (<split>)
        ## is hte split identifier
        for hyp in pfls:
            hyp_out = hyp.split('<split>')
            hyp_out = [h for h in hyp_out if 'hypo' in h]
            hyp_statements.append(hyp_out)
        hyp_statements.remove(hyp_statements[0])
        ## Flatten the list
        hyp_statements = [y for x in hyp_statements for y in x]
        
        ## Just organize the data into a dataframe
        h_cols = ['h'+str(i) for i in range(len(hyp_statements))]
        file_cols = [file]*len(hyp_statements)
        ## Concatenate by document
        if i == 0:
            hypothesis = pd.DataFrame({'file_name': file_cols,
                                       'hypothesis_num': h_cols,
                                       'sentence': hyp_statements})
        else:
            hold = pd.DataFrame({'file_name': file_cols,
                                       'hypothesis_num': h_cols,
                                       'sentence': hyp_statements})
            hypothesis = pd.concat([hypothesis, hold],axis=0,ignore_index=True)
            
        i = i+1
    return hypothesis
    
hypothesis_JB = extract_hyps(files,text_cleaned_dict_JB)

#hypothesis_VF = extract_hyps(files,text_cleaned_dict_VF)


#%%

def assign_docs_to_team(hyp_df):

    i = 0
    for q in hyp_df['file_name'].unique():
        hyp_df.loc[hyp_df['file_name']==q,'file_num']=i
        i = i+1
    
    ## Six people, split documents evenly
    files_each = np.ceil(hyp_df['file_num'].max()/6)
    
    ## Person number
    hyp_df['person_num'] = np.floor(hyp_df['file_num']/files_each)
    
    ## Person number-name match
    person_dict = {0:'Mory',
                   1:'Felipe',
                   2:'Venetia',
                   3:'Vanessa',
                   4:'John',
                   5:'Haozhe'}
    hyp_df['person'] = hyp_df['person_num'].map(person_dict)
        
    ## Keep relevant columns
    hyp_df = hyp_df[['person','file_name','hypothesis_num','sentence']]

    return hyp_df

hypothesis_JB = assign_docs_to_team(hypothesis_JB)

#hypothesis_VF = assign_docs_to_team(hypothesis_VF)

hypothesis_JB.to_excel('Outputs\\hypothesis_extraction.xlsx',engine='xlsxwriter')

#%%


def assemble_training(dropbox_path):
    
    os.chdir(dropbox_path)


    uncleaned_data = pd.read_excel('Outputs\\hypothesis_extraction.xlsx','Sheet1')
    uncleaned_data.columns = uncleaned_data.columns.str.replace(' ','')
    
    
    
    train = pd.DataFrame()
    
    files = ['hypothesis_extraction_fm',
                 'hypothesis_extraction_HZ',
                 'hypothesis_extraction_jb',
                 'hypothesis_extraction_me',
                 'hypothesis_extraction_v',
                 'hypothesis_extraction_vanessa']
    for file in files:
    
        data = pd.read_excel('Outputs\\{}.xlsx'.format(file), 'Sheet1')[['file_name', 'hypothesis_num', 'causal_sentence_0_1', 'node_1',
           'node_2', 'direction', 'causal_relationship']]
        data.columns = data.columns.str.replace(' ','')
    
        data = data.merge(uncleaned_data[['file_name','hypothesis_num','sentence']],on = ['file_name','hypothesis_num'],how='inner')
        
        train = train.append(data)
        
    train = train.loc[train['causal_sentence_0_1']==1]
    
    train = train[['file_name', 'hypothesis_num','sentence', 'node_1',
           'node_2', 'direction', 'causal_relationship']]
    
    train.to_excel('Outputs\\training_data.xlsx',index=False)
    
assemble_training(dropbox_path)
