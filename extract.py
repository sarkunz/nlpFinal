import sys
import os
import subprocess
import time
from collections import OrderedDict 

# try:
#     # install various required packages
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'joblib'])
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spacy'])
#     os.system("python -m spacy download en_core_web_sm")
# except ImportError as err:
#     print("Issue installing package: \n", err)
#     exit()

import pandas as pd
import joblib
import pickle
from spacy.lang.en import English
nlp = English()
nlp.add_pipe('sentencizer')

from featureExtract import extractFeatures

"""
python3 extract.py <doclist>

TEXT:filename identifier
ACQUIRED:entities that were acquired
ACQBUS:the business focus of the acquired entities
ACQLOC:the location of the acquired entities
DLRAMT:the amount paid for the acquired entities
PURCHASER:entities that purchased the acquired entities
SELLER:entities that sold the acquired entities
STATUS:status description of the acquisition event
"""

labDict = {'TXT': 'TEXT' ,
            'AQ': 'AQUIRED',
            'AQB': 'ACQBUS',
            'AQL': 'ACQLOC',
            'AMT': 'DLRAMT', 
            'PURCH': 'PURCHASER',
            'SELL': 'SELLER', 
            'STAT': 'STATUS'
        }

#read in args
if len(sys.argv) < 2:
    print("MISSING ARGS")
    exit()
doclisttxt = sys.argv[1]
docpath = "/".join(doclisttxt.split("/")[0:-1])
if docpath != "": docpath += "/"

#constants
dictvect = joblib.load("DiCVecFull.joblib")
loaded_model = pickle.load(open('logistic_regression_model_FULL.joblib', 'rb'))
features = ["LABEL", "ABBR","CAP","LOC","POS","POS+1","POS-1","PREF","SUFF","WORD","WORD+1","WORD-1","TAG","TAG+1","TAG-1"]

## Add all docs to dataframe
docFeats = ["path", "filename", "text"]
doclist = open(doclisttxt, 'r')
docsInfo = []
for doc in doclist.readlines():
    dcInf = []
    pcs = doc.strip().split("/")
    dcInf.append("/".join(pcs[0:-1]))
    dcInf.append(pcs[-1])
    with open(docpath + doc.strip()) as txtFile:
        txt = ""
        for l in txtFile.readlines():
            txt += (" " + l.strip())
        dcInf.append(txt.strip())
    docsInfo.append(dcInf)
df = pd.DataFrame(docsInfo, columns=docFeats)

## iterate through text and perform interence
outpFile = open(doclisttxt + ".template", "w")
for text, filename in zip(df['text'].values, df['filename'].values):
    outpDict = {'TXT': [] ,'AQ': [], 'AQB': [], 'AQL': [], 'AMT': [], 'PURCH': [], 'SELL': [], 'STAT': []}
    
    #extract features from text
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    data = []
    for s in sentences:
        data += extractFeatures(s, '-') #allDics[filename])
    
    #perform interence
    featDf = pd.DataFrame(data, columns=features)
    X = featDf.drop('LABEL', axis=1)
    test_vect = dictvect.transform(X.to_dict("records"))
    pred = loaded_model.predict(test_vect)

    #Read through predictions add to dict
    for idx, p in enumerate(pred):
        postf = p.split("-")[-1]
        if postf in outpDict:
            outpDict[postf].append(X.iloc[idx]['WORD'])
        #else ignore it, probably a -

    #stringify predictions dict and add to dataframe? or just print to file
    outStr = ""
    for lab in outpDict.keys():
        outStr += (labDict[lab] + ": ")
        if lab == "TXT":
            outStr += (filename + "\n")
        elif len(outpDict[lab]) > 0:
            outStr += "\""
            outStr += " ".join([wd for wd in list(OrderedDict.fromkeys(outpDict[lab]))])
            outStr += "\"\n"
        else:
            outStr += "\"---\"\n"

    outpFile.write(outStr + "\n")

