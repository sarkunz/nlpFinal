import spacy
import en_core_web_sm
import pandas as pd
nlp = en_core_web_sm.load()

##TODO: add embedding and context embedding

# FEATURES
features = ["LABEL", "ABBR","CAP","LOC","POS","POS+1","POS-1","PREF","SUFF","WORD","WORD+1","WORD-1","TAG","TAG+1","TAG-1"]

#locations
pd_locs = pd.read_csv('lists/locations.csv')
cntrys = [st.lower() for st in pd_locs['country'].tolist()]
capts = [st.lower() for st in pd_locs['capital'].tolist()]
locs = cntrys + capts
#prefixes
pf = open('lists/prefixes.txt', 'r')
prefixes =  [l.rstrip() for l in pf.readlines()]
#suffixes
sf = open('lists/suffixes.txt', 'r')
suffixes = [l.rstrip() for l in sf.readlines()]
#preps
ps = open('lists/prepositions.txt', 'r')
prepositions = [l.rstrip() for l in ps.readlines()]

def isAlphaPeriod(wd):
    if wd.find(".") == -1:
        return False
    noper = wd.replace(".", "")
    return noper.isalpha()

#Feed in a sentance, get rows for each word
def extractFeatures(sentence_in, labelDic):
    all_rows = []
    doc = nlp(sentence_in)
    sentence = sentence_in.split()
    for i in range(len(sentence)):
        row = []
        wd = sentence[i]
        pos = doc[i].pos_
        tag = doc[i].tag_
        start_sent = True if (i == 0) else False
        last_sent = True if (i >= len(sentence)-1) else False
        
        prevWd = 'PHI'
        prevPos = 'PHIPOS'
        prevTag = 'PHITAG'
        if not start_sent:
            prevWd = sentence[i-1]
            prevPos = doc[i-1].pos_
            prevTag = doc[i-1].tag_
        nxtWd = 'OMEGA'
        nxtPos = 'OMEGAPOS'
        nxtTag = 'OMEGATAG'
        if not last_sent:
            nxtWd = sentence[i+1]
            nxtPos = doc[i+1].pos_
            nxtTag = doc[i+1].tag_     
        
        # #to keep track of global vals
        # if wd not in glob_dict:
        #     glob_dict[wd] = {'globcap': 0, 'globpref': 0, 'globsuf': 0, 'indexes': [ar]}
        # else:
        #     glob_dict[wd]["indexes"].append(ar)

        # Read in and calc features 

        # LABEL- found in label dictionary passed into function, else "-"
        lab = "-"
        if wd in labelDic:
            lab = labelDic[wd]
        row.append(lab)


        # 1 ABBR-  (1) end with a period, (2) consist entirely of alphabetic characters [a-z][A-Z] andone or more periods (including the ending one), and (3) have length≤4
        abbr = 0
        if (wd[-1] == ".") and isAlphaPeriod(wd) and (len(wd) <= 4) and (len(wd) > 1):
            abbr = 1
        row.append(abbr)

        # 2 CAP- a binary feature indicating whether the first letter of w is capitalized
        cap = 1 if wd[0].isupper() else 0
        row.append(cap)

        # # 3 globcap- a binary feature indicating whether there is a capitalized instance ofwanywhere in the document (includingwitself).
        # if (cap == 1) and (wd.isalpha()) and (not start_sent) and (wd.lower() not in prepositions):
        #     if not wd.lower() in glob_dict:
        #         glob_dict[wd.lower()] = {'globcap': 1, 'globpref': 0, 'globsuf': 0, 'indexes': [ar]}
        #     #get all possible cases
        #     allCases = set(k for k in glob_dict if k.lower() == wd.lower())
        #     for key in allCases:
        #         glob_dict[key]['globcap'] = 1
        #         for ind in glob_dict[key]["indexes"]:
        #             if ind != ar: #don't do current row, not in all_rows yet
        #                 all_rows[ind][3] = 1 #change global val on all previous indexes

        # if wd.lower() in glob_dict and glob_dict[wd.lower()]["globcap"] == 1:
        #     row.append(glob_dict[wd.lower()]["globcap"])
        # else: row.append(glob_dict[wd]["globcap"])

        # 10 PREF- a binary feature indicating whether the wordw−1that immediately precedes w matches any of the prefix terms listed in the provided fileprefixes.txt.
        pref = 1 if (prevWd in prefixes) else 0
        # if (pref == 1) and (wd.isalpha()) and (not start_sent):
        #     glob_dict[wd]["globpref"] = 1
        #     for ind in glob_dict[wd]["indexes"]:
        #         if ind != ar: #don't do this row, not in all_rows yet
        #             all_rows[ind][4] = 1 #change global on all previous indexes

        # # 4 GLOBPREF- a  binary  feature  indicating  whether  there  is  an  instance  ofwin  thedocument (includingwitself) that has a prefix inprefixes.text
        # row.append(glob_dict[wd]["globpref"])

        # 11 SUFF- a binary feature indicating whether the wordw+1that immediately followswmatches any of the suffix terms listed in the provided filesuffixes.txt
        suf = 1 if (nxtWd in suffixes) else 0
        # if suf == 1 and (wd.isalpha()) and (not last_sent):
        #     glob_dict[wd]["globsuf"] = 1
        #     for ind in glob_dict[wd]["indexes"]:
        #         if ind != ar: #don't do this row, not in all_rows yet
        #             all_rows[ind][5] = 1 #change global on all previous indexes

        # # 5 GLOBSUF- a  binary  feature  indicating  whether  there  is  an  instance  ofwin  thedocument (includingwitself) that has a suffix insuffixes.text.
        # row.append(glob_dict[wd]["globsuf"])

        # 6 LOC- a binary feature indicating whetherwmatches any of the countries or capitalcities listed in the provided filelocations.csv.  Please docase-insensitivematchingagainst this file (e.g., “Israel” should match “ISRAEL”)
        loc = 1 if wd.lower() in locs else 0
        row.append(loc)

        # 7 POS
        row.append(pos)
        # 8 POS+1
        row.append(nxtPos)
        # 9 POS-1
        row.append(prevPos)

        # 10 PREF- a binary feature indicating whether the wordw−1that immediately precedes w matches any of the prefix terms listed in the provided fileprefixes.txt.
        row.append(pref)

        # 11 SUFF- a binary feature indicating whether the wordw+1that immediately followswmatches any of the suffix terms listed in the provided filesuffixes.txt
        row.append(suf)

        # 12 WORD
        row.append(wd)
        # 13 WORD+1
        row.append(nxtWd)
        # 14 WORD-1
        row.append(prevWd)

        # 12 WORD
        row.append(tag)
        # 13 WORD+1
        row.append(nxtTag)
        # 14 WORD-1
        row.append(prevTag)

        all_rows.append(row)
        # ar += 1  
    return all_rows

