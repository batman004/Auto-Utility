import pandas as pd
import numpy as np

from curses.ascii import NL
from spacy.symbols import nsubj, VERB, ADJ
from .information_extraction import NlpAlgos, KnowledgeGraph
from ..utility.pre_processing import clean

def IE_Operations(review):
  # create spacy doc
  doc = NlpAlgos.nlp(review)
  adjectives = set()
  verbs_all = set()
  # applying POS to each token
  print("POS Tagging : ")
  for token in doc:
      if token.pos_ not in ["SPACE", "DET", "ADP", "PUNCT", "AUX", "SCONJ", "CCONJ", "PART"]:
        print(token.text,'->',token.pos_)
      if(token.pos_=="ADJ"):
        adjectives.add(token.text)
      if(token.pos_=="VERB"):
        verbs_all.add(token.text)

  print("Dependency Graph : \n")

  print("************************************************************\n")
  NlpAlgos.dependency_graph(doc)
  print("************************************************************\n")

  print("Verb with subject : \n")

  # Finding a verb with a subject
  verbs = set()
  for possible_subject in doc:
      if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
          verbs.add(possible_subject.head)
  print(verbs)
  print("************************************************************\n")

  print("Adjectives : \n")

  # Finding adjectives with a subject
  print(adjectives)
  print("************************************************************\n")
  # NER
  for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
  print("************************************************************\n")


  print("Knowledge Graph : \n")

  KnowledgeGraph.knowledge_graph(review)

  print("************************************************************\n")


  print("Summarization of review : ")
  NlpAlgos.summarize(review)

  print("************************************************************\n")

  



def IE_brand(brand):
  path = "/content/Scraped_Car_Review_" + brand + ".csv"
  df = pd.read_csv(path,delimiter=',', nrows = 100)
  df['Review_clean'] = df['Review'].apply(clean)
  df['Review_clean'][2]
  reviews = df['Review_clean'][0:5]
  reviews = np.array(reviews)
  df2 = df["Rating"].mean()
  print(f"Mean sentiment of users associated with the brand :{df2}/5")
  i=0
  for review in reviews:
    i=i+1
    print(f"for review num :{i} \n")
    IE_Operations(review)