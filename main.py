import numpy as np
import pandas as pd I
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.sodel_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.eetrics import accuracy_score
import nltk
nltk.download(* stopwords’)