#***************************INSTALL ALL LIBRARIES***************************************************************************************************************************************************************************

#pip install transformers
#pip install torch
#pip install tensorflow
#pip install webvtt-py 
#pip install nltk
#nltk.download('omw-1.4')
#pip install wordcloud
#pip install matplotlib

#***************************IMPORT ALL LIBRARIES**********************************************************************************************************************************************************************************
import re 
import pandas as pd
import webvtt
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#specify the path
path='C:/Users/Hp/Downloads/Topic modelling'

#*********************EXTRACTING TIME AND CONTEXT FROM VTT FILE****************************************************************************************************************************************

caption_path = "{}/Session18_26th_dec2021.vtt".format(path)
start_time = []
end_time = []
text = []
for caption in webvtt.read(caption_path):
    start_time.append(caption.start.split(".")[0])
    end_time.append(caption.end.split(".")[0])
    try:
        text.append(caption.text.split(":")[1])
    except:
        text.append(caption.text)

#creating a dataframe 

df = pd.DataFrame()
df["start_timestamp"] = start_time
df["end_timestamp"] = end_time
df["text"] = text

#saving the csv file

df.to_csv("{}/vtt_csv_df.csv".format(path), index=False)

#**********************************TOPIC MODELLING******************************************************************************************************

#importing dataset

dataset=pd.read_csv('{}/vtt_csv_df.csv'.format(path))
tag=pd.read_excel('{}/tags.xlsx'.format(path))
dataset.dtypes

#***********************DIVIDE THE TEXT FOR ONE MINUTE EACH***********************************************************************************************************************

# Python code to convert string to list character-wise

def Convert(string):
    list1=[]
    list1[:0]=string
    return list1

#Divide the text by 1 minuites each

stac=[]
start_timestamp=[]
start_timestamp.append(dataset['start_timestamp'][0])
end_timestamp=[]
start_time=[]
st=[]
a=len(dataset)
for i in range(len(dataset)-1):
    if dataset['end_timestamp'][i][3:5]!=dataset['end_timestamp'][i+1][3:5]:
        d=list(dataset.loc[:i,'text'])
        listToStr = ' '.join([str(elem) for elem in d])
        start_timestamps=dataset['start_timestamp'][i+1]
        end_timestamps=dataset['end_timestamp'][i]
        stac.append(listToStr)
        start_timestamp.append(start_timestamps)
        end_timestamp.append(end_timestamps)
    elif  dataset['end_timestamp'][len(dataset['end_timestamp'])-1][0:5]==dataset['end_timestamp'][i][0:5]:
        d=dataset['text'][i]
        #listToStr = ' '.join([str(elem) for elem in d])
        start_times=dataset['start_timestamp'][i]
        st.append(d)
        start_time.append(start_times)


# joining context for few last timestamps as it does not fall in one minute interval

st = ' '.join([str(elem) for elem in st])


#******************************ADJUSTING THE TIMESTAMP**************************************************************************************************************************************************************

#defining start time for few last timestamps as it does not fall in one minute interval

start_timestamp=start_timestamp[:-1]
start_timestamp.append(start_time[1])   


#defining end time for few last timestamps as it does not fall in one minute interval

end_timestamp.append(dataset['end_timestamp'][len(dataset['end_timestamp'])-1])



#*****************************CONTEXT FOR TOPIC MODELLING*********************************************************************************************************************************************************************

#appending the context 

context=[]
for i in range(len(stac)):
    if i==0:
        stacs=stac[0]
    else:
        stacs=stac[i][len(stac[i-1]):]
    context.append(stacs)
context.append(st)
        

#******************************BUILDING A DATAFRAME******************************************************************************************************************************************************************************************************    

# Calling DataFrame constructor after zipping
# both lists, with columns specified

df = pd.DataFrame(list(zip(start_timestamp,end_timestamp,context)),columns =['start_timestamp', 'end_timestamp','context'])

#********************************DATA PREPROCESSING**************************************************************************************************************************************

names = context

with open("C:/Users/Hp/Downloads/Topic modelling/stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
stop_words.append('yeah')

lemmatizer = WordNetLemmatizer()
# Removing unwanted symbols incase if exists
index=names.index(names[-1])
ip_rev_strings=[]
for i in range(index+1):
    ip_rev_string = re.sub("[^A-Za-z" "]+"," ", names[i]).lower()
    ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)
    ip_rev_string=lemmatizer.lemmatize(ip_rev_string)
    ip_reviews_word = ip_rev_string.split(" ")
    ip_reviews_word = [w for w in ip_reviews_word if not w in stop_words]
    ip_rev_string = " ".join(ip_reviews_word)
    ip_rev_strings.append(ip_rev_string)

#SENTENCE EMBEDDING THIS EMBEDDING SHOWS HOW CLOSE ONE WORD IS TO OTHER BY DOING SOME SCALAR 
#PRODUCT OF THERE VECTOR EVERY WORD IS CONVERTED INTO SOMEKIND OF VECTOR REPRESENTATION .

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-distilroberta-base-v2')
candidate_embeddings = model.encode(tag['Description'])
from sklearn.metrics.pairwise import cosine_similarity
suit=[]
link=[]
for i in range(len(ip_rev_strings)):
    doc_embedding = model.encode([ip_rev_strings[i]])
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    dist=distances[0][np.where(distances[0]>0.3)]
    if len(dist)==2:
        tap=list(tag['Topic'][distances[0]==dist[0]]),list(tag['Topic'][distances[0]==dist[1]])
        lin=list(tag['Link'][distances[0]==dist[0]]),list(tag['Link'][distances[0]==dist[1]])
    elif len(dist)>2:
        tap=list(tag['Topic'][distances[0]==dist[0]]),list(tag['Topic'][distances[0]==dist[1]]),list(tag['Topic'][distances[0]==dist[2]])
        lin=list(tag['Link'][distances[0]==dist[0]]),list(tag['Link'][distances[0]==dist[1]]),list(tag['Link'][distances[0]==dist[2]])
    elif len(dist)==0:
        tap=[]
        lin=[]
    elif len(dist)==1:
        tap=list(tag['Topic'][distances[0]==dist[0]])
        lin=list(tag['Link'][distances[0]==dist[0]])
    tap=list(tap)
    suit.append(tap)
    link.append(lin)

df['Topics']=suit

df['Link']=link
#***************************BUILDING A WORD CLOUD FOR ALL THE TOPICS*****************************************************************************************************************************************************************************************

text = ' '.join([str(elem) for elem in suit])

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#******************************SAVING THE EXCEL FILE WITH TOPICS******************************************************************************************************************************************************************************

file_name = 'Top.xlsx'
  
# saving the excel
df.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')


#********************************CONVERTING TO JSON FORMAT****************************************************************************************************************************

#converting dictionary to json
import json 
# Serializing json  
json_object = json.dumps(df.to_dict('list'), indent = 4) 
print(json_object)
