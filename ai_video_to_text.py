#pip install youtube_transcript_api
#pip install transformers

from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

###youtube_video="video url"
youtube_video="https://www.youtube.com/watch?v=qAv_QqmEk78&list=PLNLDEHOJTZSg2bORaqLhnsgm7TsNWN0n8&index=6"

video_id= youtube_video.split("=")[1]
video_id

from youtube_transcript_api import YouTubeTranscriptApi

transcript=YouTubeTranscriptApi.get_transcript(video_id)
print(transcript)

summarizer=pipeline("summarization")

result=""
for i in transcript:
   result +=' ' + i['text']
print(result)

print(len(result))

num_iters = int(len(result)/1000)
summarized_text=[]
for i in range(0,num_iters+1):
  start=0
  start=i*1000
  end=(i + 1)*1000
  out=summarizer(result[start:end])
  out=out[0]
  out=out['summary_text']
  summarized_text.append(out )

print(summarized_text)
summarized_text_str=str(summarized_text)

f = open("transcript_sum_day6.txt", "a")
f.write(summarized_text_str)
f.close()