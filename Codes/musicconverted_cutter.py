from os import path
from pydub import AudioSegment

# files                                                                         
src = "first_half.mp3"
dst = "test.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")


#cut the lenght

from pydub import AudioSegment
sound = AudioSegment.from_file("first_half.mp3")

halfway_point = len(sound) // 2
first_half = sound[:halfway_point]

# create a new file "first_half.mp3":
first_half.export("first_half.mp3", format="mp3")