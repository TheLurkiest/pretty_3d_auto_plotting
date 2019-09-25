import urllib.request















import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats

# iris data projected onto 3d scatter plot:

from sklearn.datasets import fetch_openml

fetch_name = ''
print('This program is designed for the user to be able to visualize abstract concepts more easily-- my plugging the values for features into 3d scatter plots to see the ways in which these datasets are structures a little bit more clearly.\n')

print('If you don\'t yet know the name of the dataset you want to use, or you want to enter an ID directly simply hit enter at the following prompt (and enter a SECOND time at the prompt after that, if you don\'t yet know the ID you want to use either).  After that you will be given the option of entering a search term that will automatically search through the openml website for various datasets that are associated with that particular search term you requested -- as well as their ID-- for and allow you to pick one of them by entering that ID at the prompt that shows up after you make your search.  If the program didn\'t get you any results, or you weren\t satisfied with the results you were given, you can simply hit enter again when an ID is asked for, to make another search-- and can repeat this process again and again until you finally get the results you want.')

fetch_name = input('Enter the name of the dataset you wish to plot onto a 3d scatter plot (a few examples of some choices would be "iris", "wine", or "boston"):')



tested_out_num = 1
iris = 1

iris = fetch_openml(name=str(fetch_name), version = 'active')











# =========================================================
fetch_name = input('ctrl+c to end code at this break')
fetch_name = input('ctrl+c to end code at this break')
fetch_name = input('ctrl+c to end code at this break')

















def oneBookendGrab(textToCutSingle,startSign):
	bookendSingle=textToCutSingle.find(startSign)
	# DEBUGGING: THISSSSSSSSSS PRINT LINE HERE PRINTS OUT A GOOD CHUNK!
	print(str(textToCutSingle[bookendSingle+(len(startSign)):]))
	# IT RETURNS AS: musicTrack1
	return (textToCutSingle[bookendSingle+(len(startSign)):])


def PlugTrackIntoArray(ToCutSingle,endSign):
	#textToCutSingle = "<a href=\"d"
	#textToCutSingle = ToCutSingle
	textToCutSingle = ToCutSingle
	bookendSingle=textToCutSingle.find(endSign)
	return (textToCutSingle[:bookendSingle])


def quickGrabAll(inputUrl,outputText):
	grabBytes= urllib.request.urlopen(inputUrl)
	bText = grabBytes.read()
	outputText = bText.decode("utf8")
	grabBytes.close()
	return outputText


def finalTrackExtract(endOfTrack1,musicTrack1,total_paragraphs_corrected):
	numSongsLeft=0
	numSongsLeft=total_paragraphs_corrected
	finalTrack1=""
	headsUpSign1=""
	list_all_choices = []
	#if(len(list_all_choices) == 0):
	#	quickGrabAll(inputUrl,outputText)
	#	numSongsLeft=musicTrack1.count(headsUpSign1)
	#	numSongsLeft = total_paragraphs_corrected
	
	
	if (musicTrack1.count("duckduckgo")>=1 or 1 == 1):
		print("duckduckgo search engine results displaying: ")
		#headsUpSign1="<p>"
	else:
		headsUpSign1 = "<a href=\"d"

	#if(endOfTrack1.count("Text") > 0 or 7 == 7):
	#	headsUpSign1="<a href=\"d"
		#headsUpSign1="Text\":\""

	headsUpSign1 = "<a href=\"d"
	
	#numSongsLeft=musicTrack.count(headsUpSign1)
	musicTrackArray1=""
	musicTrackArray2=[]
	countForRef=0
	
	
	while (numSongsLeft>1):
		numSongsLeft=numSongsLeft-1
		print("Paragraphs left:", numSongsLeft)
		print("Paragraphs left:", numSongsLeft)
		print("Paragraphs left:", numSongsLeft)
		print("Paragraphs left:", numSongsLeft)
		print("Paragraphs left:", numSongsLeft)
		
		musicTrack1=oneBookendGrab(musicTrack1,headsUpSign1)
		
		print("HERE IS A BIG CHUNK: " + str(musicTrack1))
		list_all_choices.append(musicTrack1[:( int(musicTrack1.find(")</a></div>")) -2)])
		print("All options so far: " + str(list_all_choices))
		
		
		finalTrack1=PlugTrackIntoArray(musicTrack1,endOfTrack1)
		print(finalTrack1)
		
		countForRef=countForRef+1
		if (musicTrack1.count("duckduckgo")>=1 and countForRef==1):
			print("duckduckgo search engine results displaying: ")
			musicTrackArray1=str(str(musicTrackArray1)+"  Paragraph "+str(countForRef)+": "+str(finalTrack1))
		else:
			musicTrackArray1=str(str(musicTrackArray1)+"  Paragraph "+str(countForRef)+": "+str(finalTrack1))
		#if (numSongsLeft > 7 and countForRef>3):
		if (numSongsLeft < 1 and countForRef > 0):
			return musicTrackArray1
		print(musicTrackArray1)
	return musicTrackArray1


mark_start = "<a href=\"d"
mark_end = ")</a></div>"

playerReply = '333'
playerReplyU = '333'

musicTrack = ''

if (playerReplyU.count("3")>=1):
	
	fileContent=input("Enter a term or phrase to search for within the openml.org dataset storing website: ")
		
	stuffFromWebsite=""
	fileContent=str(fileContent).lower()

	wikiString=fileContent
	duckString=fileContent

	webpageG=""
	print("First we will search for a wikipedia article on the search term you entered: ")
	try:
		webpageG = quickGrabAll(("https://www.openml.org/search?q=" + str(wikiString) + "&type=data",webpageG))
		
		musicTrack = ''
		musicTrack=(""+str(webpageG))
		endOfTrack=""
		#endOfTrack="</p>"

		endOfTrack=")</a></div>"
		
		finalArray=""
		finalArray=finalTrackExtract(endOfTrack,musicTrack)
		print(str(finalArray))
	except:
		print("Sorry we could not find a wikipedia article on this subject")

	print(">>>Now we perform a secondary search using the duckduckgo search engine: ")
	
	webpageG = quickGrabAll(("https://www.openml.org/search?q="+str(duckString)+"&type=data"),webpageG)
	
	musicTrack=""
	musicTrack=(""+str(webpageG))
	
	endOfTrack=")</a></div>"
	
	finalArray=""
	
	finalArray=finalTrackExtract(endOfTrack,musicTrack, int(webpageG.count("<a href=\"d/")))
	
	print(str(finalArray))




