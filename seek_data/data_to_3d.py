import urllib.request




class RandT1(object):
	""" has methods that do turtle stuff """

	def __init__(self):
		pass






	def turtlesetup(self):

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




