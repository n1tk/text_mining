#kennesawstate tweets data cleanup and wordcloud

tweets.df <- read.csv(file="/Users/sbuciuma/Desktop/School/School_fall_2016/text_mining/project_proposal/kennesawstate_tweets.csv", header=TRUE)
class(tweets.df)
tweets.df
tweets.df$text
#dimmension of the dataframe
dim(tweets.df)
head(tweets.df)

library(twitteR)
library(stringr)
library(tm)
library(wordcloud)

#extracting the text from the tweets because the dataframe basically has the retweets and counts etc
##so are 16 variables i dataframe so we just extract the text column
mach_text <- tweets.df$text
mach_text


# clear the tweets from all data, basically
### is removing punctuation, numbers removed and all charachters and also all links 
####so we have the clean text from all 300 tweets from the dataframe and create a corpus.

t <- iconv(mach_text,to="utf-8-mac")
tweets.text.cleaned <- gsub("@\\w+ *#", "", t)
tweets.text.cleaned <- gsub("(f|ht)tp(s?)://(.*)[.][a-z]+", "", tweets.text.cleaned)
tweets.text.cleaned <- gsub("[^0-9A-Za-z///' ]", "", tweets.text.cleaned)
tweets.text.corpus <- Corpus(VectorSource(tweets.text.cleaned))
tweets.text.final <- tm_map (tweets.text.corpus, removePunctuation, mc.cores=1)
tweets.text.final2 <- tm_map (tweets.text.final, content_transformer(tolower), mc.cores=1)
tweets.text.final2 <- tm_map(tweets.text.final2, removeNumbers, mc.cores=1)
tweets.text.final2 <- tm_map(tweets.text.final2, removePunctuation, mc.cores=1)
tweets.text.final2 <- tm_map(tweets.text.final2,removeWords, stopwords("English"), mc.cores=1)
tweets.text.final2 = tm_map(tweets.text.final2, removeWords, c("amp", "&", "xfxfxxd"))
#for removing earthquak from the words so will have a better freq of the words. so just need
#to be uncomented below row.
tweets.text.final2 = tm_map(tweets.text.final2, removeWords, c("amp", "&", "earthquake"))

mach_corpus = Corpus(VectorSource(tweets.text.final2))

eartsCorpus1 <- tm_map(mach_corpus, PlainTextDocument)

eartsCorpus <- tm_map(eartsCorpus1, stemDocument)
eartsCorpus

#black color words on white background Plot the 100 most frequently occurring words.
wordcloud(eartsCorpus, min.freq = 1,
          max.words=200, random.order = FALSE)


####colored Plot the 100 most frequently occurring words.
set.seed(142)   
dark2 <- brewer.pal(8, "Dark2")  
wordcloud(eartsCorpus, min.freq = 1,
          max.words=200, colors=dark2)   

#### different colors
set.seed(142)   
dark2 <- brewer.pal(6, "Dark2")  
wordcloud(eartsCorpus, min.freq = 1,
          max.words=200, rot.per=0.35, colors=dark2) 


###########Another moethod by creating the word column and freq column ############
####this method does provide  color and alignment different#####

dtm <- TermDocumentMatrix(tweets.text.final2)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 200)

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=300, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
