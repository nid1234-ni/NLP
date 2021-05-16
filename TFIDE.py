import nltk
paragraph =  """Yoga is an act that unites the body with the soul.
 It is a means through which we can attain inner peace. 
 The great relaxing effect that yoga has on our minds has innumerable health benefits. 
 It originated in ancient India during the Indus Valley civilization 
 and has grown in popularity ever since. Originally
 only the Hindu priests practiced the art of yoga, but later, 
 even common people started practicing it for its health benefits.

Yoga is something that is practiced, not learned. 
You need to perform certain Asana or poses that form the essence of yoga. 
It is believed that there is total of 84 asanas in yoga. But this number may vary. 
Many of the Asana has been lost from the Vedic scriptures,
 and the poses we know today are a minute fraction of it.

Some asana is easy like the Padmasana or the lotus pose.
 Whereas, some asanas are difficult,
 such as the Salamba Shirshasana or the headstand."""
 
 # Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
    # Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()