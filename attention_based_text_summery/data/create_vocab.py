import re
import collections
words = re.findall('\w+', open('training-lewis.txt').read().lower())
with open('vocab', 'wb') as f:
  for k, v in collections.Counter(words).iteritems():
    f.write(str(k) + " " + str(v)+'\n')
