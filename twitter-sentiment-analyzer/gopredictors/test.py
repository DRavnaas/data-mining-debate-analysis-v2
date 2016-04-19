import re
import string

tweet = "swwet :)hi. I am sad :(sds"

tweet = re.sub('\:\)',' happy ',tweet)
tweet = re.sub('\:\(',' sad ',tweet)


print tweet


# tweet = """
# Cands/Mods: Trump, you can't run independent! We'll lose!
# @realDonaldTrump hates strong, independent, women. I'm shocked he doesn't tell his daughter to put a dick in her mouth &amp; hush #GOPDebate
# @kimguilfoyle It wasn't debate.It was an effort to bring @realDonaldTrump down. I see why he would not pledge #GOPDebate shame on @FoxNews
# RT @SaintGrimlock: Trump was thinking message from God, I haven't told these other guys anything....#GOPDebates
# """



#print tweet
#tweet = tweet.lower().replace("'", "")


#tweet = re.sub('(?<=not)+ (\w+)','$1 ~$2',tweet)

# words="never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint|wasnt".split('|')
#
#
# exclude = set(string.punctuation)
# tweet = ''.join(ch for ch in tweet if ch not in exclude)
#
# # for word in words:
# #     tweet = tweet.replace('{} '.format(word), '{} not_'.format(word))
#
# print tweet
 #
 # # exclude = set(string.punctuation)
 #    # tweet = ''.join(ch for ch in tweet if ch not in exclude)
 #
 #    negation_list ="never|none|not|hasnt|havnt|wasnt".split('|')
 #
 #    #tweet = tweet.replace("'", "")
 #
 #    # for word in negation_list:
 #    #     tweet = tweet.replace('{} '.format(word), '{} not_'.format(word))
