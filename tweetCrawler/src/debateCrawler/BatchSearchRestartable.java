package debateCrawler;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Properties;

// Argument examples:
//  "GOPDebate AND Trump" 230 0 703380244661538816 TrumpFolder\TrumpFile
//  Query = GOPDebate AND Trump
//  max # tweets to store = 230
//  since id = 0, max id = 703380244661538816
//  Output folder TrumpFolder, output file prefix = TrumpFile (TrumpFolder must exist)

//  "GOPDebate AND Trump" 230 TrumpDebate Trump 
//  Query = GOPDebate AND Trump
//  max # tweets to store = 230
//  since id and max id will be either the previously cached values in TrumpDebate_CachedBounds*.txt
//    OR if that file doesn't exist, then they are set to 0 and -1 (= most recent tweets)
//  since id = 0, max id = -1 (= start at most recent tweets)
//  Output file prefix = TrumpDebate (in current dir)
//  Tweets will be filtered to ensure the text contains Trump (the last parameter)






// taken from 
// http://www.socialseer.com/twitter-programming-in-java-with-twitter4j/how-to-retrieve-more-than-100-tweets-with-the-twitter-api-and-twitter4j/
/**
 * 	Demonstration of how to retrieve more than 100 tweets using the Twitter API.
 *
 * 	It is based upon the Application Authentication example, and therefore uses application
 * 	authentication.  It does not matter that much which type of authentication you use, although
 * 	it will effect your rate limits.
 *
 * 	You will note that this code has only the bare minimum of error handling.  A real production application
 * 	would have a lot more code in it to catch, diagnose, and recover from errors at all points of interaction
 * 	with Twitter.
 *
 * 	@author	Charles McGuinness
 *  @author Doyle Ravnaas, Yogendra Miraje, Ran Qaio
 * 
 */
import twitter4j.Query;
import twitter4j.Query.ResultType;
import twitter4j.QueryResult;
import twitter4j.RateLimitStatus;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.auth.OAuth2Token;
import twitter4j.conf.ConfigurationBuilder;

public class BatchSearchRestartable
{
    // Feb 25th: "GOPDebate  AND Trump" 5200 0 703021659893080000 TrumpBefore225
    //           "GOPDebate AND Trump" 5200 703082058395033000 703097157159514000 TrumpAfter225
    // (replace candidate last name and #maxtweets
    
    // set via config\twitter4j.properties (see example file)
    private static String consumerKey = "--your key goes here--";
    private static String consumerSecret = "--your secret goes here--";

    // How many tweets to retrieve in every call to Twitter. 100 is the maximum allowed in the API
    private static final int TWEETS_PER_QUERY = 100;

    // This controls how many queries, maximum, we will make of Twitter before cutting off the
    // results. You will retrieve up to MAX_QUERIES*TWEETS_PER_QUERY tweets.
    //
    // If you set MAX_QUERIES high enough (e.g., over 450), you will undoubtedly hit your rate
    // limits and you an see the program sleep until the rate limits reset
    private static int MAX_QUERIES = 5;

    // Threshold at which we rollover, storing tweets to a new output file.
    private static int BATCH_STORE_THRESHOLD = 2000;

    // What we want to search for in this program. Justin Bieber always returns as many results as
    // you could ever want, so it's safe to assume we'll get multiple pages back...
    private static String queryString = "GOPDebate AND Trump";

    private static int maxTweetsOverall = TWEETS_PER_QUERY * MAX_QUERIES;
    
    // We will ignore tweets from user ids that have a candidate name in them, or "news" in them
    private static List<String> ignoreTweetBasedOnUserIds = new ArrayList<String>(Arrays.asList("trump", "cruz", "rubio", "hillary", "bernie", "carson", "kasich", "news"));
    private static List<String> ignoredIds = new ArrayList<String>();
    private static long ignoredCountDueToId = 0L;
    private static long ignoredCountDueToText = 0L;
    
    // Optional tag that requires a certain string in the text.  Typically a candidate name.
    private static String textMustContain = null;
    
    private static String getBaseFileName(String prefix, int tweetCount)
    {

        // Get current timestamp
        String timestamp = new SimpleDateFormat("yyyyMMddhhmm").format(new Date());

        // Build output file name
        // TODO: add this, and make Ran's call this (move into that file?)
        String fileName = prefix + "_" + timestamp + "_" + tweetCount + ".json";
        // System.out.println("Output file = " + fileName);

        return fileName;
    }

    /**
     * Retrieve the "bearer" token from Twitter in order to make application-authenticated calls.
     *
     * This is the first step in doing application authentication, as described in Twitter's
     * documentation at https://dev.twitter.com/docs/auth/application-only-auth
     *
     * Note that if there's an error in this process, we just print a message and quit. That's a
     * pretty dramatic side effect, and a better implementation would pass an error back up the
     * line...
     *
     * @return The oAuth2 bearer token
     * @throws IOException
     */
    public static OAuth2Token getOAuth2Token() throws IOException
    {
        // Look for config file using relative dir - in tweetCrawler/config folder
        File file = new File("config/twitter4j.properties");
        System.out.println("Reading config properties from:");
        System.out.println(file.getAbsolutePath());
        if (!file.exists())
        {
            System.out.println("No configuration file found");
            System.exit(-1);
        }

        Properties prop = new Properties();
        InputStream is = null;

        is = new FileInputStream(file);
        prop.load(is);

        ConfigurationBuilder cb = new ConfigurationBuilder();

        // Set our key and secret according to our properties file
        consumerKey = prop.getProperty("oauth.consumerKey");
        consumerSecret = prop.getProperty("oauth.consumerSecret");

        OAuth2Token token = null;
        cb.setApplicationOnlyAuthEnabled(true);
        cb.setOAuthConsumerKey(consumerKey);
        cb.setOAuthConsumerSecret(consumerSecret);
        cb.setJSONStoreEnabled(true);

        try
        {
            token = new TwitterFactory(cb.build()).getInstance().getOAuth2Token();
        }
        catch (Exception e)
        {
            System.out.println("Could not get OAuth2 token");
            e.printStackTrace();
            System.exit(0);
        }

        return token;
    }

    /**
     * Get a fully application-authenticated Twitter object useful for making subsequent calls.
     *
     * @return Twitter4J Twitter object that's ready for API calls
     * @throws IOException
     */
    public static Twitter getTwitter() throws IOException
    {
        OAuth2Token token;

        // First step, get a "bearer" token that can be used for our requests
        token = getOAuth2Token();

        // Now, configure our new Twitter object to use application authentication and provide it
        // with our CONSUMER key and secret and the bearer token we got back from Twitter
        ConfigurationBuilder cb = new ConfigurationBuilder();

        cb.setApplicationOnlyAuthEnabled(true);

        cb.setOAuthConsumerKey(consumerKey);
        cb.setOAuthConsumerSecret(consumerSecret);

        cb.setOAuth2TokenType(token.getTokenType());
        cb.setOAuth2AccessToken(token.getAccessToken());
        cb.setJSONStoreEnabled(true);

        // And create the Twitter object!
        return new TwitterFactory(cb.build()).getInstance();
    }

    public static void main(String[] args) throws Exception
    {
        TweetBounds tb = null;
        String prefix = "";

        // args: [query [maxtweets [sinceid maxid] folder\filePrefix [candidateName]]
        if (args.length >= 1)
        {
            queryString = args[0];
        }

        // Default csv file prefix (if none input as an arg)
        prefix = "queryHash" + queryString.hashCode();

        // Set some parameters about how many tweets we will grab and how far back.
        if (args.length >= 2)
        {
            // Adjust the # queries for the number of max tweets we want (there could be less)
            maxTweetsOverall = Integer.parseInt(args[1]);
            MAX_QUERIES = maxTweetsOverall / TWEETS_PER_QUERY;
            if (MAX_QUERIES * TWEETS_PER_QUERY < maxTweetsOverall)
            {
                // Add one if not an even divisor
                MAX_QUERIES++;
            }

            // File prefix can be 3rd parameter or 5th (if since and max specified)
            int prefixArg = 0;
            if (args.length >= 5)
            {
                prefixArg = 4;
            } else 
            {
                prefixArg = 2;
            }
            if (prefixArg != 0)
            {
                prefix = args[prefixArg];
                if (prefixArg+1 < args.length) 
                {
                    textMustContain = args[prefixArg+1].toLowerCase();                   
                }
            }

            // If since and max are on the command line, use those
            if (prefixArg >= 4)
            {
                tb = new TweetBounds(Long.parseLong(args[2]), Long.parseLong(args[3]), queryString, prefix);

                System.out.println("Using input since/max values for the query");
            }
        }

        // If we didn't get a since/max value, get any cached values, if present.
        // Will default to 0, -1 if none cached.
        if (tb == null)
        {
            tb = new TweetBounds(queryString, prefix);
            tb.initTweetBounds();
        }

        System.out.println("Running queries with:");
        System.out.println("  Query:                 " + queryString);
        System.out.println("  Overall max # tweets:  " + maxTweetsOverall);
        System.out.println("  Starting tweet id:     " + tb.sinceID);
        System.out.println("  Max tweet id:          " + tb.maxID);
        System.out.println("  Output file prefix:    " + prefix);
        if (textMustContain != null)
        {
            System.out.println("  Mandatory term in tweet text (otherwise tweet discarded):    " + textMustContain);
            
        }

        System.out.println("  Ignoring tweets from user ids containing: " + ignoreTweetBasedOnUserIds);

        System.out.println();

        // We're curious how many tweets, in total, we've retrieved. Note that TWEETS_PER_QUERY is
        // an upper limit, but Twitter can and often will retrieve far fewer tweets
        int totalTweets = 0;
        int currentBatchSaved = 0;

        // Note:
        // The maxId variable is the key to our retrieving multiple blocks of tweets. In each batch
        // of tweets we retrieve, we use this variable to remember the LOWEST tweet ID. Tweet IDs
        // are (java) longs, and they are roughly sequential over time. Without setting the MaxId in
        // the query, Twitter will always retrieve the most recent tweets. Thus, to retrieve a
        // second (or third or ...) batch of Tweets, we need to set the Max Id in the query to be
        // one less than the lowest Tweet ID we've seen already. This allows us to page backwards
        // through time to retrieve additional blocks of tweets.

        Twitter twitter = getTwitter();

        String fileBase = getBaseFileName(prefix, totalTweets);

        int queryAttempts = 0;

        // Now do a simple search to show that the tokens work
        try
        {
            // There are limits on how fast you can make API calls to Twitter, and if you have hit
            // your limit and continue to make calls Twitter will get annoyed with you. I've found
            // that going past your limits now and then doesn't seem to be problematic, but if you
            // have a program that keeps banging the API when you're not allowed you will eventually
            // get shut down.
            //
            // Thus, the proper thing to do is always check your limits BEFORE making a call, and if
            // you have hit your limits sleeping until you are allowed to make calls again.
            //
            // Every time you call the Twitter API, it tells you how many calls you have left, so
            // you don't have to ask about the next call. But before the first call, we need to find
            // out whether we're already at our limit.

            // This returns all the various rate limits in effect for us with the Twitter API
            Map<String, RateLimitStatus> rateLimitStatus = twitter.getRateLimitStatus("search");

            // This finds the rate limit specifically for doing the search API call we use in this
            // program
            RateLimitStatus searchTweetsRateLimit = rateLimitStatus.get("/search/tweets");

            // Always nice to see these things when debugging code...
            System.out.printf("You have %d calls remaining out of %d, Limit resets in %d seconds\n",
                    searchTweetsRateLimit.getRemaining(), searchTweetsRateLimit.getLimit(),
                    searchTweetsRateLimit.getSecondsUntilReset());

            // This is the loop that retrieve multiple blocks of tweets from Twitter
            //for (int queryNumber = 0; queryNumber < MAX_QUERIES; queryNumber++)
            int queryNumber = 0;

            while (totalTweets < maxTweetsOverall)
            {                
                System.out.printf("\n\n!!! Starting loop %d, sinceId = %d, maxId= %d\n\n", queryNumber, tb.sinceID,
                        tb.maxID);

                
                // Do we need to delay because we've already hit our rate limits?
                 if (searchTweetsRateLimit == null)
                {
                    // Every once in a while, this is null, even though it should get
                    // set on every iteration.
                    searchTweetsRateLimit = rateLimitStatus.get("/search/tweets");
                }
                
                if (searchTweetsRateLimit.getRemaining() == 0)
                {
                    // Twitter has an annoying habit of sometimes returning negative numbers
                    // and needing a bit of buffer beyond what they claim is the wait time.
                    int secondsToWait = 10;
                    if (searchTweetsRateLimit.getSecondsUntilReset() > 0)
                    {
                        secondsToWait = searchTweetsRateLimit.getSecondsUntilReset() + secondsToWait; 
                    }
                    
                    // Yes we do, unfortunately ...
                    System.out.printf("!!! Sleeping for %d seconds due to rate limits\n",
                            secondsToWait);


                    // If you sleep exactly the number of seconds, you can make your query a bit too
                    // early and still get an error for exceeding rate limitations
                    //
                    // Adding two seconds seems to do the trick. Sadly, even just adding one second
                    // still triggers a rate limit exception more often than not. I have no idea
                    // why, and I know from a Comp Sci standpoint this is really bad, but just add
                    // in 2 seconds and go about your business. Or else.
                    // Update from Doyle - I'm trying +10, 2 didn't work for me.
                    Thread.sleep(secondsToWait * 1000L);  
                    
                    System.out.println();

                }

                Query q = new Query(queryString); // Search for tweets that contains this term
                q.setCount(TWEETS_PER_QUERY); // How many tweets, max, to retrieve
                q.setResultType(ResultType.recent); // Get all tweets
                q.setLang("en"); // English language tweets, please
                q.setSinceId(0);
                q.setMaxId(-1);

                // If maxID is -1, then this is our first call and we do not want to tell Twitter
                // what the maximum tweet id is we want to retrieve. But if it is not -1, then it
                // represents the lowest tweet ID we've seen, so we want to start at it-1 (if we
                // start at maxID, we would see the lowest tweet a second time...
                if (tb.maxID != -1)
                {
                    q.setMaxId(tb.maxID - 1);
                }

                // Use sinceID to say "don't back up beyond this id"
                if (tb.sinceID != 0)
                {
                    q.setSinceId(tb.sinceID);
                }

                // This actually does the search on Twitter and makes the call across the network
                QueryResult r = null;

                try {
                    queryAttempts++;

                    // This actually does the search on Twitter and makes the
                    // call across the network
                    r = twitter.search(q);
                    
                    // No exception = reset retry counter.
                    queryAttempts = 0;
                    queryNumber++;
                    
                } catch (TwitterException e) {
                    
                    // Decrement number of successful queries - this one didn't work.
                    queryNumber--;
                    
                    System.out.println("Warning - exception querying Twitter");
                    
                    if (e.exceededRateLimitation()) {
                        // Get that rate limit again?
                        // searchTweetsRateLimit =
                        // rateLimitStatus.get("/search/tweets");
                        searchTweetsRateLimit = e.getRateLimitStatus();
                        
                        System.out.println("Rate limit - will sleep and retry query");
                       
                    } else if (e.isCausedByNetworkIssue()) {                        
                        
                        System.out.println("Network issue, will retry... Exception = " + e.toString());
                        
                    }else {
                    
                        // Are there other exceptions we could retry? 
                        throw e;
                    }
                }

                
                if (r == null)
                {
                    if (queryAttempts < 5) 
                    {
                        // Likely a rate limit retry or a network issue
                        continue;
                    }
                    else {
                        // Break out of loop and save what info we got.
                        
                        System.out.println("Failed after 5 retries, done!");
                        break;
                    }
                }

                // If there are NO tweets in the result set, it is Twitter's way of telling us that
                // there are no more tweets to be retrieved. Remember that Twitter's search index
                // only contains about a week's worth of tweets, and uncommon search terms can run
                // out of week before they run out of tweets
                if (r.getTweets().size() == 0)
                {
                    break; // Nothing? We must be done
                }
                
                // As part of what gets returned from Twitter when we make the search API call, we
                // get an updated status on rate limits. We save this now so at the top of the loop
                // we can decide whether we need to sleep or not before making the next call.
                searchTweetsRateLimit = r.getRateLimitStatus();

                // Loop through returned tweets in this batch to
                // get the high and low tweetid value (and set our next max)

                long lowestIdInBatch = -1; // these are compared as unsigned longs, so -1 is max.
                long highestIdInBatch = 0;
                List<Status> filteredList = new ArrayList<Status>();
                for (Status s : r.getTweets()) // Loop through all the tweets...
                {
                    // Keep track of the lowest tweet ID. If you do not do this, you cannot retrieve
                    // multiple blocks of tweets...
                    if (Long.compareUnsigned(s.getId(), tb.maxID) < 0)
                    {
                        // this is our max id for the next batch = lower than
                        // any in this batch since we are going back in time.
                        tb.maxID = s.getId();
                    }                   
                    

                    if (Long.compareUnsigned(highestIdInBatch, s.getId()) < 0)
                    {
                        highestIdInBatch = s.getId();
                    }

                    if (Long.compareUnsigned(lowestIdInBatch, s.getId()) > 0)
                    {
                        lowestIdInBatch = s.getId();
                    }
                    
                    boolean ignore = false;
                    
                    // Skip tweets that look like they are from candidates or news                    
                    String lowerUserId = s.getUser().getScreenName().toLowerCase();
                    
                    for (String ignoreWithSubString : ignoreTweetBasedOnUserIds)
                    {
                        // If we find our candidate last names or "news" in the user id...
                        if (lowerUserId.indexOf(ignoreWithSubString) != -1)
                        {
                            ignore = true;
                            if (!ignoredIds.contains(lowerUserId))
                            {
                                ignoredIds.add(lowerUserId);
                            }                            

                            ignoredCountDueToId++;
                            break;
                        }
                    }
 
                    
                    // If we have a "must contain" argument specified, check
                    // that the text contains it.
                    if (ignore == false && textMustContain != null && !textMustContain.isEmpty())
                    {
                        if(!s.getText().toLowerCase().contains(textMustContain))
                        {
                            ignore = true;
                            ignoredCountDueToText++;
                        }
                    }
                    
                    if (!ignore) {
                        filteredList.add(s);
                    }
                }

                // Now store our batch of tweets
                System.out.println("  Batch tweet ids range = " + lowestIdInBatch + " to " + highestIdInBatch);
                System.out.println("  Number of ignored tweets = " + (r.getTweets().size() - filteredList.size()));

                if ((currentBatchSaved == 0) || (currentBatchSaved >= BATCH_STORE_THRESHOLD))
                {
                    // Rollover to new file
                    fileBase = getBaseFileName(prefix, totalTweets);
                    currentBatchSaved = 0;

                    // Overwrite any current file, do not append.
                    SaveTweets.storeQueryResult(fileBase, filteredList, r.getQuery(), false);

                } else
                {
                    // Append to current file for this batch.
                    SaveTweets.storeQueryResult(fileBase, filteredList, r.getQuery(), true);
                }

                // Keep track of how many we've processed.
                totalTweets = totalTweets + filteredList.size();
                currentBatchSaved = currentBatchSaved + filteredList.size();

                // Once we have successfully stored our tweets, update our
                // saved boundaries (just in case we crash and need to restart)
                // (We save some debug info too, but this isn't used for next run)
                tb.saveTweetBounds("Tweets stored: " + totalTweets + ", requested min = " + maxTweetsOverall 
                        + "; ignored user ids: " + ignoredIds.toString());
            }
        }
        catch (Exception e)
        {

            System.out.println("Exception occured - terminating crawl unexpectedly!");

            e.printStackTrace();
        }


        System.out.println("\n\nTweet search bounds of " + tb.sinceID + " and " + tb.maxID + " have been saved for this query (see *CachedBounds*.txt)");
        System.out.printf("\n\nNumber tweets requested = %d", maxTweetsOverall);

        System.out.printf("\n\nA total of %d tweets stored", totalTweets);
        System.out.printf("\n\nNumber tweets ignored due to user id: %d", ignoredCountDueToId);
        if (ignoredIds.size() != 0)
        {
            System.out.printf("\n  User ids ignored: " + ignoredIds);
        }
        System.out.printf("\n\nNumber tweets ignored due to no hit on specified mandatory text: %d", ignoredCountDueToText);
        

    }
}
