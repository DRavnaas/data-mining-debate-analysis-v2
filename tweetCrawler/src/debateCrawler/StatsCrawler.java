package debateCrawler;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

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

public class StatsCrawler {

    // taken from
    // http://www.socialseer.com/twitter-programming-in-java-with-twitter4j/how-to-retrieve-more-than-100-tweets-with-the-twitter-api-and-twitter4j/
    /**
     * Demonstration of how to retrieve more than 100 tweets using the Twitter
     * API.
     *
     * It is based upon the Application Authentication example, and therefore
     * uses application authentication. It does not matter that much which type
     * of authentication you use, although it will effect your rate limits.
     *
     * You will note that this code has only the bare minimum of error handling.
     * A real production application would have a lot more code in it to catch,
     * diagnose, and recover from errors at all points of interaction with
     * Twitter.
     *
     * @author Charles McGuinness
     * @author Doyle Ravnaas, Yogendra Miraje, Ran Qaio
     * 
     */

    // set via config\twitter4j.properties (see example file)
    private static String consumerKey = "--your key goes here--";
    private static String consumerSecret = "--your secret goes here--";

    // How many tweets to retrieve in every call to Twitter. 100 is the maximum
    // allowed in the API
    private static final int TWEETS_PER_QUERY = 100;

    // What we want to search for in this program. Justin Bieber always returns
    // as many results as
    // you could ever want, so it's safe to assume we'll get multiple pages
    // back...
    private static String queryString = "GOPDebate";

    /**
     * Retrieve the "bearer" token from Twitter in order to make
     * application-authenticated calls.
     *
     * This is the first step in doing application authentication, as described
     * in Twitter's documentation at
     * https://dev.twitter.com/docs/auth/application-only-auth
     *
     * Note that if there's an error in this process, we just print a message
     * and quit. That's a pretty dramatic side effect, and a better
     * implementation would pass an error back up the line...
     *
     * @return The oAuth2 bearer token
     * @throws IOException
     */
    public static OAuth2Token getOAuth2Token() throws IOException {
        // Look for config file using relative dir - in tweetCrawler/config
        // folder
        File file = new File("config/twitter4j.properties");
        System.out.println("Reading config properties from:");
        System.out.println(file.getAbsolutePath());
        if (!file.exists()) {
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

        try {
            token = new TwitterFactory(cb.build()).getInstance().getOAuth2Token();
        } catch (Exception e) {
            System.out.println("Could not get OAuth2 token");
            e.printStackTrace();
            System.exit(0);
        }

        return token;
    }

    /**
     * Get a fully application-authenticated Twitter object useful for making
     * subsequent calls.
     *
     * @return Twitter4J Twitter object that's ready for API calls
     * @throws IOException
     */
    public static Twitter getTwitter() throws IOException {
        OAuth2Token token;

        // First step, get a "bearer" token that can be used for our requests
        token = getOAuth2Token();

        // Now, configure our new Twitter object to use application
        // authentication and provide it
        // with our CONSUMER key and secret and the bearer token we got back
        // from Twitter
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

    public static void main(String[] args) throws Exception {

        // args: sinceDate fromDate [query [maxId]]
        // >= since, < until
        // Typically, set min to the day before the debate, and max to two or three days
        // after.  These dates are for the Feb 25th debate
        // NOTE:  You have to go 5 hours ahead to catch a full day (due to GMT to EST conversion)
        // so if you need times past 7pm or before 5am, adjust your dates accordingly
        // ie: to capture >= 2/25 midnight est, sinceDate needs to be 2/24
        //     to capture < 2/26 11:59pm est, maxDate would need to be 2/27        
        String minDate = "2016-02-24", maxDate = "2016-02-28";
        //String minDate = "2016-02-25", maxDate = "2016-02-28";

        // MaxId is used to go backwards in time from the most recent tweets (on
        // date < maxDate)
        long maxID = -1;
        long totalTweets = 0;
        
        // Set some parameters about how many tweets we will grab and how far
        // back.
        if (args.length >= 2) {
            minDate = args[0];
            maxDate = args[1];

            if (args.length >= 3) {
                queryString = args[2];
            }
            
            // this allows starting in the middle of a day
            if (args.length >=4) {
                maxID = Long.parseLong(args[3]);
            }
        }

        System.out.println("Running stats for query:");
        System.out.println("  Query:                 " + queryString);
        System.out.println("  Min date (YYYY-MM-DD):              " + minDate);
        System.out.println("  Max date (YYYY-MM-DD):              " + maxDate);
 

        Twitter twitter = getTwitter();

        Map<String, Long[]> countsPerHour = new HashMap<String, Long[]>();
        // Three values per hour = count, minid, max id
        int countIndex = 0;
        int minIdIndex = 1;
        int maxIdIndex = 2;
        boolean done = false;
        int queryAttempts = 0;

        try {

            Map<String, RateLimitStatus> rateLimitStatus = twitter.getRateLimitStatus("search");

            // This finds the rate limit specifically for doing the search API
            // call we use in this program
            RateLimitStatus searchTweetsRateLimit = rateLimitStatus.get("/search/tweets");

            // Always nice to see these things when debugging code...
            System.out.printf(
                    "You have %d calls remaining out of %d, Limit resets in %d seconds\n",
                    searchTweetsRateLimit.getRemaining(), searchTweetsRateLimit.getLimit(),
                    searchTweetsRateLimit.getSecondsUntilReset());

            int queryNumber = 0;
            // This is the loop that retrieve multiple blocks of tweets from
            // Twitter
            do {
                System.out.printf("\n\nTwitter query iteration # %d\n\n", queryNumber);

                // Do we need to delay because we've already hit our rate
                // limits?
                if (searchTweetsRateLimit == null)
                {
                    // Every once in a while, this is null, even though it should get
                    // set on every iteration.
                    searchTweetsRateLimit = rateLimitStatus.get("/search/tweets");
                    if (searchTweetsRateLimit == null)
                    {
                        System.out.println("Null rate limit, terminating early");
                        break;
                    }
                }
                
                
                if (searchTweetsRateLimit.getRemaining() == 0) {
                    
                    // Twitter occasionally lies on how much time to wait, and 
                    // even has negative numbers sometimes.  We do 10 seconds minimum.
                    int secondsToWait = 10;
                    if (searchTweetsRateLimit.getSecondsUntilReset() > 0)
                    {
                        secondsToWait = searchTweetsRateLimit.getSecondsUntilReset() + secondsToWait; 
                    }
                    
                    // Yes we do, unfortunately ...
                    System.out.printf("!!! Sleeping for %d seconds due to rate limits\n",
                            secondsToWait);

                    Thread.sleep(secondsToWait * 1000L);
                    System.out.println();
                }

                Query q = new Query(queryString); // Search for tweets that
                                                  // contains this term
                q.setCount(TWEETS_PER_QUERY); // How many tweets, max, to
                                              // retrieve
                q.setResultType(ResultType.recent); // Get all tweets
                q.setLang("en"); // English language tweets, please

                if (maxID != -1) {
                    q.setMaxId(maxID - 1);
                }

                // These might be GMT dates? = +5 from EST and + 8 from PST
                q.setSince(minDate); // >= this date
                q.setUntil(maxDate); // < this date

                QueryResult r = null;

                try {
                    queryAttempts++;

                    // This actually does the search on Twitter and makes the
                    // call across the network
                    r = twitter.search(q);
                    
                    // No exception = reset retry counter.
                    queryAttempts = 0;
                    
                    // Increment number of successful queries
                    queryNumber++;
                    
                } catch (TwitterException e) {
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
                    
                        // TODO: Are there other exceptions we could retry? 
                        // Set done = true, we'll write the file and finish up. 
                        //throw e;
                        done = true;
                    }
                }

                
                // We we were not able to get any tweets - should we retry or stop?
                if (r == null)
                {
                    if (queryAttempts < 5) 
                    {
                        // Could be rate limit or network issue we will retry
                        // Also could be a Twitter exception we won't retry (done == true)
                        continue;
                    }
                    else {
                        // Break out of loop and save what info we got.
                        
                        System.out.println("Failed after 5 retries, done!");
                        break;
                    }
                }

                if (r.getTweets().size() == 0) {
                    break; // Nothing? We must be done
                }

                // As part of what gets returned from Twitter when we make the
                // search API call, we
                // get an updated status on rate limits. We save this now so at
                // the top of the loop
                // we can decide whether we need to sleep or not before making
                // the next call.
                searchTweetsRateLimit = r.getRateLimitStatus();

                // Loop through returned tweets in this batch to
                // get the high and low tweetid value (and set our next max)
                // Stop when we hit our minDate

                int numTweetsThisBatch = r.getTweets().size();

                long lowestIdInBatch = -1; // these are compared as unsigned
                                           // longs, so -1 is max.
                long highestIdInBatch = 0;

                Date lowTimestampInBatch = new Date();

                for (Status s : r.getTweets()) // Loop through all the tweets...
                {

                    // Add this tweet to the statistics we are gathering
                    long tweetId = s.getId();
                    Date timestamp = s.getCreatedAt();

                    // Note - be sure to format in 24 hour time (otherwise 1am =
                    // 1pm in this key format)
                    // I believe this all gets formatted into EST, 5 hours behind UTC
                    // TODO: should we adjust the "since" logic for that?                    

                    SimpleDateFormat keyFormat = new SimpleDateFormat("yyyy-MM-dd HH"); 
                    String hourKey = keyFormat.format(timestamp);
                    String justDate = new SimpleDateFormat("yyyy-MM-dd").format(timestamp);

                    if (justDate.compareTo(minDate) < 0) {
                        // We have dipped below our minimum date
                        // We will continue to process tweets in this list
                        // (not sure if they are always sorted by date & time in
                        // results)
                        // so keep looping in this batch, but once we are done
                        // don't query twitter for another batch.
                        done = true;
                    } else {

                        if (lowTimestampInBatch.after(timestamp)) {
                            lowTimestampInBatch = timestamp;
                        }

                        if (countsPerHour.containsKey(hourKey)) {
                            Long[] statsForHour = countsPerHour.get(hourKey);
                            statsForHour[countIndex] = statsForHour[countIndex] + 1L;

                            if (Long.compareUnsigned(statsForHour[maxIdIndex], tweetId) < 0) {
                                // New max id
                                statsForHour[maxIdIndex] = tweetId;
                            } else if (Long.compareUnsigned(statsForHour[minIdIndex], tweetId) > 0) {
                                // New min id
                                statsForHour[minIdIndex] = tweetId;
                            }
                        } else {
                            Long[] statsForHour = new Long[3];
                            statsForHour[countIndex] = 1L;
                            statsForHour[minIdIndex] = tweetId;
                            statsForHour[maxIdIndex] = tweetId;
                            countsPerHour.put(hourKey, statsForHour);
                        }

                        // Keep track of the lowest tweet ID. If you do not do
                        // this,
                        // you cannot retrieve
                        // multiple blocks of tweets...
                        if (Long.compareUnsigned(s.getId(), maxID) < 0) {
                            // this is our max id for the next batch = lower
                            // than
                            // any in this batch since we are going back in
                            // time.
                            maxID = s.getId();
                        }

                        if (Long.compareUnsigned(highestIdInBatch, s.getId()) < 0) {
                            highestIdInBatch = s.getId();
                        }

                        if (Long.compareUnsigned(lowestIdInBatch, s.getId()) > 0) {
                            lowestIdInBatch = s.getId();
                        }
                    }

                }

                System.out.println("  Batch tweet ids range = " + lowestIdInBatch + " to "
                        + highestIdInBatch + ", number tweets = " + numTweetsThisBatch
                        + ", oldest timestamp = " + lowTimestampInBatch);

                // Keep track of how many we've processed.
                totalTweets = totalTweets + r.getTweets().size();

                // As part of what gets returned from Twitter when we make the
                // search API call, we
                // get an updated status on rate limits. We save this now so at
                // the top of the loop
                // we can decide whether we need to sleep or not before making
                // the next call.
                searchTweetsRateLimit = r.getRateLimitStatus();

            } while (!done);

            System.out.println("Date before min date reached, stopping.");

            // write stats to file

            SortedSet<String> keys = new TreeSet<String>(countsPerHour.keySet());

            BufferedWriter wr = null;
            try {
                String fileName = queryString + "-Stats" + minDate + "to" + maxDate + ".csv";
                wr = new BufferedWriter(new FileWriter(fileName));

                System.out.println("Writing stats to " + fileName);
                String firstLine = "Date and hour (UTC = EST + 5), count, minid, maxid";

                // Note: this overwrites the file, if the file exists
                wr.write(firstLine);

                wr.newLine();

                for (String hourKey : keys) {
                    Long[] statsForHour = countsPerHour.get(hourKey);

                    wr.write(hourKey + "," + statsForHour[countIndex] + ","
                            + statsForHour[minIdIndex] + "," + statsForHour[maxIdIndex]);
                    wr.newLine();
                }

            } finally {
                if (wr != null) {
                    wr.flush();
                    wr.close();
                }
            }

        } catch (Exception e) {
            // Catch all -- you're going to read the stack trace and figure out
            // what needs to be done to fix it
            System.out.println("That didn't work well...wonder why?");

            e.printStackTrace();
        }

        System.out.printf("\n\nA total of %d tweets retrieved\n", totalTweets);
        // That's all, folks!

    }

}
