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
        
        // Dates for March 3rd debate  - day of debate (= before) and day after (< 3/5)
        // debate started at 9pm and ended at 11pm ET
        String minDate = "2016-03-10", maxDate = "2016-03-12";

        // MaxId is used to go backwards in time from the most recent tweets (on
        // date < maxDate)
        // NOTE: maxId and sinceID will override the dates
        long maxID = -1, sinceID = 0L;
        long totalTweets = 0;
        
        // working day before the debate (Feb)
        //2016-02-25 10   1011    702870674151501000      702885762119966000
        //2016-02-25 11   1292    702885770839859000      702900863388557000
        //2016-02-25 12   1231    702900883470753000      702915958973861000
        //2016-02-25 13   1524    702915965374349000      702931059294539000
        //2016-02-25 14   1562    702931068744118000      702946161158557000
        //2016-02-25 15   2035    702946165931778000      702961255812825000
        //2016-02-25 16   2467    702961262573985000      702976353386291000
        //2016-02-25 17   2846    702976365746782000      702991459579006000
        //2016-02-25 18   4390    702991461453860000      703006560469954000
        //2016-02-25 19   7011    703006561954877000      703021659893080000
        
        // Next two hours after debate:
        //2016-02-25 23   113645  703066958862938000      703082057979838000
        //2016-02-26 00   41475   703082058395033000      703097157159514000
        
        // You can hardcode a since to max range here - these override any dates
        
        //sinceID = 702712234468909000L; // over 24 hours before the debate
        //maxID = 703445853063782400L; // 24 hours after debate
        
        //sinceID = 703066958862938000L; // end of the debate
        //maxID = 703097157159514000L;  // two hours after debate
        
        // 10am to <8pm, day of the debate
        //sinceID = 702870674151501000L;
        //maxID = 703021659893080000L;
        
        //range to get stats = 3/3 at 0:00 to 3/4 at 6 pm (ish)
        // since = 705256427263234048, max = 705905661146701825
        
        //sinceID = 705256427263234048L;
        //maxID = 705905661146701825L;
        
        // Set some parameters about how many tweets we will grab and how far
        // back.
        if (args.length >= 1) {

            queryString = args[0];

            if (args.length >= 3) {
                
                minDate = args[1];
                maxDate = args[2];                       
            }
            
            // this allows starting in the middle of a day
            if (args.length >=4) {
                maxID = Long.parseLong(args[3]);
            }
        }

        System.out.println("Running stats for query:");
        System.out.println("  Query:                 " + queryString);

        System.out.println("  Min id:                             " + sinceID);
        System.out.println("  Max id:                             " + maxID);

        if ((sinceID == 0L && maxID == -1L))
        {
            System.out.println("  Min date (YYYY-MM-DD):              " + minDate);
            System.out.println("  Max date (YYYY-MM-DD):              " + maxDate);
        }

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

                // These might be GMT dates? = +5 from EST and + 8 from PST
                //q.setSince(minDate); // >= this date  - we will do this via the twitter timestamps

                
                q.setSinceId(sinceID);
                q.setMaxId(-1);
                if (maxID != -1) {
                    q.setMaxId(maxID - 1);
                } else {
                    // We only need to set this if we aren't using maxid as our boundary
                    q.setUntil(maxDate); // < this date
                }

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
                        System.out.println("Unexpected exception, will not retry... Exception = " + e.toString());

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

                // As part of what gets returned from Twitter when we make the
                // search API call, we
                // get an updated status on rate limits. We save this now so at
                // the top of the loop
                // we can decide whether we need to sleep or not before making
                // the next call.
                searchTweetsRateLimit = r.getRateLimitStatus();
                
                if (r.getTweets().size() == 0) {
                    System.out.println("(zero tweets found for this batch)");
                    
                    // Not sure what to do here?
                    break;
                } 

                // Process this batch of tweets....
                
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
                    // THESE ARE ALL EST Times!
                    SimpleDateFormat keyFormat = new SimpleDateFormat("yyyy-MM-dd HH"); 
                    String hourKey = keyFormat.format(timestamp);
                    String justDate = new SimpleDateFormat("yyyy-MM-dd").format(timestamp);

                    if (sinceID == 0 && (justDate.compareTo(minDate) < 0)) {
                        // We have dipped below our minimum date
                        // We will continue to process tweets in this list
                        // (not sure if they are always sorted by date & time in
                        // results)
                        // so keep looping in this batch, but once we are done
                        // don't query twitter for another batch.
                        System.out.println("Tweet time " + timestamp + " less than minDate " + minDate);
                        done = true;
                        continue;
                        
                    } else if (Long.compareUnsigned(sinceID, s.getId()) > 0)
                    {                        
                        System.out.println("Tweet id " + s.getId() + " less than sinceID " + sinceID);
                        done = true;
                        continue;
                        
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

            } while (!done);
 
            // write stats to file

            SortedSet<String> keys = new TreeSet<String>(countsPerHour.keySet());

            BufferedWriter wr = null;
            try {
                String fileName = queryString + "-Stats" + minDate + "to" + maxDate + ".csv";
                wr = new BufferedWriter(new FileWriter(fileName));

                System.out.println("Writing stats to " + fileName);
                String firstLine = "Date and hour EST (UTC = + 5), count, minid, maxid";

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
