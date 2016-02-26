package twitter4j.examples;

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
 * 
 */
 
import twitter4j.*;
import twitter4j.Query.ResultType;
import twitter4j.auth.OAuth2Token;
import twitter4j.conf.ConfigurationBuilder;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Properties;

public class BatchSearchRestartable {

	//	Set this to your actual CONSUMER KEY and SECRET for your application as given to you by dev.twitter.com
	private static String CONSUMER_KEY		= "--your key goes here--";
	private static String CONSUMER_SECRET 	= "--your secret goes here--";

	//	How many tweets to retrieve in every call to Twitter. 100 is the maximum allowed in the API
	private static final int TWEETS_PER_QUERY		= 100;

	//	This controls how many queries, maximum, we will make of Twitter before cutting off the results.
	//	You will retrieve up to MAX_QUERIES*TWEETS_PER_QUERY tweets.
	//
	//  If you set MAX_QUERIES high enough (e.g., over 450), you will undoubtedly hit your rate limits
	//  and you an see the program sleep until the rate limits reset
	private static int MAX_QUERIES			= 5;

	//	What we want to search for in this program.  Justin Bieber always returns as many results as you could
	//	ever want, so it's safe to assume we'll get multiple pages back...
	private static String SEARCH_TERM			= "GOPDebate AND Trump";

	private static int maxTweetsOverall = TWEETS_PER_QUERY * MAX_QUERIES;

	/**
	 * Replace newlines and tabs in text with escaped versions to making printing cleaner
	 *
	 * @param text	The text of a tweet, sometimes with embedded newlines and tabs
	 * @return		The text passed in, but with the newlines and tabs replaced
	 */
	public static String cleanText(String text)
	{
		text = text.replace("\n", "\\n");
		text = text.replace("\t", "\\t");

		return text;
	}


	/**
	 * Retrieve the "bearer" token from Twitter in order to make application-authenticated calls.
	 *
	 * This is the first step in doing application authentication, as described in Twitter's documentation at
	 * https://dev.twitter.com/docs/auth/application-only-auth
	 *
	 * Note that if there's an error in this process, we just print a message and quit.  That's a pretty
	 * dramatic side effect, and a better implementation would pass an error back up the line...
	 *
	 * @return	The oAuth2 bearer token
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
	        
	        CONSUMER_KEY = prop.getProperty("oauth.consumerKey");
	        CONSUMER_SECRET = prop.getProperty("oauth.consumerSecret");
	        //cb.setDebugEnabled(true)
//	          .setOAuthConsumerKey(prop.getProperty("oauth.consumerKey"))
//	          .setOAuthConsumerSecret(prop.getProperty("oauth.consumerSecret"))
//	          .setOAuthAccessToken(prop.getProperty("oauth.accessToken"))
//	          .setOAuthAccessTokenSecret(prop.getProperty("oauth.accessTokenSecret"));
//	        TwitterFactory tf = new TwitterFactory(cb.build());
	        
	        
		OAuth2Token token = null;
		//ConfigurationBuilder cb;

		//cb = new ConfigurationBuilder();
		cb.setApplicationOnlyAuthEnabled(true);

		cb.setOAuthConsumerKey(CONSUMER_KEY).setOAuthConsumerSecret(CONSUMER_SECRET);
 
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
	 * @return	Twitter4J Twitter object that's ready for API calls
	 * @throws IOException 
	 */
	public static Twitter getTwitter() throws IOException
	{
		OAuth2Token token;

		//	First step, get a "bearer" token that can be used for our requests
		token = getOAuth2Token();

		//	Now, configure our new Twitter object to use application authentication and provide it with
		//	our CONSUMER key and secret and the bearer token we got back from Twitter
		ConfigurationBuilder cb = new ConfigurationBuilder();

		cb.setApplicationOnlyAuthEnabled(true);

		cb.setOAuthConsumerKey(CONSUMER_KEY);
		cb.setOAuthConsumerSecret(CONSUMER_SECRET);

		cb.setOAuth2TokenType(token.getTokenType());
		cb.setOAuth2AccessToken(token.getAccessToken());

		//	And create the Twitter object!
		return new TwitterFactory(cb.build()).getInstance();
	}

	public static void main(String[] args) throws Exception
	{
            TweetBounds tb = new TweetBounds();
	    
	        // args:  [query [maxtweets [sinceid maxid]]]
	        if (args.length >= 1)
	        {
	            SEARCH_TERM = args[0];
	        }   

	        // Initialize since and max ids for this query, if we've stored them from previous searches.
                tb.initTweetBounds(SEARCH_TERM);

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

	            // If since and max are on the command line, use those
	            if (args.length >= 4)
	            {	                
	                tb.sinceID = Long.parseLong(args[2]);
	                tb.maxID = Long.parseLong(args[3]);	   
	                
	                System.out.println("Using cached since/max values for the query");
	            }	            
	        }        
       
	    
	    
		//	We're curious how many tweets, in total, we've retrieved.  Note that TWEETS_PER_QUERY is an upper limit,
		//	but Twitter can and often will retrieve far fewer tweets
		int	totalTweets = 0;

		//	This variable is the key to our retrieving multiple blocks of tweets.  In each batch of tweets we retrieve,
		//	we use this variable to remember the LOWEST tweet ID.  Tweet IDs are (java) longs, and they are roughly
		//	sequential over time.  Without setting the MaxId in the query, Twitter will always retrieve the most
		//	recent tweets.  Thus, to retrieve a second (or third or ...) batch of Tweets, we need to set the Max Id
		//	in the query to be one less than the lowest Tweet ID we've seen already.  This allows us to page backwards
		//	through time to retrieve additional blocks of tweets
		//long maxID = -1;

		Twitter twitter = getTwitter();

		//	Now do a simple search to show that the tokens work
		try
		{
			//	There are limits on how fast you can make API calls to Twitter, and if you have hit your limit
			//	and continue to make calls Twitter will get annoyed with you.  I've found that going past your
			//	limits now and then doesn't seem to be problematic, but if you have a program that keeps banging
			//	the API when you're not allowed you will eventually get shut down.
			//
			//	Thus, the proper thing to do is always check your limits BEFORE making a call, and if you have
			//	hit your limits sleeping until you are allowed to make calls again.
			//
			//	Every time you call the Twitter API, it tells you how many calls you have left, so you don't have
			//	to ask about the next call.  But before the first call, we need to find out whether we're already
			//	at our limit.

			//	This returns all the various rate limits in effect for us with the Twitter API
			Map<String, RateLimitStatus> rateLimitStatus = twitter.getRateLimitStatus("search");

			//	This finds the rate limit specifically for doing the search API call we use in this program
			RateLimitStatus searchTweetsRateLimit = rateLimitStatus.get("/search/tweets");


			//	Always nice to see these things when debugging code...
			System.out.printf("You have %d calls remaining out of %d, Limit resets in %d seconds\n",
							  searchTweetsRateLimit.getRemaining(),
							  searchTweetsRateLimit.getLimit(),
							  searchTweetsRateLimit.getSecondsUntilReset());


			//	This is the loop that retrieve multiple blocks of tweets from Twitter
			for (int queryNumber=0;queryNumber < MAX_QUERIES; queryNumber++)
			{
				System.out.printf("\n\n!!! Starting loop %d\n\n", queryNumber);

				//	Do we need to delay because we've already hit our rate limits?
				if (searchTweetsRateLimit.getRemaining() == 0)
				{
					//	Yes we do, unfortunately ...
					System.out.printf("!!! Sleeping for %d seconds due to rate limits\n", searchTweetsRateLimit.getSecondsUntilReset());

					//	If you sleep exactly the number of seconds, you can make your query a bit too early
					//	and still get an error for exceeding rate limitations
					//
					// 	Adding two seconds seems to do the trick. Sadly, even just adding one second still triggers a
					//	rate limit exception more often than not.  I have no idea why, and I know from a Comp Sci
					//	standpoint this is really bad, but just add in 2 seconds and go about your business.  Or else.
					Thread.sleep((searchTweetsRateLimit.getSecondsUntilReset()+2) * 1000l);
				}

				Query q = new Query(SEARCH_TERM);			// Search for tweets that contains this term
				q.setCount(TWEETS_PER_QUERY);				// How many tweets, max, to retrieve
				q.setResultType(ResultType.recent);			// Get all tweets
				q.setLang("en");					// English language tweets, please

				//	If maxID is -1, then this is our first call and we do not want to tell Twitter what the maximum
				//	tweet id is we want to retrieve.  But if it is not -1, then it represents the lowest tweet ID
				//	we've seen, so we want to start at it-1 (if we start at maxID, we would see the lowest tweet
				//	a second time...
				if (tb.maxID != -1)
				{
					q.setMaxId(tb.maxID - 1);
				}
				
				// Use sinceID to say "don't back up beyond this id"
				// Might not be necessary?
				if (tb.sinceID != 0)
				{
				    q.setSinceId(tb.sinceID);
				}			    
				

				//	This actually does the search on Twitter and makes the call across the network
				QueryResult r = twitter.search(q);

				//	If there are NO tweets in the result set, it is Twitter's way of telling us that there are no
				//	more tweets to be retrieved.  Remember that Twitter's search index only contains about a week's
				//	worth of tweets, and uncommon search terms can run out of week before they run out of tweets
				if (r.getTweets().size() == 0)
				{
					break;			// Nothing? We must be done
				}


				// DELETE THIS LOOP AND REPLACE WITH SAVE
				//	loop through all the tweets and process them.  In this sample program, we just print them
				//	out, but in a real application you might save them to a database, a CSV file, do some
				//	analysis on them, whatever...
				for (Status s: r.getTweets())				// Loop through all the tweets...
				{
					//	Increment our count of tweets retrieved
					totalTweets++;

					//	Keep track of the lowest tweet ID.  If you do not do this, you cannot retrieve multiple
					//	blocks of tweets...
					if (tb.maxID == -1 || s.getId() < tb.maxID)
					{
						tb.maxID = s.getId();
					}

					//	Do something with the tweet....
					System.out.printf("At %s, @%-20s said:  %s\n",
									  s.getCreatedAt().toString(),
									  s.getUser().getScreenName(),
									  cleanText(s.getText()));

					
				}

				
				
			
				
				// STORE TWEETS HERE?

                                // Uncomment this so we have the total tweet count
                                // totalTweets = totalTweets + r.getTweets.size();

				
				// Once we have successfully stored our tweets...
				tb.saveTweetBounds();
				

				//	As part of what gets returned from Twitter when we make the search API call, we get an updated
				//	status on rate limits.  We save this now so at the top of the loop we can decide whether we need
				//	to sleep or not before making the next call.
				searchTweetsRateLimit = r.getRateLimitStatus();
			}

		}
		catch (Exception e)
		{
			//	Catch all -- you're going to read the stack trace and figure out what needs to be done to fix it
			System.out.println("That didn't work well...wonder why?");

			e.printStackTrace();

		}

		System.out.printf("\n\nA total of %d tweets retrieved\n", totalTweets);
		//	That's all, folks!

    }
}
