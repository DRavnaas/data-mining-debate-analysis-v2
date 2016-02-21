/*
 * Copyright 2007 Yusuke Yamamoto
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package twitter4j.examples;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

import twitter4j.Query;
import twitter4j.Query.ResultType;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.conf.ConfigurationBuilder;

/**
 * https://raw.githubusercontent.com/yusuke/twitter4j/master/twitter4j-examples/src/main/java/twitter4j/examples/search/SearchTweets.java
 * @author Yusuke Yamamoto - yusuke at mac.com
 * @author GOPredictors
 * @since Twitter4J 2.1.7
 */
public class SearchTweetsExample {
      public static void main(String[] args) {
    	  
    	  // Possible config (args or properties file) = two dates, 
    	  
      	//File file = new File("/Users/Yogi/git/data-mining-debate-analysis/config/
    	// Look for config file using relative dir - in tweetCrawler/config folder 
    	File file = new File("config/twitter4j.properties");
    	System.out.println("Reading config properties from:");
    	System.out.println(file.getAbsolutePath());
    	if (!file.exists())
    	{
    		System.out.println("No configuration file found");
            System.exit(-1);
    	}
    	
    	Twitter twitter = null;
    	
    	// Get Twitter Instance
		try {
			twitter = getTwitterInstance(file);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
    	
     	String searchString = "#GOPdebate cruz";
     	
     	// Perform basic search for the search string
     	
     	try{
		QueryResult result = basicSearch(twitter,searchString);
     	} catch (TwitterException te) {
            te.printStackTrace();
            System.out.println("Failed to search tweets: " + te.getMessage());
            System.exit(-1);
        }
        
     	System.exit(0);
			
    }


// This method reads the authentication properties from input file, creates 
 // ConfigurationBuilder from these properties and returns twitter instance
 // created using the configuration
 private static Twitter getTwitterInstance(File file) throws IOException {

        Properties prop = new Properties();
        InputStream is = null;
        
        is = new FileInputStream(file);
        prop.load(is);
     	
    	ConfigurationBuilder cb = new ConfigurationBuilder();
    	cb.setDebugEnabled(true)
    	  .setOAuthConsumerKey(prop.getProperty("oauth.consumerKey"))
    	  .setOAuthConsumerSecret(prop.getProperty("oauth.consumerSecret"))
    	  .setOAuthAccessToken(prop.getProperty("oauth.accessToken"))
    	  .setOAuthAccessTokenSecret(prop.getProperty("oauth.accessTokenSecret"));
    	TwitterFactory tf = new TwitterFactory(cb.build());
    	Twitter twitter = tf.getInstance();
    	
       return twitter;
	}
 
// This methid returns the reults for 
 private static QueryResult basicSearch(Twitter twitter,String searchString) throws TwitterException {  	 
         Query query = new Query(searchString);
         query.count(15);
         
         int curCount = 0;
         int maxCount = 150;
         
         query.resultType(ResultType.recent);
         query.since("2016-02-12");  // >=  this date
         query.until("2016-02-13");  // < this date
         
         QueryResult result = null;
         do {
        	 // ensure we aren't about to hit our rate limit.
             if (curCount >= maxCount-1)
             {
            	 // We are at the rate limit...
            	 // TODO: could sleep here instead or ...
            	 break;
             }
             if (curCount + query.getCount() > maxCount-1)
             {
            	 // Nearing rate limit.
            	 // Set the last batch size
            	 // TODO: double check, might be off by one here.
            	 query.count(maxCount - 1 - curCount);
             }
        	 
             result = twitter.search(query);
             List<Status> tweets = result.getTweets();
             curCount = curCount + tweets.size();
             
             System.out.println("Number of tweets returned = " + tweets.size());
             
             for (Status tweet : tweets) {
                  System.out.println("@" + tweet.getUser().getScreenName() + " - " + tweet.getText() + tweet.getCreatedAt().toString());                  
                  // other interesting fields - created time, plus hashtag, url and user entities.
                  
                  // For each tweet - save to csv...
             }
         } while ((query = result.nextQuery()) != null);
         
		return result;   
     
	}

 
}
