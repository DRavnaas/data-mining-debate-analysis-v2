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
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import twitter4j.GeoLocation;
import twitter4j.Query;
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
public class SearchTweets {
      public static void main(String[] args) {
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
     	
     	/* Perform basic search for the search string */
    	
     	try{
        Query query = formQuery(searchString);
		QueryResult result = search(twitter,query);
     	} catch (TwitterException te) {
            te.printStackTrace();
            System.out.println("Failed to do basic search tweets: " + te.getMessage());
            System.exit(-1);
        }
   /*    	
     	try{
          	Query query = formQuery(searchString);
     		QueryResult advancedResult = batchSearch(twitter,query);
		
     	} catch (TwitterException te) {
            te.printStackTrace();
            System.out.println("Failed to do advanced search tweets: " + te.getMessage());
            System.exit(-1);
        }
 */    	
			
}


private static Query formQuery(String searchString) {
	long maxId = 698657892912726016L;
	long sinceId = 688657892912720000L; ;
 	String since = "2015-02-15";
 	String until = "2015-02-17";
 	int count = 100;

 	Query query = new Query(searchString);
	query.setCount(count);
    //query.setSince(since);
    //query.setUntil(until);
    //query.setResultType(Query.ResultType.recent);
	
	query.setMaxId(maxId);
	query.setSinceId(sinceId);
	
	return query;
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
 
// This method performs the basic search on given string and returns the reults  
 private static QueryResult search(Twitter twitter,Query query) throws TwitterException {  	 
         QueryResult result;
         do {
             result = twitter.search(query);
             List<Status> tweets = result.getTweets();
             
             for (Status tweet : tweets) {
                  System.out.println("id: " + tweet.getId() + "Time: " + tweet.getCreatedAt() );
                  System.out.println(tweet.getText());
             }
         } while ((query = result.nextQuery()) != null);
         System.exit(0);
         
		return result;
}

//This method performs the basic search on given string and returns the reults  
private static QueryResult batchSearch(Twitter twitter,Query query) throws TwitterException{
	
	  QueryResult result = null;
   	  long lastID = query.getMaxId();
	  int numberOfTweets = query.getCount();

	  ArrayList<Status> tweets = new ArrayList<Status>();
	  while (tweets.size () < numberOfTweets) {
	    if (numberOfTweets - tweets.size() > 100)
	      query.setCount(100);
	    else 
	      query.setCount(numberOfTweets - tweets.size());
	    try {
	       result = twitter.search(query);
	      tweets.addAll(result.getTweets());
	      System.out.println("Gathered " + tweets.size() + " tweets");
	      for (Status t: tweets) 
	        if(t.getId() < lastID) 
	        	lastID = t.getId();

	    }

	    catch (TwitterException te) {
	    	 System.out.println("Couldn't connect: " + te);
	    }; 
	    query.setMaxId(lastID-1);
	  }

	  for (int i = 0; i < tweets.size(); i++) {
	    Status t = (Status) tweets.get(i);

	    GeoLocation loc = t.getGeoLocation();

	    String user = t.getUser().getScreenName();
	    String msg = t.getText();
	    String time = "";
	    if (loc!=null) {
	      Double lat = t.getGeoLocation().getLatitude();
	      Double lon = t.getGeoLocation().getLongitude();
	      System.out.println(i + " USER: " + user + " wrote: " + msg + " located at " + lat + ", " + lon);
	    } 
	    else 
	    	 System.out.println(i + " USER: " + user + " wrote: " + msg);
	  }
	  
	 return  result;
	}
}



