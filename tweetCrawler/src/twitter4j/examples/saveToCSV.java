package twitter4j.examples;


import twitter4j.*;
import twitter4j.Query.ResultType;
import twitter4j.conf.ConfigurationBuilder;

import java.io.*;
import java.util.List;
import java.util.Properties;

import com.opencsv.*;

public final class saveToCSV {
    /**
     * Usage: java twitter4j.examples.json.SaveRawJSON
     *
     * @param args String[]
     */
    public static void main(String[] args) {
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
     		basicSearch(twitter,searchString);
     	} catch (TwitterException te) {
            te.printStackTrace();
            System.out.println("Failed to search tweets: " + te.getMessage());
            System.exit(-1);
        }
        
     	System.exit(0);
    }
 
 //Ran added*************************************************************************************************   
    public static void storeTweets(QueryResult queryResult) {
    	if (queryResult==null) return;
    	List<Status> tweets = queryResult.getTweets();
    	CSVWriter writer = null;
    	try {
    		new File("savedTweets").mkdir();
    		
    		//get target CSV file object, the file name is specify by query string, 
    		//replaced : with %~ in the string, since windows doesn't allow to have : in filename 
    		File targetCSVFile = new File("savedTweets/"+queryResult.getQuery().replaceAll(":", "%~")+".csv");
    		
    		//create or open target csv file, if target csv file already exists, new records will be append to that file.
    		writer = new CSVWriter(new FileWriter(targetCSVFile,true));
    		 
    		//for each tweet record, write one line to target csv
    		//for each record, stored createAt,TimeZone and whole tweet text
            for (Status status : tweets) {
            	 String[] tweetRecord={status.getCreatedAt().toString(), status.getUser().getTimeZone(),status.getText()};
                 writer.writeNext(tweetRecord);
            }
         } catch (IOException ioe) {
             ioe.printStackTrace();
             System.out.println("Failed to store tweets: " + ioe.getMessage());
         } finally {
             if (writer != null) {
                 try {
                	 writer.flush();
                	 writer.close();
                 } catch (IOException ignore) {
                 }
              }
           }
            
    }
    
 //*******************************************************************************************************************************
    
    
    
    
    
    
    
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
    
    private static void basicSearch(Twitter twitter,String searchString) throws TwitterException {  	 
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
            storeTweets(result);
            curCount = curCount + tweets.size();
            
            System.out.println("Number of tweets returned = " + tweets.size());
        } while ((query = result.nextQuery()) != null);
           
    
	}

}