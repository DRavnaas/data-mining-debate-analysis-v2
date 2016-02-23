package twitter4j.examples;


import twitter4j.*;
import twitter4j.Query.ResultType;
import twitter4j.conf.ConfigurationBuilder;

import java.io.*;
import java.util.Hashtable;
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
    	
     	String searchString = "weahter";
     	
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
    
    //input: queryResult, targetcsv filename, flag for if retweets need to be filtered out
    public static void storeTweets(QueryResult queryResult, String targetCSV, boolean filterOutRetweet) {
    	if (queryResult==null) return;
    	List<Status> tweets = queryResult.getTweets();
    	CSVWriter writer = null;
    	CSVReader reader = null;
    	try {
    		new File("savedTweets").mkdir();
    		
    		//get target CSV file object, the file name is specify by parameter targetCSV, 
    		//It will also check if there is already a ".csv" in the giving csv file name. 
    		if (targetCSV.endsWith(".csv") || targetCSV.endsWith(".CSV"))
    			targetCSV = targetCSV.substring(0, targetCSV.length()-4);
    		File targetCSVFile = new File("savedTweets/"+targetCSV+".csv");
    		
    		
    		
    		//lookup table for avoiding duplicates
    		Hashtable<String, String[]> lookupTable = new Hashtable<String, String[]>();
    		
    		
    		
    		//if the target csv file exists, read each recorder and store it to a hashtable to avoid duplicates
    		//an existing tweet record will be a string array {tweetID, username, creattime, timezone, tweettext}
    		//tweetID is used as key in storing to hashtable. 
    		if (targetCSVFile.exists()) {
    			reader = new CSVReader(new FileReader(targetCSVFile));
    		
    			String [] nextLine;
	    	    while ((nextLine = reader.readNext()) != null) {
	    	        // nextLine[] is an array of values from the line
	    	    	lookupTable.put(nextLine[0], nextLine);
	    	    }
	    	    reader.close();
	    	    reader = null;
    		}
    		
    		
    		
    		//open csv writer to add new tweet records to target csv file.
    		writer = new CSVWriter(new FileWriter(targetCSVFile,true));
    		
    		
    		 
    		//process query result, for each tweet record, if it doesn't already exist in the target csv file, write one line to target csv
    		//for each new record, stored tweetID, username createAt,TimeZone and whole tweet text
    		//added prefix "ID" to each tweet id, so it won't be shown as a large number in csv file
            for (Status status : tweets) {
            	 //if retweet need to be filtered out.
            	 if ((filterOutRetweet)&& status.isRetweet())
            		 continue;
            	 String[] tweetRecord={"ID"+String.valueOf(status.getId()),status.getUser().getName(), status.getCreatedAt().toString(), status.getUser().getTimeZone(),status.getText()};
            	 if (lookupTable.get(tweetRecord[0])== null) {
            		 lookupTable.put(tweetRecord[0], tweetRecord);
            		 writer.writeNext(tweetRecord);
            	 }
            }
         } catch (IOException ioe) {
             ioe.printStackTrace();
             System.out.println("Failed to store tweets: " + ioe.getMessage());
         } finally {
        	 if (reader != null) {
                 try {
                	 reader.close();
                 } catch (IOException ignore) {
                 }
              }
        	 
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
        query.since("2016-02-20");  // >=  this date
        query.until("2016-02-21");  // < this date
        
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
            storeTweets(result,"targetCSV", false);
            curCount = curCount + tweets.size();
            
            System.out.println("Number of tweets returned = " + tweets.size());
        } while ((query = result.nextQuery()) != null);
           
    
	}

}