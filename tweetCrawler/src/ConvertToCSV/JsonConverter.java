package ConvertToCSV;


import twitter4j.*;
import twitter4j.Query.ResultType;
import twitter4j.conf.ConfigurationBuilder;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import com.opencsv.*;

public class JsonConverter {
	private Hashtable<String,Status> aggregatedTable;
	private Hashtable<String,Status> duplicatedTable;
	private Hashtable<String,Integer> duplicatedCount;
	private Hashtable<String,String> aggregatedTableCandidate; 
	public int total_tweets;
	JsonConverter() {
		this.aggregatedTable = new Hashtable<String, Status>();
    	this.duplicatedTable = new Hashtable<String, Status>();
    	this.duplicatedCount = new Hashtable<String, Integer>();
    	this.aggregatedTableCandidate = new Hashtable<String,String>();
    	this.total_tweets = 0;
	}
    public static void main(String[] args) {
    	if (args.length != 2) {
    		System.out.println("Usage: JsonConverter JSON_source_dir output_csv");
    		System.exit(-1);
    	}

    	JsonConverter jc = new JsonConverter();
    	try {
    		File[] files = new File(args[0]).listFiles(new FilenameFilter() {
                 //@Override
                 public boolean accept(File dir, String name) {
                     return name.endsWith(".json");
                 }
             });
             for (File file : files) { 
            	 jc.aggregate(file,  false);
                 System.out.println("restored tweets from " + file.getName());
             }
             jc.saveToCSV(args[1]);
             System.out.println("total_tweets: " +jc.total_tweets);
             System.exit(0);
         } catch (Exception e) {
             e.printStackTrace();
             System.out.println("Failed to store tweets: " + e.getMessage());
             System.exit(-1);
         }
 
 }
    public static String  getCandidateFromFileName(String fileName) {
    	if (fileName.indexOf("Before")!=-1)
    		return fileName.substring(0, fileName.indexOf("Before"));
    	else if (fileName.indexOf("After")!=-1)
    		return fileName.substring(0, fileName.indexOf("After"));
    	else if (fileName.indexOf("During")!=-1)
    		return fileName.substring(0, fileName.indexOf("During"));
    	else return "no candidate found";
    }
    
    public void aggregate(File file,  boolean filterOutRetweet) throws Exception{
    	List<Status> restoredTweets = readLines(file);
   	 	String candidate = getCandidateFromFileName(file.getName());
   	 	this.total_tweets += restoredTweets.size();
        for (Status status : restoredTweets) {
            	 //if retweet need to be filtered out.
        	if ((filterOutRetweet)&& status.isRetweet())
            		 continue;
            	 //specify fields to save to CSV
            	 
            if (this.aggregatedTable.get(String.valueOf(status.getId()))== null) {
            	this.aggregatedTable.put(String.valueOf(status.getId()), status);
            	this.aggregatedTableCandidate.put(String.valueOf(status.getId()), candidate);
            }
            else if (this.duplicatedTable.get(String.valueOf(status.getId()))== null) {
            	this.duplicatedTable.put(String.valueOf(status.getId()), status);
            	this.duplicatedCount.put(String.valueOf(status.getId()), 1);
            }
            else {
            	this.duplicatedCount.put(String.valueOf(status.getId()),this.duplicatedCount.get(String.valueOf(status.getId())) + 1);
            }
       }
    }


    public void saveToCSV(String targetCSV) throws Exception{
    	CSVWriter writer = null;
    	CSVWriter duplicatedWriter = null;
    	try {
    		if (targetCSV.endsWith(".csv") || targetCSV.endsWith(".CSV"))
    			targetCSV = targetCSV.substring(0, targetCSV.length()-4);
    		File targetCSVFile = new File(targetCSV+".csv");

  
    		writer = new CSVWriter(new FileWriter(targetCSVFile,true));
    		String[] headers = {"tweet_id","candidate", "name","tweet_created","tweet_location","user_timezone","retweet_count","text"};
    		writer.writeNext(headers);
    		
    		
    		
    		SimpleDateFormat formatter = new SimpleDateFormat("MM/dd/yyyy HH:mm"); 
    		Set<String> keys = this.aggregatedTable.keySet();
            for (String key : keys) {
            	Status status = this.aggregatedTable.get(key);
				String[] tweetRecord={String.valueOf(status.getId()),
										this.aggregatedTableCandidate.get(key),
            			 				status.getUser().getName(), 
            			 				formatter.format(status.getCreatedAt()), 
            			 				status.getUser().getLocation(),
            			 				status.getUser().getTimeZone(),
            			 				String.valueOf(status.getRetweetCount()),
            			 				status.getText()};
            	writer.writeNext(tweetRecord);
            }
            int total_duplicates=0;
            if (this.duplicatedTable.size()!=0) {
	            File duplicatedCSV = new File(targetCSV+"_duplicated.csv");
	            duplicatedWriter = new CSVWriter(new FileWriter(duplicatedCSV,true));
	    		String[] duplicateHeaders = {"tweet_id","duplicate_count", "name","tweet_created","tweet_location","user_timezone","retweet_count","text"};
	    		duplicatedWriter.writeNext(duplicateHeaders);
	    		
	    		keys = this.duplicatedTable.keySet();
	            for (String key : keys) {	            	
	            	Status status = this.duplicatedTable.get(key);
	            	total_duplicates += this.duplicatedCount.get(key).intValue();
					String[] tweetRecord={String.valueOf(status.getId()),
											String.valueOf(this.duplicatedCount.get(key).intValue()),
	            			 				status.getUser().getName(), 
	            			 				formatter.format(status.getCreatedAt()), 
	            			 				status.getUser().getLocation(),
	            			 				status.getUser().getTimeZone(),
	            			 				String.valueOf(status.getRetweetCount()),
	            			 				status.getText()};
					duplicatedWriter.writeNext(tweetRecord);
	            }
	        }
            System.out.println(total_duplicates);
       	 	
         } finally {   	 
             if (writer != null) {
                 try {
                	 writer.flush();
                	 writer.close();
                 } catch (IOException ignore) {
                 }
              }
             if (duplicatedWriter != null) {
                 try {
                	 duplicatedWriter.flush();
                	 duplicatedWriter.close();
                 } catch (IOException ignore) {
                 }
              }
          }
            
    }
    
    private static List<Status> readLines(File fileName) throws Exception {
        FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null;
        List<Status> ret = new ArrayList<Status>();
        try {
            fis = new FileInputStream(fileName);
            isr = new InputStreamReader(fis, "UTF-8");
            br = new BufferedReader(isr);
            br.readLine();
            String nextLine = br.readLine();
            while(nextLine!=null) {
            	try {
					ret.add(TwitterObjectFactory.createStatus(nextLine));
				} catch (TwitterException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            	nextLine = br.readLine();
            }
         return ret;
            
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException ignore) {
                }
            }
            if (isr != null) {
                try {
                    isr.close();
                } catch (IOException ignore) {
                }
            }
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException ignore) {
                }
            }
        }
    }

}